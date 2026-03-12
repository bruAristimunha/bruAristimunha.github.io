(function () {
  function initResearchOverview() {
    var sections = document.querySelectorAll(".research-overview");
    if (!sections.length) return;

    Array.prototype.forEach.call(sections, function (section) {
      setupSection(section);
    });
  }

  function setupSection(section) {
    var dataEl = section.querySelector(".research-overview__data");
    var scrollEl = section.querySelector(".research-overview__scroll");
    var stageEl = section.querySelector(".research-overview__stage");
    var svgEl = section.querySelector(".research-overview__svg");
    var labelsEl = section.querySelector(".research-overview__labels");

    if (!dataEl || !scrollEl || !stageEl || !svgEl || !labelsEl) return;

    var parsed;
    try {
      parsed = JSON.parse(dataEl.textContent);
    } catch (error) {
      console.error("Failed to parse research overview data.", error);
      return;
    }

    var rawRoot = Array.isArray(parsed) ? parsed[0] : parsed;
    if (!rawRoot) return;

    var resizeHandler = debounce(function () {
      renderSection(section, rawRoot, scrollEl, stageEl, svgEl, labelsEl);
    }, 120);

    renderSection(section, rawRoot, scrollEl, stageEl, svgEl, labelsEl);

    if ("ResizeObserver" in window) {
      var observer = new ResizeObserver(resizeHandler);
      observer.observe(scrollEl);
      section._researchOverviewObserver = observer;
    } else {
      window.addEventListener("resize", resizeHandler);
    }
  }

  function renderSection(section, rawRoot, scrollEl, stageEl, svgEl, labelsEl) {
    var root = normalizeTree(rawRoot);
    computeNodeMetrics(root);
    var nodes = flattenTree(root);
    var disableLocalLinks = !!section.closest(".figure-export");
    var availableWidth = Math.max(scrollEl.clientWidth || 0, section.clientWidth || 0, 320);
    var settings = getLayoutSettings(availableWidth, section);

    if (disableLocalLinks) {
      nodes.forEach(function (node) {
        if (node.anchorUrl && node.primaryUrl === node.anchorUrl) {
          node.primaryUrl = null;
        }
      });
    }

    measureNodes(section, nodes, settings);
    setGlobalYearRange(nodes, settings);

    var maxLeafHeight = nodes.reduce(function (maxHeight, node) {
      return node.isLeaf ? Math.max(maxHeight, node.labelHeight || 0) : maxHeight;
    }, 0);
    var maxBranchHeight = nodes.reduce(function (maxHeight, node) {
      return !node.isLeaf ? Math.max(maxHeight, node.labelHeight || 0) : maxHeight;
    }, 0);

    settings.branchGap = Math.max(
      settings.branchGap,
      maxBranchHeight + settings.branchLabelOffset + settings.dotRadius + 28
    );
    settings.leafGap = Math.max(
      settings.leafGap,
      Math.max(maxLeafHeight * 0.52, 24) + settings.leafLabelOffset + settings.dotRadius
    );
    settings.topPadding = Math.max(
      settings.topPadding,
      (root.labelHeight || 0) + settings.rootLabelOffset + settings.rootDotRadius + settings.canvasPadding
    );

    computeSubtreeWidths(root, settings);

    var treeWidth = root.subtreeWidth + settings.leftPadding + settings.rightPadding;
    var stageWidth = Math.max(availableWidth, settings.minWidth, treeWidth);
    var offsetX = Math.max((stageWidth - treeWidth) / 2, 0);
    assignPositions(root, settings.leftPadding + offsetX, settings.topPadding, settings);

    var bounds = getTreeBounds(nodes, root, settings);
    shiftNodePositions(nodes, settings.canvasPadding - bounds.top);
    bounds = getTreeBounds(nodes, root, settings);

    var naturalStageHeight = bounds.bottom - bounds.top + settings.canvasPadding * 2;
    var stageHeight = Math.max(settings.minHeight, Math.ceil(naturalStageHeight));
    var centerShift = Math.max((stageHeight - naturalStageHeight) / 2, 0);
    if (centerShift > 0) {
      shiftNodePositions(nodes, centerShift);
    }

    stageEl.style.width = Math.ceil(stageWidth) + "px";
    stageEl.style.height = Math.ceil(stageHeight) + "px";

    svgEl.setAttribute("viewBox", "0 0 " + Math.ceil(stageWidth) + " " + Math.ceil(stageHeight));
    svgEl.setAttribute("width", Math.ceil(stageWidth));
    svgEl.setAttribute("height", Math.ceil(stageHeight));

    renderSvg(svgEl, nodes, root, settings);
    renderLabels(labelsEl, nodes, settings);

    section.classList.remove("is-rendered");
    var raf = window.requestAnimationFrame || function (callback) {
      window.setTimeout(callback, 16);
    };
    raf(function () {
      section.classList.add("is-rendered");
    });
  }

  function normalizeTree(node, depth) {
    var currentDepth = typeof depth === "number" ? depth : 0;
    var normalized = {
      id: "research-node-" + Math.random().toString(36).slice(2, 10) + "-" + currentDepth,
      label: node.label || "",
      depth: currentDepth,
      paperId: node.paper_id || null,
      note: node.note || null,
      url: node.url || null,
      refUrl: node.ref_url || null,
      refLabel: node.ref_label || null,
      year: extractYear(node.note || ""),
      children: [],
    };

    normalized.children = (node.children || []).map(function (child) {
      return normalizeTree(child, currentDepth + 1);
    });
    normalized.isLeaf = normalized.children.length === 0;
    normalized.anchorUrl = normalized.paperId ? "#paper-" + normalized.paperId.toLowerCase() : null;
    normalized.primaryUrl = normalized.url || normalized.refUrl || normalized.anchorUrl || null;
    normalized.primaryText = buildPrimaryLeafText(normalized);

    return normalized;
  }

  function extractYear(text) {
    var match = String(text || "").match(/\b(19|20)\d{2}\b/);
    return match ? parseInt(match[0], 10) : null;
  }

  function computeNodeMetrics(node) {
    if (node.isLeaf) {
      node.leafCount = 1;
      node.yearCount = typeof node.year === "number" ? 1 : 0;
      node.yearTotal = typeof node.year === "number" ? node.year : 0;
      node.avgYear = typeof node.year === "number" ? node.year : null;
      node.branchDepth = 0;
      return node;
    }

    var leafCount = 0;
    var yearCount = 0;
    var yearTotal = 0;
    var branchDepth = 0;

    node.children.forEach(function (child) {
      computeNodeMetrics(child);
      leafCount += child.leafCount || 0;
      yearCount += child.yearCount || 0;
      yearTotal += child.yearTotal || 0;
      branchDepth = Math.max(branchDepth, child.branchDepth || 0);
    });

    node.leafCount = leafCount;
    node.yearCount = yearCount;
    node.yearTotal = yearTotal;
    node.avgYear = yearCount ? yearTotal / yearCount : null;
    node.branchDepth = branchDepth + 1;
    return node;
  }

  function buildPrimaryLeafText(node) {
    var text = node.note || node.label || "";
    if (node.refUrl && !node.url && node.refLabel) text += " (" + node.refLabel + ")";
    return text;
  }

  function flattenTree(root) {
    var nodes = [];

    function visit(node) {
      nodes.push(node);
      node.children.forEach(visit);
    }

    visit(root);
    return nodes;
  }

  function getLayoutSettings(viewportWidth, section) {
    var isCompact = section.classList.contains("research-overview--compact");
    if (viewportWidth < 700) {
      return {
        topPadding: isCompact ? 62 : 72,
        bottomPadding: isCompact ? 38 : 44,
        leftPadding: isCompact ? 24 : 30,
        rightPadding: isCompact ? 24 : 30,
        depthGap: isCompact ? 144 : 156,
        branchWidth: isCompact ? 112 : 128,
        leafWidth: isCompact ? 84 : 146,
        gap: isCompact ? 10 : 26,
        minWidth: isCompact ? 860 : 980,
        minHeight: isCompact ? 360 : 380,
        dotRadius: 5,
        rootDotRadius: 6,
        rootLabelOffset: 16,
        branchLabelOffset: 12,
        leafLabelOffset: 20,
        branchGap: isCompact ? 134 : 144,
        leafGap: isCompact ? 120 : 128,
        sizeGapBoost: isCompact ? 54 : 60,
        structureGapBoost: isCompact ? 28 : 30,
        centralGapBoost: isCompact ? 18 : 20,
        yearGapAdjust: isCompact ? 10 : 12,
        canvasPadding: 18,
        leafAngle: isCompact ? 34 : 38,
      };
    }

    if (viewportWidth < 1024) {
      return {
        topPadding: isCompact ? 72 : 84,
        bottomPadding: isCompact ? 42 : 48,
        leftPadding: isCompact ? 32 : 40,
        rightPadding: isCompact ? 32 : 40,
        depthGap: isCompact ? 166 : 182,
        branchWidth: isCompact ? 118 : 150,
        leafWidth: isCompact ? 74 : 176,
        gap: isCompact ? 10 : 34,
        minWidth: isCompact ? 940 : 1360,
        minHeight: isCompact ? 420 : 420,
        dotRadius: 5.5,
        rootDotRadius: 6.5,
        rootLabelOffset: 18,
        branchLabelOffset: 14,
        leafLabelOffset: 22,
        branchGap: isCompact ? 144 : 166,
        leafGap: isCompact ? 126 : 140,
        sizeGapBoost: isCompact ? 62 : 72,
        structureGapBoost: isCompact ? 30 : 36,
        centralGapBoost: isCompact ? 20 : 24,
        yearGapAdjust: isCompact ? 12 : 14,
        canvasPadding: 20,
        leafAngle: isCompact ? 34 : 38,
      };
    }

    return {
      topPadding: isCompact ? 76 : 96,
      bottomPadding: isCompact ? 46 : 56,
      leftPadding: isCompact ? 40 : 54,
      rightPadding: isCompact ? 40 : 54,
      depthGap: isCompact ? 176 : 208,
      branchWidth: isCompact ? 120 : 182,
      leafWidth: isCompact ? 64 : 220,
      gap: isCompact ? 8 : 42,
      minWidth: isCompact ? 1160 : 1960,
      minHeight: isCompact ? 500 : 620,
      dotRadius: 6,
      rootDotRadius: 7,
      rootLabelOffset: 20,
      branchLabelOffset: 16,
      leafLabelOffset: 24,
      branchGap: isCompact ? 154 : 184,
      leafGap: isCompact ? 136 : 154,
      sizeGapBoost: isCompact ? 70 : 88,
      structureGapBoost: isCompact ? 34 : 42,
      centralGapBoost: isCompact ? 22 : 28,
      yearGapAdjust: isCompact ? 12 : 16,
      canvasPadding: 22,
      leafAngle: isCompact ? 34 : 38,
    };
  }

  function setGlobalYearRange(nodes, settings) {
    var years = nodes
      .map(function (node) {
        return node.year;
      })
      .filter(function (year) {
        return typeof year === "number";
      });

    settings.globalMinYear = years.length ? Math.min.apply(Math, years) : null;
    settings.globalMaxYear = years.length ? Math.max.apply(Math, years) : null;
  }

  function measureNodes(section, nodes, settings) {
    var measureLayer = document.createElement("div");
    measureLayer.className = "research-overview__measure";
    section.appendChild(measureLayer);

    nodes.forEach(function (node) {
      var labelEl = createLabelElement(node);
      labelEl.classList.add("is-measuring");
      labelEl.style.maxWidth = (node.isLeaf ? settings.leafWidth : settings.branchWidth) + "px";
      measureLayer.appendChild(labelEl);
      node.labelWidth = Math.ceil(labelEl.offsetWidth);
      node.labelHeight = Math.ceil(labelEl.offsetHeight);
    });

    section.removeChild(measureLayer);
  }

  function computeSubtreeWidths(node, settings) {
    if (node.isLeaf) {
      node.subtreeWidth = node.labelWidth || 0;
      return node.subtreeWidth;
    }

    var childrenWidth = 0;
    node.children.forEach(function (child, index) {
      childrenWidth += computeSubtreeWidths(child, settings);
      if (index < node.children.length - 1) {
        childrenWidth += settings.gap;
      }
    });

    node.subtreeWidth = Math.max(childrenWidth, node.labelWidth || 0);
    return node.subtreeWidth;
  }

  function assignPositions(node, left, currentY, settings) {
    node.x = left + node.subtreeWidth / 2;
    node.y = currentY;

    if (node.isLeaf) return;

    var childrenWidth = 0;
    node.children.forEach(function (child, index) {
      childrenWidth += child.subtreeWidth;
      if (index < node.children.length - 1) {
        childrenWidth += settings.gap;
      }
    });

    var childLeft = left + (node.subtreeWidth - childrenWidth) / 2;
    node.children.forEach(function (child, index) {
      var childGap = getChildVerticalGap(node, child, index, settings);
      assignPositions(child, childLeft, currentY + childGap, settings);
      childLeft += child.subtreeWidth + settings.gap;
    });
  }

  function getChildVerticalGap(parent, child, index, settings) {
    var baseGap = child.isLeaf ? settings.leafGap : settings.branchGap;
    var centeredFactor = 1 - Math.abs(getCenteredFactor(index, parent.children.length));
    var sizeFactor =
      parent.leafCount && parent.leafCount > 1 ? (child.leafCount - 1) / (parent.leafCount - 1) : 0;
    var structureFactor =
      parent.branchDepth && parent.branchDepth > 1 ? (child.branchDepth || 0) / (parent.branchDepth - 1) : 0;
    var yearFactor = getYearFactor(child.avgYear, settings.globalMinYear, settings.globalMaxYear);

    return Math.max(
      baseGap * 0.88,
      baseGap +
        sizeFactor * settings.sizeGapBoost +
        structureFactor * settings.structureGapBoost +
        centeredFactor * settings.centralGapBoost -
        yearFactor * settings.yearGapAdjust
    );
  }

  function getCenteredFactor(index, length) {
    if (length <= 1) return 0;
    var midpoint = (length - 1) / 2;
    return (index - midpoint) / Math.max(midpoint, 1);
  }

  function getYearFactor(year, minYear, maxYear) {
    if (typeof year !== "number" || typeof minYear !== "number" || typeof maxYear !== "number") {
      return 0;
    }
    if (minYear === maxYear) return 0;
    return ((year - minYear) / (maxYear - minYear)) * 2 - 1;
  }

  function getTreeBounds(nodes, root, settings) {
    var bounds = {
      top: Infinity,
      bottom: -Infinity,
    };
    var leafAngle = (settings.leafAngle || 38) * (Math.PI / 180);

    nodes.forEach(function (node) {
      var radius = node === root ? settings.rootDotRadius : settings.dotRadius;
      bounds.top = Math.min(bounds.top, node.y - radius);
      bounds.bottom = Math.max(bounds.bottom, node.y + radius);

      if (node.isLeaf) {
        var leafTop = node.y + settings.leafLabelOffset;
        var rotatedHeight =
          Math.abs(Math.sin(leafAngle)) * (node.labelWidth || 0) +
          Math.abs(Math.cos(leafAngle)) * (node.labelHeight || 0);
        bounds.bottom = Math.max(bounds.bottom, leafTop + rotatedHeight);
      } else {
        var labelOffset = node.depth === 0 ? settings.rootLabelOffset : settings.branchLabelOffset;
        bounds.top = Math.min(bounds.top, node.y - labelOffset - (node.labelHeight || 0));
      }
    });

    if (!isFinite(bounds.top) || !isFinite(bounds.bottom)) {
      bounds.top = 0;
      bounds.bottom = 0;
    }

    return bounds;
  }

  function shiftNodePositions(nodes, deltaY) {
    if (!deltaY) return;
    nodes.forEach(function (node) {
      node.y += deltaY;
    });
  }

  function renderSvg(svgEl, nodes, root, settings) {
    while (svgEl.firstChild) {
      svgEl.removeChild(svgEl.firstChild);
    }

    var namespace = "http://www.w3.org/2000/svg";
    var linksGroup = document.createElementNS(namespace, "g");
    linksGroup.setAttribute("class", "research-overview__links");
    svgEl.appendChild(linksGroup);

    var nodesGroup = document.createElementNS(namespace, "g");
    nodesGroup.setAttribute("class", "research-overview__dots");
    svgEl.appendChild(nodesGroup);

    nodes.forEach(function (node) {
      node.children.forEach(function (child) {
        var path = document.createElementNS(namespace, "path");
        path.setAttribute("class", "research-overview__link");
        path.setAttribute("d", buildLinkPath(node, child));
        linksGroup.appendChild(path);
      });
    });

    nodes.forEach(function (node) {
      var circle = document.createElementNS(namespace, "circle");
      var classes = ["research-overview__dot"];
      if (node === root) {
        classes.push("research-overview__dot--root");
      } else if (node.isLeaf) {
        classes.push("research-overview__dot--leaf");
      } else {
        classes.push("research-overview__dot--branch");
      }

      circle.setAttribute("class", classes.join(" "));
      circle.setAttribute("cx", node.x);
      circle.setAttribute("cy", node.y);
      circle.setAttribute("r", node === root ? settings.rootDotRadius : settings.dotRadius);
      nodesGroup.appendChild(circle);
    });
  }

  function renderLabels(labelsEl, nodes, settings) {
    while (labelsEl.firstChild) {
      labelsEl.removeChild(labelsEl.firstChild);
    }

    nodes.forEach(function (node) {
      var labelEl = createLabelElement(node);
      labelEl.style.maxWidth = (node.isLeaf ? settings.leafWidth : settings.branchWidth) + "px";
      labelEl.style.left = node.x + "px";
      labelEl.style.top =
        (node.isLeaf ? node.y + settings.leafLabelOffset : node.y - (node.depth === 0 ? settings.rootLabelOffset : settings.branchLabelOffset)) +
        "px";

      labelsEl.appendChild(labelEl);
    });
  }

  function createLabelElement(node) {
    var wrapper = document.createElement("div");
    wrapper.className =
      "research-overview__label " +
      (node.isLeaf
        ? "research-overview__label--leaf"
        : node.depth === 0
          ? "research-overview__label--root"
          : "research-overview__label--branch");

    if (!node.isLeaf) {
      wrapper.textContent = node.label;
      return wrapper;
    }

    if (node.paperId) {
      var paperId = document.createElement("span");
      paperId.className = "research-overview__paper-id";
      paperId.textContent = "[" + node.paperId + "] ";
      wrapper.appendChild(paperId);
    }

    var primaryEl = document.createElement(node.primaryUrl ? "a" : "span");
    primaryEl.className = "research-overview__paper-link";
    primaryEl.textContent = node.primaryText;

    if (node.primaryUrl) {
      primaryEl.setAttribute("href", node.primaryUrl);
      if (isExternalUrl(node.primaryUrl)) {
        primaryEl.setAttribute("target", "_blank");
        primaryEl.setAttribute("rel", "noopener noreferrer");
      } else {
        primaryEl.setAttribute("target", "_self");
      }
    } else {
      primaryEl.classList.add("is-static");
    }

    wrapper.appendChild(primaryEl);

    if (node.refUrl && node.url && node.refLabel) {
      var spacer = document.createTextNode(" ");
      var refEl = document.createElement("a");
      refEl.className = "research-overview__ref-link";
      refEl.setAttribute("href", node.refUrl);
      refEl.setAttribute("target", "_blank");
      refEl.setAttribute("rel", "noopener noreferrer");
      refEl.textContent = "(" + node.refLabel + ")";
      wrapper.appendChild(spacer);
      wrapper.appendChild(refEl);
    }

    return wrapper;
  }

  function buildLinkPath(source, target) {
    var bend = Math.max((target.y - source.y) * 0.55, 48);
    return [
      "M",
      round(source.x),
      round(source.y),
      "C",
      round(source.x),
      round(source.y + bend),
      round(target.x),
      round(target.y - bend),
      round(target.x),
      round(target.y),
    ].join(" ");
  }

  function round(value) {
    return Math.round(value * 10) / 10;
  }

  function isExternalUrl(url) {
    return /^(https?:)?\/\//.test(url);
  }

  function debounce(fn, delay) {
    var timeoutId = null;
    return function () {
      var args = arguments;
      clearTimeout(timeoutId);
      timeoutId = window.setTimeout(function () {
        fn.apply(null, args);
      }, delay);
    };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initResearchOverview);
  } else {
    initResearchOverview();
  }
})();
