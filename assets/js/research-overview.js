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
    var nodes = flattenTree(root);
    var availableWidth = Math.max(scrollEl.clientWidth || 0, section.clientWidth || 0, 320);
    var settings = getLayoutSettings(availableWidth);

    measureNodes(section, nodes, settings);

    var leftPadding = Math.max(
      settings.leftPadding,
      (root.labelWidth || 0) + settings.rootLabelOffset + 12
    );
    var maxLeafWidth = nodes.reduce(function (maxWidth, node) {
      return node.isLeaf ? Math.max(maxWidth, node.labelWidth || 0) : maxWidth;
    }, 0);
    var maxDepth = nodes.reduce(function (depth, node) {
      return Math.max(depth, node.depth);
    }, 0);

    computeSubtreeHeights(root, settings);

    var treeWidth =
      leftPadding +
      maxDepth * settings.depthGap +
      settings.leafLabelOffset +
      maxLeafWidth +
      settings.rightPadding;
    var treeHeight = root.subtreeHeight + settings.topPadding + settings.bottomPadding;
    var stageWidth = Math.max(availableWidth, treeWidth);
    var stageHeight = Math.max(settings.minHeight, treeHeight);
    var offsetX = Math.max((stageWidth - treeWidth) / 2, 0);
    var offsetY = Math.max((stageHeight - treeHeight) / 2, 0);

    assignPositions(root, settings.topPadding + offsetY, leftPadding + offsetX, settings);

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

  function buildPrimaryLeafText(node) {
    var text = node.label || "";
    if (node.note) text += " (" + node.note + ")";
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

  function getLayoutSettings(viewportWidth) {
    if (viewportWidth < 700) {
      return {
        topPadding: 28,
        bottomPadding: 32,
        leftPadding: 168,
        rightPadding: 42,
        depthGap: 164,
        branchWidth: 168,
        leafWidth: 250,
        gap: 18,
        minHeight: 380,
        dotRadius: 5,
        rootDotRadius: 6,
        rootLabelOffset: 18,
        branchLabelOffset: 14,
        leafLabelOffset: 16,
      };
    }

    if (viewportWidth < 1024) {
      return {
        topPadding: 32,
        bottomPadding: 36,
        leftPadding: 196,
        rightPadding: 52,
        depthGap: 188,
        branchWidth: 212,
        leafWidth: 312,
        gap: 22,
        minHeight: 420,
        dotRadius: 5.5,
        rootDotRadius: 6.5,
        rootLabelOffset: 20,
        branchLabelOffset: 16,
        leafLabelOffset: 18,
      };
    }

    return {
      topPadding: 38,
      bottomPadding: 44,
      leftPadding: 232,
      rightPadding: 76,
      depthGap: 220,
      branchWidth: 248,
      leafWidth: 420,
      gap: 24,
      minHeight: 520,
      dotRadius: 6,
      rootDotRadius: 7,
      rootLabelOffset: 22,
      branchLabelOffset: 18,
      leafLabelOffset: 20,
    };
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

  function computeSubtreeHeights(node, settings) {
    if (node.isLeaf) {
      node.subtreeHeight = node.labelHeight || 0;
      return node.subtreeHeight;
    }

    var childrenHeight = 0;
    node.children.forEach(function (child, index) {
      childrenHeight += computeSubtreeHeights(child, settings);
      if (index < node.children.length - 1) {
        childrenHeight += settings.gap;
      }
    });

    node.subtreeHeight = Math.max(childrenHeight, node.labelHeight || 0);
    return node.subtreeHeight;
  }

  function assignPositions(node, top, baseX, settings) {
    node.x = baseX + node.depth * settings.depthGap;
    node.y = top + node.subtreeHeight / 2;

    if (node.isLeaf) return;

    var childrenHeight = 0;
    node.children.forEach(function (child, index) {
      childrenHeight += child.subtreeHeight;
      if (index < node.children.length - 1) {
        childrenHeight += settings.gap;
      }
    });

    var childTop = top + (node.subtreeHeight - childrenHeight) / 2;
    node.children.forEach(function (child) {
      assignPositions(child, childTop, baseX, settings);
      childTop += child.subtreeHeight + settings.gap;
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
      labelEl.style.top = node.y + "px";

      if (node.isLeaf) {
        labelEl.style.left = node.x + settings.leafLabelOffset + "px";
      } else if (node.depth === 0) {
        labelEl.style.left = node.x - settings.rootLabelOffset + "px";
      } else {
        labelEl.style.left = node.x - settings.branchLabelOffset + "px";
      }

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
    var bend = Math.max((target.x - source.x) * 0.55, 48);
    return [
      "M",
      round(source.x),
      round(source.y),
      "C",
      round(source.x + bend),
      round(source.y),
      round(target.x - bend),
      round(target.y),
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
