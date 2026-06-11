---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

<span class='anchor' id='about-me'></span>

{% include_relative includes/intro.md %}

<!-- # 🔥 News
- *20XX*: &nbsp;🎉🎉 TO-DO... copy from my thesis report the event list... -->

<span class='anchor' id='journey'></span>

# 🗺️ Journey

<div class="journey" data-journey>
  <div class="journey__header">
    <div class="journey__eyebrow">Three Acts</div>
    <h2 class="journey__title">A research scientist, in motion</h2>
    <p class="journey__lede">
      Three countries, one through-line: building tools to decode signals from the brain. It started at a 2012 high-school science fair in Mato Grosso do Sul (I was sixteen), went through a cotutelle PhD between Paris-Saclay and UFABC, and continues today at Yneuro, with an honorary affiliation at UC San Diego INC.
    </p>
  </div>

  <div class="journey__rail" role="tablist" aria-label="Three acts">
    <button type="button" role="tab" class="journey__rail-tab" aria-selected="false" data-journey-tab="now">
      <span class="journey__rail-kicker">Act 01 — Now</span>
      <span class="journey__rail-period">2026 →</span>
    </button>
    <button type="button" role="tab" class="journey__rail-tab is-active" aria-selected="true" data-journey-tab="before">
      <span class="journey__rail-kicker">Act 02 — Before</span>
      <span class="journey__rail-period">2020 — 2026</span>
    </button>
    <button type="button" role="tab" class="journey__rail-tab" aria-selected="false" data-journey-tab="origin">
      <span class="journey__rail-kicker">Act 03 — Origin</span>
      <span class="journey__rail-period">2012 — 2020</span>
    </button>
  </div>

  <div class="journey__panels">
    <section class="journey__panel" data-journey-panel="now" role="tabpanel" aria-hidden="true">
      <header class="journey__panel-head">
        <div class="journey__panel-kicker">Act 01 — Now</div>
        <div class="journey__panel-period">2026 →</div>
        <h3 class="journey__panel-title">Research Scientist</h3>
        <div class="journey__panel-place">Yneuro 🇫🇷 · UC San Diego INC 🇺🇸 (Honorary)</div>
        <p class="journey__panel-summary">Research Scientist at Yneuro and Honorary Research Associate at UC San Diego (INC). Lead maintainer of <strong>Braindecode</strong> and <strong>MOABB</strong>. Lead organizer of the MLSP 2025 Special Session on Decoding the Brain Time Series.</p>
      </header>
      <ol class="journey__milestones">
        <li class="journey__ms is-highlight">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2026</span>
            <span class="journey__ms-label">Yneuro — Research Scientist</span>
          </div>
          <div class="journey__ms-note">Joined post-PhD to continue work on EEG decoding and foundation models for neural signals.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2025</span>
            <span class="journey__ms-label">MLSP 2025 Special Session</span>
          </div>
          <div class="journey__ms-note">Lead organizer of Decoding the Brain Time Series at IEEE MLSP.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2026 →</span>
            <span class="journey__ms-label">UC San Diego INC</span>
          </div>
          <div class="journey__ms-note">Honorary Research Associate at the Institute for Neural Computation.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2023 →</span>
            <span class="journey__ms-label">Reviewer for ML/neuro venues</span>
          </div>
          <div class="journey__ms-note">NeurIPS (×2), ICLR, ICML, JMLR, NeuroImage, Imaging Neuroscience, PeerJ CS, L4DC@ICLR.</div>
        </li>
      </ol>
    </section>

    <section class="journey__panel journey__panel--wide is-active" data-journey-panel="before" role="tabpanel" aria-hidden="false">
      <header class="journey__panel-head">
        <div class="journey__panel-kicker">Act 02 — Before</div>
        <div class="journey__panel-period">2020 — 2026</div>
        <h3 class="journey__panel-title">PhD in Computer Science (cotutelle)</h3>
        <div class="journey__panel-place">Université Paris-Saclay 🇫🇷 · UFABC 🇧🇷</div>
        <p class="journey__panel-summary">Cotutelle PhD: <em>Learning Structure In Electroencephalogram Using Deep Learning</em> (Paris-Saclay) / <em>Geração de Representações Compactas de Sinais EEG</em> (UFABC). Advisors: Sylvain Chevallier, Marie-Constance Corsi, Raphael Y. de Camargo. Sandwich period at King's College London with Walter H. L. Pinaya. Funded by INRIA (FR) and CAPES (BR).</p>
      </header>
      <ol class="journey__milestones">
        <li class="journey__ms is-highlight">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2026</span>
            <span class="journey__ms-label">PhD defense</span>
          </div>
          <div class="journey__ms-note">Cotutelle thesis defended February 2026 at Paris-Saclay & UFABC.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2024</span>
            <span class="journey__ms-label">Geometric Neural Network (JNE)</span>
          </div>
          <div class="journey__ms-note">Phase-space SPDNet for BCI-EEG decoding, in the <em>Journal of Neural Engineering</em> with Carrara, Corsi, Papadopoulo.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2024</span>
            <span class="journey__ms-label">MOABB benchmark study</span>
          </div>
          <div class="journey__ms-note">Largest EEG-based BCI reproducibility study for open science. With Chevallier, Carrara, Guetschel, et al.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2024</span>
            <span class="journey__ms-label">Euclidean alignment (JNE)</span>
          </div>
          <div class="journey__ms-note">Systematic evaluation of Euclidean alignment with deep learning for EEG decoding. Junqueira, Aristimunha, Chevallier, de Camargo.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2024</span>
            <span class="journey__ms-label">Alljoined dataset (CVPR-W)</span>
          </div>
          <div class="journey__ms-note">EEG-to-Image decoding dataset, CVPR 2024 Workshop on Data Curation in Medical Imaging.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2024</span>
            <span class="journey__ms-label">MOABB Zenodo release</span>
          </div>
          <div class="journey__ms-note">Mother of all BCI Benchmarks: software registry at INRIA, DOI 10.5281/zenodo.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2023</span>
            <span class="journey__ms-label">Synthetic Sleep EEG (NeurIPS DGM4H)</span>
          </div>
          <div class="journey__ms-note">Latent diffusion models for EEG generation, NeurIPS 2023 DGM4H Workshop (Spotlight).</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2023</span>
            <span class="journey__ms-label">Sleep-Energy (IEEE Access)</span>
          </div>
          <div class="journey__ms-note">Energy optimization for sleep stage scoring. With Bayerlein, Cardoso, Pinaya, de Camargo.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2023</span>
            <span class="journey__ms-label">IVA for Motor Imagery (ICASSP)</span>
          </div>
          <div class="journey__ms-note">Independent Vector Analysis on EEG-Based Motor Imagery Classification, ICASSP 2023.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2023</span>
            <span class="journey__ms-label">Braindecode registered</span>
          </div>
          <div class="journey__ms-note">Software registration with INRIA, V1.0 (01/08/2023).</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2023</span>
            <span class="journey__ms-label">Braindecode Code-Sprint</span>
          </div>
          <div class="journey__ms-note">Organized the European 2023 sprint.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2023</span>
            <span class="journey__ms-label">King's College London (sandwich)</span>
          </div>
          <div class="journey__ms-note">Visiting period under Walter H. L. Pinaya.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2023</span>
            <span class="journey__ms-label">Started Paris-Saclay leg</span>
          </div>
          <div class="journey__ms-note">Cotutelle PhD enrollment at Paris-Saclay (in addition to UFABC). INRIA scholarship.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2022</span>
            <span class="journey__ms-label">Glasgow / FGV intern</span>
          </div>
          <div class="journey__ms-note">Data Scientist intern at the University of Glasgow & Fundação Getúlio Vargas.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2021</span>
            <span class="journey__ms-label">FGV consultant</span>
          </div>
          <div class="journey__ms-note">Data Science consultant on an IDB-funded urban-data project (Waze car-accident detection in São Paulo). Stack: AWS, SQL, Python, Dash.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2020</span>
            <span class="journey__ms-label">PhD start (UFABC)</span>
          </div>
          <div class="journey__ms-note">Began PhD in Computer Science at UFABC under Raphael Y. de Camargo. CAPES scholarship.</div>
        </li>
      </ol>
    </section>

    <section class="journey__panel" data-journey-panel="origin" role="tabpanel" aria-hidden="true">
      <header class="journey__panel-head">
        <div class="journey__panel-kicker">Act 03 — Origin</div>
        <div class="journey__panel-period">2012 — 2020</div>
        <h3 class="journey__panel-title">From science fairs to undergrad</h3>
        <div class="journey__panel-place">IFMS · UCDB · UFMS · UFABC 🇧🇷</div>
        <p class="journey__panel-summary">The spark: a 2012 high-school science fair in Mato Grosso do Sul. From a Junior Scientific Initiation scholarship at IFMS, to a computer-vision internship at UCDB INOVISÃO lab, to two undergrad degrees at UFABC, with prizes and first-author papers along the way.</p>
      </header>
      <ol class="journey__milestones">
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2020</span>
            <span class="journey__ms-label">Double BSc graduation</span>
          </div>
          <div class="journey__ms-note">UFABC: BSc in Science & Technology (with parallel work toward Computer Science). 2nd best undergrad paper at ERAMIA-SP 2020.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2018</span>
            <span class="journey__ms-label">Scientific Initiation — Neuroscience</span>
          </div>
          <div class="journey__ms-note">CNPq fellow: functional brain connectivity via causality in time series. UFABC.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2016</span>
            <span class="journey__ms-label">Entered UFABC</span>
          </div>
          <div class="journey__ms-note">Bacharelado Interdisciplinar em Ciência e Tecnologia. Transferred from UFMS.</div>
        </li>
        <li class="journey__ms is-highlight">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2015</span>
            <span class="journey__ms-label">First conference papers</span>
          </div>
          <div class="journey__ms-note">Sibgrapi 2015 + Computer on the Beach: computer-vision work on bamboo-borer (<em>Dinoderus minutus</em>) detection from the UCDB INOVISÃO lab.</div>
        </li>
        <li class="journey__ms is-highlight">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2014</span>
            <span class="journey__ms-label">UCDB Computer Vision intern</span>
          </div>
          <div class="journey__ms-note">INOVISÃO lab under Prof. Hemerson Pistori: animal-behavior extraction from images. While still in high school.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2014</span>
            <span class="journey__ms-label">FEBRACE + ABRITEC + ABRIC awards</span>
          </div>
          <div class="journey__ms-note">4th place Biological Sciences (FEBRACE national fair); ABRITEC Distinction in Science Incentive; ABRIC Excellence in Scientific Initiation. For the Wi-Fi pest-repellence project.</div>
        </li>
        <li class="journey__ms is-highlight">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2012</span>
            <span class="journey__ms-label">First science fair (IFMS)</span>
          </div>
          <div class="journey__ms-note">SESC Prize at FETEC/MS. The very first project: photography use among IFMS high-school students.</div>
        </li>
      </ol>
    </section>
  </div>
</div>

# 🔬 Featured Manuscript

<section class="featured-paper">
  <a
    class="featured-paper__figure"
    href="{{ '/assets/pdfs/learning-aligned-eeg-representations.pdf' | relative_url }}"
    target="_blank"
    rel="noopener noreferrer"
  >
    <img
      src="{{ '/assets/images/publications/learning-aligned-eeg-representations.png' | relative_url }}"
      alt="Figure from Learning aligned EEG representations with subject-specific encoders"
      loading="lazy"
    >
  </a>
  <div class="featured-paper__body">
    <p class="featured-paper__eyebrow">EEG representation learning</p>
    <h2 class="featured-paper__title">Learning aligned EEG representations with subject-specific encoders</h2>
    <p class="featured-paper__authors">Bruna J. Lopes, Gabriel Schwartz, Sylvain Chevallier, Raphael Y. de Camargo, and Bruno Aristimunha</p>
    <p class="featured-paper__summary">Subject-specific encoders can internalise part of the alignment role usually handled by Euclidean Alignment. Cross-subject decoding performance holds, and head selection becomes the main remaining bottleneck.</p>
    <p class="featured-paper__links">
      <a href="{{ '/assets/pdfs/learning-aligned-eeg-representations.pdf' | relative_url }}" target="_blank" rel="noopener noreferrer">Read PDF</a>
      <a href="{{ '/assets/images/publications/learning-aligned-eeg-representations.png' | relative_url }}" target="_blank" rel="noopener noreferrer">Open figure</a>
    </p>
  </div>
</section>

# 🧭 Research Overview

{% include research_map.html %}

<p class="research-map__actions">
  <a
    class="research-map__action-link"
    href="{{ '/research-overview-figure.html' | relative_url }}"
    target="_blank"
    rel="noopener noreferrer"
  >
    Open standalone figure
  </a>
</p>


# 📝 Publications (Full List)

[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-profile-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?user=2Gd5gOQAAAAJ)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0001--5258--2995-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0000-0001-5258-2995)
[![OpenAlex](https://img.shields.io/badge/OpenAlex-profile-2D7DD2)](https://openalex.org/A5060466816)

{% assign pubs = site.data.publications %}
{% assign n_journal = pubs | where: "type", "journal" | size %}
{% assign n_conference = pubs | where: "type", "conference" | size %}
{% assign n_workshop = pubs | where: "type", "workshop" | size %}
{% assign n_abstract = pubs | where: "type", "abstract" | size %}
{% assign n_software = pubs | where: "type", "software" | size %}
{% assign n_report = pubs | where: "type", "report" | size %}
{% assign n_all = pubs | size %}
{% assign year_groups = pubs | group_by: "year" | sort: "name" | reverse %}
{% assign chart_groups = pubs | group_by: "year" | sort: "name" %}
<div class="pubtrack" data-pubtrack>
<div class="pubtrack__chart-wrap">
<div class="pubtrack__chart" role="img" aria-label="One square per publication, stacked by year from 2015 to 2026 and colored by type. Output rises sharply from 2023 onward.">
{% for y in (2015..2026) %}
{% assign ystr = y | append: "" %}
{% assign g = chart_groups | where: "name", ystr | first %}
{% if g %}
<a class="pubtrack__col" href="#pubyear-{{ y }}" data-chart-year="{{ y }}" aria-label="{{ y }}: {{ g.items | size }} publications">
<span class="pubtrack__stack">{% for pub in g.items %}<i class="pubtrack__cell pubtrack__cell--{{ pub.type }}" data-cell-type="{{ pub.type }}" title="{{ pub.title | escape }}"></i>{% endfor %}</span>
<span class="pubtrack__col-year">&rsquo;{{ ystr | slice: 2, 2 }}</span>
</a>
{% else %}
<span class="pubtrack__col pubtrack__col--empty" aria-hidden="true">
<span class="pubtrack__stack"><i class="pubtrack__tick"></i></span>
<span class="pubtrack__col-year">&rsquo;{{ ystr | slice: 2, 2 }}</span>
</span>
{% endif %}
{% endfor %}
</div>
<p class="pubtrack__chart-note">One square = one publication. Click a year to jump to it.</p>
</div>
<div class="pubtrack__filters" role="group" aria-label="Filter publications by type">
<button type="button" class="pubtrack__filter is-active" data-filter="all" aria-pressed="true">All <span class="pubtrack__count">{{ n_all }}</span></button>
<button type="button" class="pubtrack__filter pubtrack__filter--journal" data-filter="journal" aria-pressed="false">Journal <span class="pubtrack__count">{{ n_journal }}</span></button>
<button type="button" class="pubtrack__filter pubtrack__filter--conference" data-filter="conference" aria-pressed="false">Conference <span class="pubtrack__count">{{ n_conference }}</span></button>
<button type="button" class="pubtrack__filter pubtrack__filter--workshop" data-filter="workshop" aria-pressed="false">Workshop <span class="pubtrack__count">{{ n_workshop }}</span></button>
<button type="button" class="pubtrack__filter pubtrack__filter--abstract" data-filter="abstract" aria-pressed="false">Abstract <span class="pubtrack__count">{{ n_abstract }}</span></button>
<button type="button" class="pubtrack__filter pubtrack__filter--software" data-filter="software" aria-pressed="false">Software <span class="pubtrack__count">{{ n_software }}</span></button>
<button type="button" class="pubtrack__filter pubtrack__filter--report" data-filter="report" aria-pressed="false">Report <span class="pubtrack__count">{{ n_report }}</span></button>
</div>
<div class="pubtrack__years">
{% for group in year_groups %}
<div class="pubtrack__year anchor" id="pubyear-{{ group.name }}" data-year-group>
<p class="pubtrack__year-label">{{ group.name }}</p>
<div class="pubtrack__rows">
{% for pub in group.items %}
{% assign authors = pub.authors | replace: "Aristimunha, B.", "<strong>Aristimunha, B.</strong>" | replace: "Pinto, B. A.", "<strong>Pinto, B. A.</strong>" %}
{% if pub.links.pdf %}{% assign title_href = pub.links.pdf | relative_url %}{% elsif pub.links.arxiv %}{% assign title_href = pub.links.arxiv %}{% elsif pub.links.doi %}{% assign title_href = pub.links.doi %}{% elsif pub.links.site %}{% assign title_href = pub.links.site %}{% else %}{% assign title_href = "" %}{% endif %}
<article class="pubtrack__entry pubtrack__entry--{{ pub.type }} anchor" id="{{ pub.id }}" data-type="{{ pub.type }}">
{% if pub.figure %}{% capture pubfig %}<span class="pubfig" aria-hidden="true"><img src="{{ pub.figure | relative_url }}" alt="" loading="lazy"></span>{% endcapture %}{% else %}{% assign pubfig = "" %}{% endif %}
{% if title_href != "" %}<p class="pubtrack__title"><a href="{{ title_href }}">{{ pub.title }}{{ pubfig }}</a></p>{% else %}<p class="pubtrack__title">{{ pub.title }}{{ pubfig }}</p>{% endif %}
<p class="pubtrack__meta"><span class="pubtrack__type pubtrack__type--{{ pub.type }}">{{ pub.type }}</span><span class="pubtrack__venue">{{ pub.venue }}</span>{% if pub.status %}<span class="pubtrack__status">{{ pub.status | replace: "-", " " }}</span>{% endif %}<span class="pubtrack__num">P{{ pub.num }}</span></p>
<p class="pubtrack__authors">{{ authors }}</p>
{% if pub.links %}<p class="pubtrack__links">{% if pub.links.pdf %}<a href="{{ pub.links.pdf | relative_url }}">PDF<span class="pubtrack__glyph">&nbsp;&darr;</span></a>{% endif %}{% if pub.links.arxiv %}<a href="{{ pub.links.arxiv }}">arXiv<span class="pubtrack__glyph">&nbsp;&nearr;</span></a>{% endif %}{% if pub.links.doi %}<a href="{{ pub.links.doi }}">DOI<span class="pubtrack__glyph">&nbsp;&nearr;</span></a>{% endif %}{% if pub.links.site %}<a href="{{ pub.links.site }}">Site<span class="pubtrack__glyph">&nbsp;&nearr;</span></a>{% endif %}</p>{% endif %}
</article>
{% endfor %}
</div>
</div>
{% endfor %}
</div>
</div>


# 📖 Education

<div class="timeline-island">
  <ol class="timeline-island__list">
    <li class="timeline-island__item timeline-island__item--highlight">
      <div class="timeline-island__row">
        <span class="timeline-island__date">09/2020 – 02/2026</span>
        <span class="timeline-island__title">PhD in Computer Science</span>
      </div>
      <p class="timeline-island__detail">Cotutelle between <a href="https://www.universite-paris-saclay.fr/">Université Paris-Saclay</a> 🇫🇷 and <a href="https://www.ufabc.edu.br/">UFABC</a> 🇧🇷. Advised by <a href="https://sylvchev.github.io/">Sylvain Chevallier</a>, <a href="https://marieconstance-corsi.netlify.app/">Marie-Constance Corsi</a>, and <a href="https://rycamargo.github.io">Raphael Y. de Camargo</a>.</p>
    </li>
    <li class="timeline-island__item">
      <div class="timeline-island__row">
        <span class="timeline-island__date">2016 – 2020</span>
        <span class="timeline-island__title">Double BSc in Computer Science &amp; Science and Technology</span>
      </div>
      <p class="timeline-island__detail">Center for Mathematics, Computing, and Cognition, Federal University of ABC (UFABC), Brazil 🇧🇷.</p>
    </li>
  </ol>
</div>

<!-- # 💬 Invited Talks
- Fill with all the talks... -->

# 💻 Work Experience

<div class="timeline-island">
  <ol class="timeline-island__list">
    <li class="timeline-island__item timeline-island__item--highlight">
      <div class="timeline-island__row">
        <span class="timeline-island__date">2026 →</span>
        <span class="timeline-island__title">Research Scientist, <a href="https://yneuro.com/">Yneuro</a></span>
      </div>
      <p class="timeline-island__detail">France 🇫🇷. Tools for EEG decoding and foundation models on neural signals.</p>
    </li>
    <li class="timeline-island__item">
      <div class="timeline-island__row">
        <span class="timeline-island__date">2026 →</span>
        <span class="timeline-island__title">Honorary Research Associate, <a href="https://inc.ucsd.edu/people/#Associate-Members">UC San Diego (INC)</a></span>
      </div>
      <p class="timeline-island__detail">Institute for Neural Computation, USA 🇺🇸.</p>
    </li>
    <li class="timeline-island__item">
      <div class="timeline-island__row">
        <span class="timeline-island__date">03/2022 – 06/2022</span>
        <span class="timeline-island__title">Data Scientist Intern, University of Glasgow / FGV</span>
      </div>
      <p class="timeline-island__detail">United Kingdom 🇬🇧.</p>
    </li>
    <li class="timeline-island__item">
      <div class="timeline-island__row">
        <span class="timeline-island__date">03/2021 – 08/2021</span>
        <span class="timeline-island__title">Data Scientist Intern, Getúlio Vargas Foundation (FGV)</span>
      </div>
      <p class="timeline-island__detail">Brazil 🇧🇷.</p>
    </li>
    <li class="timeline-island__item">
      <div class="timeline-island__row">
        <span class="timeline-island__date">07/2014 – 12/2015</span>
        <span class="timeline-island__title">Research Intern (Computer Vision), Dom Bosco Catholic University</span>
      </div>
      <p class="timeline-island__detail">Brazil 🇧🇷. INOVISÃO lab during high school. I published two papers :)</p>
    </li>
  </ol>
</div>

# 👥 Mentorship

Students I was lucky to work with and mentor:

<ul class="mentorship">
  <li class="mentorship__card">
    <h2 class="mentorship__name"><a href="https://www.linkedin.com/in/leoburgund/">Léo Burgund</a></h2>
    <span class="mentorship__role">Master student → Yneuro</span>
    <p class="mentorship__detail">Université Paris-Saclay (M2 Mathematics &amp; AI), now Machine Learning Researcher at Yneuro.</p>
  </li>
  <li class="mentorship__card">
    <h2 class="mentorship__name"><a href="https://www.linkedin.com/in/mariani-tom/">Tom Mariani</a></h2>
    <span class="mentorship__role">Master student → Yneuro</span>
    <p class="mentorship__detail">MVA, ENS Paris-Saclay / Mines, now Research Scientist at Yneuro.</p>
  </li>
  <li class="mentorship__card">
    <h2 class="mentorship__name"><a href="https://www.linkedin.com/in/amanjaiswal1503/">Aman Jaiswal</a></h2>
    <span class="mentorship__role">Master student</span>
    <p class="mentorship__detail">UC San Diego, MS in Computer Science.</p>
  </li>
  <li class="mentorship__card">
    <h2 class="mentorship__name"><a href="https://www.linkedin.com/in/kuntal-kokate-b05743169">Kuntal Kokate</a></h2>
    <span class="mentorship__role">Master student</span>
    <p class="mentorship__detail">UC San Diego, MS in Machine Learning &amp; Data Science (ECE).</p>
  </li>
  <li class="mentorship__card">
    <h2 class="mentorship__name"><a href="https://www.linkedin.com/in/jos%C3%A9-maur%C3%ADcio-nunes-de-oliveira-junior-aa174b92/">Jose Mauricio</a></h2>
    <span class="mentorship__role">Master student</span>
    <p class="mentorship__detail">Federal University of ABC, Computer Science.</p>
  </li>
  <li class="mentorship__card">
    <h2 class="mentorship__name"><a href="https://www.linkedin.com/in/taha-habib-a694a31b7/">Taha Habib</a></h2>
    <span class="mentorship__role">Undergrad → Master</span>
    <p class="mentorship__detail">Université Paris-Saclay, now a master student.</p>
  </li>
  <li class="mentorship__card">
    <h2 class="mentorship__name"><a href="https://orcid.org/0000-0002-0922-126X">Gustavo H. Rodrigues</a></h2>
    <span class="mentorship__role">Undergrad → Master</span>
    <p class="mentorship__detail">Universidade de São Paulo (USP), now a master student at USP.</p>
  </li>
  <li class="mentorship__card">
    <h2 class="mentorship__name"><a href="https://www.linkedin.com/in/brunajaflopes/">Bruna Junqueira</a></h2>
    <span class="mentorship__role">Undergrad → Master</span>
    <p class="mentorship__detail">USP, now in the Mathématiques, Vision, Apprentissage master at Université Paris-Saclay.</p>
  </li>
  <li class="mentorship__card">
    <h2 class="mentorship__name"><a href="https://www.linkedin.com/in/alexandre-janoni-bayerlein-047955220/">Alexandre Janoni</a></h2>
    <span class="mentorship__role">Undergrad → Industry</span>
    <p class="mentorship__detail">Federal University of ABC, now at Hospital Albert Einstein.</p>
  </li>
</ul>
