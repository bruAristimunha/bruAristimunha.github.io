---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

{% include_relative includes/intro.md %}

<!-- # 🔥 News
- *20XX*: &nbsp;🎉🎉 TO-DO... copy from my thesis report the event list... -->

# 🗺️ Journey

<div class="journey" data-journey>
  <div class="journey__header">
    <div class="journey__eyebrow">Three Acts</div>
    <h2 class="journey__title">A research scientist, in motion</h2>
    <p class="journey__lede">
      Three countries, one through-line: building tools to decode signals from the brain. The arc bends from a 2012 high-school science fair in Mato Grosso do Sul (I was sixteen) to a cotutelle PhD between Paris-Saclay and UFABC, and now a Research Scientist at Yneuro with an honorary affiliation at UC San Diego INC.
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
        <p class="journey__panel-summary">Research Scientist at Yneuro and Honorary Research Associate at UC San Diego (INC). Lead maintainer of <strong>Braindecode</strong> and <strong>MOABB</strong>. Co-organizing the MLSP 2025 Special Session on Decoding the Brain Time Series.</p>
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
          <div class="journey__ms-note">Lead organizer — Decoding the Brain Time Series, IEEE MLSP.</div>
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
          <div class="journey__ms-note">Cotutelle thesis defended February 2026 — Paris-Saclay & UFABC.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2024</span>
            <span class="journey__ms-label">Geometric Neural Network (JNE)</span>
          </div>
          <div class="journey__ms-note">Phase-space SPDNet for BCI-EEG decoding — <em>Journal of Neural Engineering</em>, with Carrara, Corsi, Papadopoulo.</div>
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
          <div class="journey__ms-note">EEG-to-Image decoding dataset — CVPR 2024 Workshop on Data Curation in Medical Imaging.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2024</span>
            <span class="journey__ms-label">MOABB Zenodo release</span>
          </div>
          <div class="journey__ms-note">Mother of all BCI Benchmarks — software registry at INRIA, DOI 10.5281/zenodo.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2023</span>
            <span class="journey__ms-label">Synthetic Sleep EEG (NeurIPS DGM4H)</span>
          </div>
          <div class="journey__ms-note">Latent diffusion models for EEG generation — NeurIPS 2023 DGM4H Workshop (Spotlight).</div>
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
          <div class="journey__ms-note">Independent Vector Analysis on EEG-Based Motor Imagery Classification — ICASSP 2023.</div>
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
          <div class="journey__ms-note">Data Scientist intern — University of Glasgow & Fundação Getúlio Vargas.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2021</span>
            <span class="journey__ms-label">FGV consultant</span>
          </div>
          <div class="journey__ms-note">Data Science consultant — IDB-funded urban-data project (Waze car-accident detection in São Paulo). Stack: AWS, SQL, Python, Dash.</div>
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
        <p class="journey__panel-summary">The spark: a 2012 high-school science fair in Mato Grosso do Sul. From a Junior Scientific Initiation scholarship at IFMS, to a computer-vision internship at UCDB INOVISÃO lab, to two undergrad degrees at UFABC — with prizes and first-author papers along the way.</p>
      </header>
      <ol class="journey__milestones">
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2020</span>
            <span class="journey__ms-label">Double BSc graduation</span>
          </div>
          <div class="journey__ms-note">UFABC — BSc in Science & Technology (with parallel work toward Computer Science). 2nd best undergrad paper at ERAMIA-SP 2020.</div>
        </li>
        <li class="journey__ms">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2018</span>
            <span class="journey__ms-label">Scientific Initiation — Neuroscience</span>
          </div>
          <div class="journey__ms-note">CNPq fellow — functional brain connectivity via causality in time series. UFABC.</div>
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
          <div class="journey__ms-note">Sibgrapi 2015 + Computer on the Beach — computer-vision work on bamboo-borer (<em>Dinoderus minutus</em>) detection from the UCDB INOVISÃO lab.</div>
        </li>
        <li class="journey__ms is-highlight">
          <span class="journey__ms-dot" aria-hidden="true"></span>
          <div class="journey__ms-row">
            <span class="journey__ms-year">2014</span>
            <span class="journey__ms-label">UCDB Computer Vision intern</span>
          </div>
          <div class="journey__ms-note">INOVISÃO lab under Prof. Hemerson Pistori — animal-behavior extraction from images. While still in high school.</div>
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
          <div class="journey__ms-note">SESC Prize at FETEC/MS — the very first project: photography use among IFMS high-school students.</div>
        </li>
      </ol>
    </section>
  </div>
</div>

# 📄 Curriculum Vitae

<div class="cv-embed">
  <div class="cv-embed__header">
    <div class="cv-embed__header-text">
      <p class="cv-embed__subtitle">Last updated — April 2026</p>
    </div>
    <a
      class="cv-embed__btn"
      href="https://docs.google.com/document/d/1Pba371Sv7Epgjz5kMwT2t8i3PgueNhpE8hN1nAR98iU/edit?usp=sharing"
      target="_blank"
      rel="noopener noreferrer"
    >
      <svg class="cv-embed__btn-icon" viewBox="0 0 20 20" fill="currentColor" width="16" height="16" aria-hidden="true">
        <path d="M4.5 2A2.5 2.5 0 0 0 2 4.5v11A2.5 2.5 0 0 0 4.5 18h11a2.5 2.5 0 0 0 2.5-2.5v-4a.75.75 0 0 0-1.5 0v4a1 1 0 0 1-1 1h-11a1 1 0 0 1-1-1v-11a1 1 0 0 1 1-1h4a.75.75 0 0 0 0-1.5h-4Zm7 0a.75.75 0 0 0 0 1.5h2.69L9.22 8.47a.75.75 0 0 0 1.06 1.06l4.97-4.97V7.25a.75.75 0 0 0 1.5 0v-4.5a.75.75 0 0 0-.75-.75h-4.5Z"/>
      </svg>
      Open Full CV
    </a>
  </div>
  <div class="cv-embed__frame-wrap">
    <iframe
      class="cv-embed__frame"
      src="https://docs.google.com/document/d/1Pba371Sv7Epgjz5kMwT2t8i3PgueNhpE8hN1nAR98iU/preview"
      title="Bruno Aristimunha — Curriculum Vitae"
      loading="lazy"
      sandbox="allow-scripts allow-same-origin"
    ></iframe>
  </div>
</div>

# 🧭 Research Overview

Grouped from the CV update on February 28, 2026. Labels `[P#]` match your CV numbering.

<p class="research-overview__actions">
  <a
    class="research-overview__action-link"
    href="{{ '/research-overview-figure.html' | relative_url }}"
    target="_blank"
    rel="noopener noreferrer"
  >
    Open standalone figure
  </a>
</p>

<div class="research-overview">
  <div class="research-overview__scroll">
    <div class="research-overview__stage">
      <svg
        class="research-overview__svg"
        aria-hidden="true"
        focusable="false"
      ></svg>
      <div class="research-overview__labels" aria-label="Brain Decoding contribution tree"></div>
    </div>
  </div>
  <script type="application/json" class="research-overview__data">
    {{ site.data.research_overview | jsonify }}
  </script>
  <noscript>
    <ul class="research-tree">
      {% include research_tree_node.html nodes=site.data.research_overview %}
    </ul>
  </noscript>
</div>


# 📝 Publications (Full List)

1. <span class="anchor" id="paper-p1"></span>Hajhassani, D., Aristimunha, B., Graignic, P-A., Mellot, A., Kusch, L., Delorme, A., Semah, T., Caillet, A. H. From EEG Cleaning to Decoding: The Role of Artifact Rejection in MI-based BCIs. In 2026 34nd European Signal Processing Conference (EUSIPCO). IEEE. ***SUBMITTED***
2. <span class="anchor" id="paper-p2"></span>Guetschel, P., Aristimunha, B., Truong, D., Kokate, K., Tangermann, M., & Delorme, A. (2026). Toward OpenEEG-Bench: A live community-driven benchmark for EEG foundation models. In EUSIPCO 2026. ***SUBMITTED***
3. <span class="anchor" id="paper-p3"></span>Aristimunha, B., Truong, D., Guetschel, P., Shirazi, S. Y., Guyon, I., Franco, A. R., ... & Delorme, A. EEG Foundation Challenge: From Cross-Task to Cross-Subject EEG Decoding. NeurIPS 2025.
4. <span class="anchor" id="paper-p4"></span>Klepachevskyi, D., Romano, A., Aristimunha, B., Angiolelli, M., Trojsi, F., Bonavita, S., ..., Corsi M.-C. & Sorrentino, P. (2024). Magnetoencephalography-based interpretable automated differential diagnosis in neurodegenerative diseases. Heliyon Cells.
5. <span class="anchor" id="paper-p5"></span>Wimpff, M., Aristimunha, B., Chevallier, S. & Yang, B. (2025). Fine-Tuning Strategies for Continual Online EEG Motor Imagery Decoding: Insights from a Large-Scale Longitudinal Study. In EMBC 2025. IEEE.
6. <span class="anchor" id="paper-p6"></span>Darvishi-Bayazi, M. J., Ghonia, H., Riachi, R., Aristimunha, B., Khorasani, A., Arefin, M. R., Dumas, G. & Rish, I. (2024). General-Purpose Brain Foundation Models for Time-Series Neuroimaging Data. NeurIPS 2024 Workshop.
7. <span class="anchor" id="paper-p7"></span>Carrara, I.*, Aristimunha, B.*, Corsi, M. C., de Camargo, R. Y., Chevallier, S., & Papadopoulo, T. (2024). Geometric Neural Network based on Phase Space for BCI decoding. Journal of Neural Engineering.
8. <span class="anchor" id="paper-p8"></span>Aristimunha, B., Moreau, T., Chevallier, S., Camargo, R. Y., & Corsi, M. C. (2024). What is the best model for decoding neurophysiological signals? Depends on how you evaluate. CNS 2024.
9. <span class="anchor" id="paper-p9"></span>Rodrigues, G., Aristimunha, B., Chevallier, S. & Camargo, R. Y. de (2024). Combining Euclidean Alignment and Data Augmentation for BCI decoding. In EUSIPCO 2024. IEEE.
10. <span class="anchor" id="paper-p10"></span>Xu, J.*, Aristimunha, B.*, Feucht, M. E.*, Qian, E., Liu, C., Shahjahan, T., ... & Nestor, A. (2024). Alljoined: A dataset for EEG-to-Image decoding. CVPR 2024 Workshop.
11. <span class="anchor" id="paper-p11"></span>Junqueira, B., Aristimunha, B., Chevallier, S., & de Camargo, R. Y. (2024). A systematic evaluation of Euclidean alignment with deep learning for EEG decoding. Journal of Neural Engineering, 21(3), 036038. doi:10.1088/1741-2552/ad4f18
12. <span class="anchor" id="paper-p12"></span>Aristimunha, B., de Camargo, R. Y., Chevallier, S., Lucena, O., Thomas, A. G., Cardoso, M. J., Pinaya, W. L. & Dafflon, J. (2023). Synthetic Sleep EEG Signal Generation using Latent Diffusion Models. NeurIPS 2023 DGM4H Workshop (Spotlight).
13. <span class="anchor" id="paper-p13"></span>Aristimunha, B., de Camargo, R. Y., Pinaya, W. L., Chevallier, S., Gramfort, A., & Rommel, C. (2023). Evaluating the structure of cognitive tasks with transfer learning. NeurIPS 2023 AI for Science Workshop.
14. <span class="anchor" id="paper-p14"></span>Moraes, C. P., Aristimunha, B., Dos Santos, L. H., Pinaya, W. H. L., de Camargo, R. Y., Fantinato, D. G., & Neves, A. (2023). Applying independent vector analysis on EEG-based motor imagery classification. ICASSP 2023. IEEE.
15. <span class="anchor" id="paper-p15"></span>Aristimunha, B., De Camargo, R. Y., Pinaya, W. H. L., Yger, F., Corsi, M. C., & Chevallier, S. (2023). CONCERTO: Coherence & Functional Connectivity Graph Network. Journee CORTICO 2023.
16. <span class="anchor" id="paper-p16"></span>Carrara, I.*, Aristimunha, B.*, Chevallier, S., Corsi, M. C., & Papadopoulo, T. (2023). Holographic EEG: multi-view deep learning for BCI. Journee CORTICO 2023.
17. <span class="anchor" id="paper-p17"></span>Aristimunha, B., Bayerlein, A. J., Cardoso, M. J., Pinaya, W. H. L., & De Camargo, R. Y. (2023). Sleep-Energy: An Energy Optimization Method to Sleep Stage Scoring. IEEE Access, 11, 34595-34602.
18. <span class="anchor" id="paper-p18"></span>Chevallier, S., Carrara, I., Aristimunha, B., Guetschel, P., Lopes, B., ... & Moreau, T. (2024). The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark. arXiv:2404.15319. Under review at Journal of Neural Engineering.
19. <span class="anchor" id="paper-p19"></span>Aristimunha, B., Pinaya, W. H. L., de Camargo, R. Y., Chevallier, S., Gramfort, A., & Rommel, C. Uncovering and improving the structure of cognitive tasks with transfer learning. Under review at Imaging Neuroscience.
20. <span class="anchor" id="paper-p20"></span>Aristimunha, B., Carrara, I., Guetschel, P., Sedlar, S., Rodrigues, P., Sosulski, J., Narayanan, D., Bjareholt, E., Quentin, B., Schirrmeister, R. T., Kobler, R., Kalunga, E., Darmet, L., Gregoire, C., Abdul Hussain, A., Gatti, R., Goncharenko, V., Thielen, J., Moreau, T., ... Chevallier, S. (2024). Mother of all BCI Benchmarks. Zenodo. [https://doi.org/10.5281/zenodo.11545401](https://doi.org/10.5281/zenodo.11545401)
21. <span class="anchor" id="paper-p21"></span>Aristimunha, B., Tibor, R., Gemein, L., Gramfort, A., Rommel, C., Banville, H., Sliwowskim, M., Wilson, D., Theo gnassou, P., Gtch, P., Lopes, B., Moreau, T., Sedlar, S., Zamboni, M., Paillard, J., Terris, M., Chevallier, S., ... Yao, E. (2023). Braindecode. Zenodo. [https://braindecode.org](https://braindecode.org)
22. <span class="anchor" id="paper-p22"></span>Aristimunha, B., Ju, C., Collas, A., Bouchard, F., Mian, A., Thirion, B., Chevallier, S., & Kobler, R. (2026). SPD Learn: A geometric deep learning Python library for neural decoding through trivialization. Journal of Machine Learning - Open Source Track. [https://spdlearn.org](https://spdlearn.org) ***SUBMITTED***
23. <span class="anchor" id="paper-p23"></span>Aristimunha, B., Dotan, A., Guetschel, P., Truong, D., Kokate, K., Majumdar, A., Shriki, O., Delorme, A. (2026). EEG-DaSh: An Open Data, Tool, and Compute Resource for Machine Learning on Neuroelectromagnetic Data. Journal of Database. [https://eegdash.org](https://eegdash.org) ***SUBMITTED***


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
      <p class="timeline-island__detail">France 🇫🇷 — building tools for EEG decoding and foundation models on neural signals.</p>
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
      <p class="timeline-island__detail">Brazil 🇧🇷.</p>
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
      <p class="timeline-island__detail">Brazil 🇧🇷 — INOVISÃO lab during high school. I published two papers :)</p>
    </li>
  </ol>
</div>

# 👥 Mentorship

I was privileged to work with and mentor a group of outstanding students:

<ul class="mentorship">
  <li class="mentorship__card">
    <h4 class="mentorship__name"><a href="https://www.linkedin.com/in/jos%C3%A9-maur%C3%ADcio-nunes-de-oliveira-junior-aa174b92/">Jose Mauricio</a></h4>
    <span class="mentorship__role">Master student</span>
    <p class="mentorship__detail">Federal University of ABC, Computer Science.</p>
  </li>
  <li class="mentorship__card">
    <h4 class="mentorship__name"><a href="https://www.linkedin.com/in/taha-habib-a694a31b7/">Taha Habib</a></h4>
    <span class="mentorship__role">Undergrad → Master</span>
    <p class="mentorship__detail">Université Paris-Saclay, now a master student.</p>
  </li>
  <li class="mentorship__card">
    <h4 class="mentorship__name"><a href="https://orcid.org/0000-0002-0922-126X">Gustavo H. Rodrigues</a></h4>
    <span class="mentorship__role">Undergrad → Master</span>
    <p class="mentorship__detail">Universidade de São Paulo (USP), now a master student at USP.</p>
  </li>
  <li class="mentorship__card">
    <h4 class="mentorship__name"><a href="https://www.linkedin.com/in/brunajaflopes/">Bruna Junqueira</a></h4>
    <span class="mentorship__role">Undergrad → Master</span>
    <p class="mentorship__detail">USP, now in the Mathématiques, Vision, Apprentissage master at Université Paris-Saclay.</p>
  </li>
  <li class="mentorship__card">
    <h4 class="mentorship__name"><a href="https://www.linkedin.com/in/alexandre-janoni-bayerlein-047955220/">Alexandre Janoni</a></h4>
    <span class="mentorship__role">Undergrad → Industry</span>
    <p class="mentorship__detail">Federal University of ABC, now at Hospital Albert Einstein.</p>
  </li>
</ul>
