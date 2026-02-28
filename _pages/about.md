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

# 🧭 Research Overview

Grouped from the CV update on 28/02/2026. Labels `[P#]` match your CV numbering.

<div class="research-overview">
  <ul class="research-tree">
    <li>
      <span class="research-tree__branch">Brain Decoding</span>
      <ul>
        <li>
          <span class="research-tree__branch">Preprocessing</span>
          <ul>
            <li><span class="research-tree__leaf"><strong>[P1]</strong> From EEG Cleaning to Decoding (EUSIPCO 2026). Semantic: artifact-aware preprocessing for MI decoding.</span></li>
          </ul>
        </li>

        <li>
          <span class="research-tree__branch">Data</span>
          <ul>
            <li>
              <span class="research-tree__branch">Data resources and platforms</span>
              <ul>
                <li><span class="research-tree__leaf"><strong>[P10]</strong> Alljoined EEG-to-Image Dataset (CVPR Workshop 2024). Semantic: multimodal EEG-vision grounding.</span></li>
                <li><span class="research-tree__leaf"><strong>[P20]</strong> MOABB (Zenodo 2024). Semantic: reproducible benchmark infrastructure. (<a href="https://doi.org/10.5281/zenodo.11545401">DOI</a>)</span></li>
                <li><span class="research-tree__leaf"><strong>[P23]</strong> EEG-DaSh (Journal of Database 2026). Semantic: open data/tool/compute ecosystem. (<a href="https://eegdash.org">project</a>)</span></li>
              </ul>
            </li>
            <li>
              <span class="research-tree__branch">Data generation</span>
              <ul>
                <li><span class="research-tree__leaf"><strong>[P12]</strong> Synthetic Sleep EEG with Latent Diffusion (NeurIPS DGM4H 2023). Semantic: generative augmentation for neuro-signals.</span></li>
              </ul>
            </li>
          </ul>
        </li>

        <li>
          <span class="research-tree__branch">Data Alignment</span>
          <ul>
            <li>
              <span class="research-tree__branch">Cross-session and cross-subject alignment</span>
              <ul>
                <li><span class="research-tree__leaf"><strong>[P9]</strong> Euclidean Alignment + Augmentation (EUSIPCO 2024). Semantic: domain alignment plus augmentation synergy.</span></li>
                <li><span class="research-tree__leaf"><strong>[P11]</strong> Systematic Euclidean Alignment Evaluation (JNE 2024). Semantic: rigorous protocol-level validation.</span></li>
                <li><span class="research-tree__leaf"><strong>[P14]</strong> Independent Vector Analysis for MI (ICASSP 2023). Semantic: source-separation-informed decoding.</span></li>
              </ul>
            </li>
            <li>
              <span class="research-tree__branch">Online adaptation</span>
              <ul>
                <li><span class="research-tree__leaf"><strong>[P5]</strong> Continual Online EEG MI Fine-Tuning (EMBC 2025). Semantic: continual adaptation for online BCI.</span></li>
              </ul>
            </li>
          </ul>
        </li>

        <li>
          <span class="research-tree__branch">Models</span>
          <ul>
            <li>
              <span class="research-tree__branch">Decoding architectures</span>
              <ul>
                <li><span class="research-tree__leaf"><strong>[P7]</strong> Geometric Neural Network on Phase Space (JNE 2024). Semantic: Riemannian phase-space learning.</span></li>
                <li><span class="research-tree__leaf"><strong>[P15]</strong> CONCERTO (Journee CORTICO 2023). Semantic: coherence/connectivity graph modeling.</span></li>
                <li><span class="research-tree__leaf"><strong>[P16]</strong> Holographic EEG (Journee CORTICO 2023). Semantic: multi-view representation learning.</span></li>
                <li><span class="research-tree__leaf"><strong>[P17]</strong> Sleep-Energy (IEEE Access 2023). Semantic: energy-based optimization for sleep staging.</span></li>
              </ul>
            </li>
            <li>
              <span class="research-tree__branch">Foundation and transfer models</span>
              <ul>
                <li><span class="research-tree__leaf"><strong>[P2]</strong> OpenEEG-Bench (EUSIPCO 2026). Semantic: live benchmark for EEG foundation models.</span></li>
                <li><span class="research-tree__leaf"><strong>[P3]</strong> EEG Foundation Challenge (NeurIPS 2025). Semantic: cross-task and cross-subject generalization.</span></li>
                <li><span class="research-tree__leaf"><strong>[P6]</strong> General-Purpose Brain Foundation Models (NeurIPS Workshop 2024). Semantic: universal pretrained neuro-time-series models.</span></li>
                <li><span class="research-tree__leaf"><strong>[P13]</strong> Cognitive Task Structure with Transfer Learning (NeurIPS Workshop 2023). Semantic: representation transfer across tasks.</span></li>
                <li><span class="research-tree__leaf"><strong>[P19]</strong> Uncovering/Improving Cognitive Task Structure (under review). Semantic: task geometry and transfer-aware refinement.</span></li>
              </ul>
            </li>
          </ul>
        </li>

        <li>
          <span class="research-tree__branch">Benchmark and Evaluation</span>
          <ul>
            <li><span class="research-tree__leaf"><strong>[P8]</strong> Best Model Depends on Evaluation (CNS 2024). Semantic: benchmark protocol sensitivity.</span></li>
            <li><span class="research-tree__leaf"><strong>[P18]</strong> Largest MOABB Reproducibility Study (under review). Semantic: open-science reproducibility at scale.</span></li>
          </ul>
        </li>

        <li>
          <span class="research-tree__branch">Clinical</span>
          <ul>
            <li><span class="research-tree__leaf"><strong>[P4]</strong> Interpretable MEG Differential Diagnosis (Heliyon Cells 2024). Semantic: clinically interpretable neurodiagnostic decoding.</span></li>
          </ul>
        </li>

        <li>
          <span class="research-tree__branch">Software</span>
          <ul>
            <li><span class="research-tree__leaf"><strong>[P21]</strong> Braindecode (Zenodo 2023). Semantic: deep-learning EEG toolbox. (<a href="https://braindecode.org">project</a>)</span></li>
            <li><span class="research-tree__leaf"><strong>[P22]</strong> SPD Learn (JMLR OST 2026). Semantic: geometric deep-learning primitives for decoding. (<a href="https://spdlearn.org">project</a>)</span></li>
          </ul>
        </li>
      </ul>
    </li>
  </ul>
</div>


# 📝 Publications (Full List)

1. Hajhassani, D., Aristimunha, B., Graignic, P-A., Mellot, A., Kusch, L., Delorme, A., Semah, T., Caillet, A. H. From EEG Cleaning to Decoding: The Role of Artifact Rejection in MI-based BCIs. In 2026 34nd European Signal Processing Conference (EUSIPCO). IEEE.
2. Guetschel, P., Aristimunha, B., Truong, D., Kokate, K., Tangermann, M., & Delorme, A. (2026). Toward OpenEEG-Bench: A live community-driven benchmark for EEG foundation models. In EUSIPCO 2026.
3. Aristimunha, B., Truong, D., Guetschel, P., Shirazi, S. Y., Guyon, I., Franco, A. R., ... & Delorme, A. EEG Foundation Challenge: From Cross-Task to Cross-Subject EEG Decoding. NeurIPS 2025.
4. Klepachevskyi, D., Romano, A., Aristimunha, B., Angiolelli, M., Trojsi, F., Bonavita, S., ..., Corsi M.-C. & Sorrentino, P. (2024). Magnetoencephalography-based interpretable automated differential diagnosis in neurodegenerative diseases. Heliyon Cells.
5. Wimpff, M., Aristimunha, B., Chevallier, S. & Yang, B. (2025). Fine-Tuning Strategies for Continual Online EEG Motor Imagery Decoding: Insights from a Large-Scale Longitudinal Study. In EMBC 2025. IEEE.
6. Darvishi-Bayazi, M. J., Ghonia, H., Riachi, R., Aristimunha, B., Khorasani, A., Arefin, M. R., Dumas, G. & Rish, I. (2024). General-Purpose Brain Foundation Models for Time-Series Neuroimaging Data. NeurIPS 2024 Workshop.
7. Carrara, I.*, Aristimunha, B.*, Corsi, M. C., de Camargo, R. Y., Chevallier, S., & Papadopoulo, T. (2024). Geometric Neural Network based on Phase Space for BCI decoding. Journal of Neural Engineering.
8. Aristimunha, B., Moreau, T., Chevallier, S., Camargo, R. Y., & Corsi, M. C. (2024). What is the best model for decoding neurophysiological signals? Depends on how you evaluate. CNS 2024.
9. Rodrigues, G., Aristimunha, B., Chevallier, S. & Camargo, R. Y. de (2024). Combining Euclidean Alignment and Data Augmentation for BCI decoding. In EUSIPCO 2024. IEEE.
10. Xu, J.*, Aristimunha, B.*, Feucht, M. E.*, Qian, E., Liu, C., Shahjahan, T., ... & Nestor, A. (2024). Alljoined: A dataset for EEG-to-Image decoding. CVPR 2024 Workshop.
11. Junqueira, B., Aristimunha, B., Chevallier, S., & de Camargo, R. Y. (2024). A systematic evaluation of Euclidean alignment with deep learning for EEG decoding. Journal of Neural Engineering, 21(3), 036038. doi:10.1088/1741-2552/ad4f18
12. Aristimunha, B., de Camargo, R. Y., Chevallier, S., Lucena, O., Thomas, A. G., Cardoso, M. J., Pinaya, W. L. & Dafflon, J. (2023). Synthetic Sleep EEG Signal Generation using Latent Diffusion Models. NeurIPS 2023 DGM4H Workshop (Spotlight).
13. Aristimunha, B., de Camargo, R. Y., Pinaya, W. L., Chevallier, S., Gramfort, A., & Rommel, C. (2023). Evaluating the structure of cognitive tasks with transfer learning. NeurIPS 2023 AI for Science Workshop.
14. Moraes, C. P., Aristimunha, B., Dos Santos, L. H., Pinaya, W. H. L., de Camargo, R. Y., Fantinato, D. G., & Neves, A. (2023). Applying independent vector analysis on EEG-based motor imagery classification. ICASSP 2023. IEEE.
15. Aristimunha, B., De Camargo, R. Y., Pinaya, W. H. L., Yger, F., Corsi, M. C., & Chevallier, S. (2023). CONCERTO: Coherence & Functional Connectivity Graph Network. Journee CORTICO 2023.
16. Carrara, I.*, Aristimunha, B.*, Chevallier, S., Corsi, M. C., & Papadopoulo, T. (2023). Holographic EEG: multi-view deep learning for BCI. Journee CORTICO 2023.
17. Aristimunha, B., Bayerlein, A. J., Cardoso, M. J., Pinaya, W. H. L., & De Camargo, R. Y. (2023). Sleep-Energy: An Energy Optimization Method to Sleep Stage Scoring. IEEE Access, 11, 34595-34602.
18. Chevallier, S., Carrara, I., Aristimunha, B., Guetschel, P., Lopes, B., ... & Moreau, T. (2024). The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark. arXiv:2404.15319. Under review at Journal of Neural Engineering.
19. Aristimunha, B., Pinaya, W. H. L., de Camargo, R. Y., Chevallier, S., Gramfort, A., & Rommel, C. Uncovering and improving the structure of cognitive tasks with transfer learning. Under review at Imaging Neuroscience.
20. Aristimunha, B., Carrara, I., Guetschel, P., Sedlar, S., Rodrigues, P., Sosulski, J., Narayanan, D., Bjareholt, E., Quentin, B., Schirrmeister, R. T., Kobler, R., Kalunga, E., Darmet, L., Gregoire, C., Abdul Hussain, A., Gatti, R., Goncharenko, V., Thielen, J., Moreau, T., ... Chevallier, S. (2024). Mother of all BCI Benchmarks. Zenodo. [https://doi.org/10.5281/zenodo.11545401](https://doi.org/10.5281/zenodo.11545401)
21. Aristimunha, B., Tibor, R., Gemein, L., Gramfort, A., Rommel, C., Banville, H., Sliwowskim, M., Wilson, D., Theo gnassou, P., Gtch, P., Lopes, B., Moreau, T., Sedlar, S., Zamboni, M., Paillard, J., Terris, M., Chevallier, S., ... Yao, E. (2023). Braindecode. Zenodo. [https://braindecode.org](https://braindecode.org)
22. Aristimunha, B., Ju, C., Collas, A., Bouchard, F., Mian, A., Thirion, B., Chevallier, S., & Kobler, R. (2026). SPD Learn: A geometric deep learning Python library for neural decoding through trivialization. Journal of Machine Learning - Open Source Track. [https://spdlearn.org](https://spdlearn.org)
23. Aristimunha, B., Dotan, A., Guetschel, P., Truong, D., Kokate, K., Majumdar, A., Shriki, O., Delorme, A. (2026). EEG-DaSh: An Open Data, Tool, and Compute Resource for Machine Learning on Neuroelectromagnetic Data. Journal of Database. [https://eegdash.org](https://eegdash.org)


# 📖 Educations
- *09/2020 – 02/2026*, Ph.D. IN COMPUTER SCIENCE @Université Paris-Saclay and UFABC. 

- *2016-2020*, Double BSc COMPUTER SCIENCE and Science and Technology, at the Center for Mathematics, Computing, and Cognition, Federal University of ABC (UFABC), Brazil.

<!-- # 💬 Invited Talks
- Fill with all the talks... -->

# 💻 Work Experience
- *03/2022 – 06/2022*, Data Scientist Intern, University of Glasgow/FGV, Brazil.
- *03/2021 – 08/2021*, Data Scientist internship, Getúlio Vargas Foundation - FGV, Brazil.
- *07/2014 – 12/2015*, Research Intern during High school in Computer Vision, Dom Bosco Catholic University, Brazil. I published two papers :) 

# Menthorship

I was privileged to work with and mentor a group of outstanding students:

- [Jose Mauricio](https://www.linkedin.com/in/jos%C3%A9-maur%C3%ADcio-nunes-de-oliveira-junior-aa174b92/) Master Student at Federal University of ABC in Computer Science.
- [Taha Habib](https://www.linkedin.com/in/taha-habib-a694a31b7/) Undergraduate student at Paris-Saclay University, now master student.
- [Gustavo H Rodrigues](https://orcid.org/0000-0002-0922-126X) Undergraduate student at Universidade de Sao Paulo, now master student at Universidade de Sao Paulo.
- [Bruna Juqueira](https://www.linkedin.com/in/brunajaflopes/) Undergraduate student at Universidade de Sao Paulo. Now Mathématiques, Vision, Apprentissage master student at Universite Paris-Saclay.
- [Alexandre Janoni](https://www.linkedin.com/in/alexandre-janoni-bayerlein-047955220/) Undergraduate student at Federal University of ABC, now working at Hospital Albert Einstein, the best hospital in Latin America.
