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

1. <span class="anchor" id="paper-p1"></span>Hajhassani, D., Aristimunha, B., Graignic, P-A., Mellot, A., Kusch, L., Delorme, A., Semah, T., Caillet, A. H. From EEG Cleaning to Decoding: The Role of Artifact Rejection in MI-based BCIs. In 2026 34nd European Signal Processing Conference (EUSIPCO). IEEE.
2. <span class="anchor" id="paper-p2"></span>Guetschel, P., Aristimunha, B., Truong, D., Kokate, K., Tangermann, M., & Delorme, A. (2026). Toward OpenEEG-Bench: A live community-driven benchmark for EEG foundation models. In EUSIPCO 2026.
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
22. <span class="anchor" id="paper-p22"></span>Aristimunha, B., Ju, C., Collas, A., Bouchard, F., Mian, A., Thirion, B., Chevallier, S., & Kobler, R. (2026). SPD Learn: A geometric deep learning Python library for neural decoding through trivialization. Journal of Machine Learning - Open Source Track. [https://spdlearn.org](https://spdlearn.org)
23. <span class="anchor" id="paper-p23"></span>Aristimunha, B., Dotan, A., Guetschel, P., Truong, D., Kokate, K., Majumdar, A., Shriki, O., Delorme, A. (2026). EEG-DaSh: An Open Data, Tool, and Compute Resource for Machine Learning on Neuroelectromagnetic Data. Journal of Database. [https://eegdash.org](https://eegdash.org)


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
