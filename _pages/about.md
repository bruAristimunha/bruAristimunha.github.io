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

# 📝 Publications 

<!-- <div class='paper-box'><div class='paper-box-image'><div><div class="badge">CVPR 2016</div><img src='images/500x300.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

**Kaiming He**, Xiangyu Zhang, Shaoqing Ren, Jian Sun

[**Project**](https://scholar.google.com/citations?view_op=view_citation&hl=zh-CN&user=DhtAFkwAAAAJ&citation_for_view=DhtAFkwAAAAJ:ALROH1vI_8AC) <strong><span class='show_paper_citations' data='DhtAFkwAAAAJ:ALROH1vI_8AC'></span></strong>
- Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet. 
</div>
</div>

- [Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet](https://github.com), A, B, C, **CVPR 2020** -->

- Klepachevskyi, D., Romano, A., **Aristimunha, B.**, Angiolelli, M., Trojsi, F., Bonavita, S., ..., Corsi M.-C. & Sorrentino, P. (2024).
    Magnetoencephalography-based interpretable automated differential diagnosis in neurodegenerative diseases. Heliyon Cells, The paper has been accepted.

- Wimpff, M. , **Aristimunha, B.**, Chevallier, S. & Yang, B. (2025). 
    Fine-Tuning Strategies for Continual Online EEG Motor Imagery Decoding: Insights from a Large-Scale Longitudinal Study. In 2025 47th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC) (pp. 1-7). IEEE.

- Darvishi-Bayazi M. J., Ghonia H., Riachi R., **Aristimunha, B.**, Khorasani A., Arefin M. R., Dumas G. & Rish I. (2024) 
    General-Purpose Brain Foundation Models for Time-Series Neuroimaging Data. Workshop on Time Series in the Age of Large Models @ NeurIPS 2024.

- Carrara, I.*, **Aristimunha, B.***, Corsi, M. C., de Camargo, R. Y., Chevallier, S., & Papadopoulo, T. (2024). 
    Geometric Neural Network based on Phase Space for BCI decoding. Journal of Neural Engineering. Joint first author

- **Aristimunha, B.**, Moreau T., Chevallier, S, Camargo, R. Y., & Corsi, M. C. 
    What is the best model for decoding neurophysiological signals? Depends on how you evaluate. In 33rd Annual Computational Neuroscience Meeting*CNS 2024.

- Rodrigues, G., **Aristimunha, B.**, Chevallier, S. & Camargo, R. Y. de (2024). 
    Combining Euclidean Alignment and Data Augmentation for BCI decoding. In 2024 32nd European Signal Processing Conference (EUSIPCO) (pp. 1-6). IEEE.

- Xu, J.*, **Aristimunha, B.***, Feucht, M. E.*, Qian, E., Liu, C., Shahjahan, T., ... & Nestor, A. (2024). 
    Alljoined--A dataset for EEG-to-Image decoding. Workshop Data Curation and Augmentation in Medical Imaging at 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 1–9. 

- Junqueira, B., **Aristimunha, B.**, Chevallier, S., & de Camargo, R. Y. (2024). 
    A systematic evaluation of Euclidean alignment with deep learning for EEG decoding. Journal of Neural Engineering, 21(3), 036038. doi:10.1088/1741-2552/ad4f18

- **Aristimunha, B.**, de Camargo, R. Y., Chevallier, S., Lucena, O., Thomas, A. G., Cardoso, M. J., Pinaya, W. L. & Dafflon, J. (2023). 
    Synthetic Sleep EEG Signal Generation using Latent Diffusion Models. In Deep Generative Models for Health Workshop NeurIPS 2023. SPOTLIGHT

- **Aristimunha, B.**, de Camargo, R. Y., Pinaya, W. L., Chevallier, S., Gramfort, A., & Rommel, C. (2023). 
    Evaluating the structure of cognitive tasks with transfer learning. In AI for Science Workshop NeurIPS 2023. 

- Moraes, C. P., **Aristimunha, B.**, Dos Santos, L. H., Pinaya, W. H. L., de Camargo, R. Y., Fantinato, D. G., & Neves, A. (2023, June).
   Applying independent vector analysis on eeg-based motor imagery classification. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5). IEEE.

- **Aristimunha, B.**, De Camargo, R. Y., Pinaya, W. H. L., Yger, F., Corsi, M. C., & Chevallier, S. (2023). 
    CONCERTO: Coherence & Functional Connectivity Graph Network. In Journée CORTICO 2023.

- Carrara, I.*, **Aristimunha, B.***, Chevallier, S., Corsi, M. C., & Papadopoulo, T. (2023). 
    Holographic EEG: multi-view deep learning for BCI. In Journée CORTICO 2023.

- **Aristimunha, B.**, Bayerlein, A. J., Cardoso, M. J., Pinaya, W. H. L., & De Camargo, R. Y. (2023). 
    Sleep-Energy: An Energy Optimization Method to Sleep Stage Scoring. IEEE Access, 11, 34595-34602.

- Chevallier, S., Carrara, I., **Aristimunha, B.**, Guetschel, P., Lopes, B., ... & Moreau, T. (2024). 
    The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark. arXiv preprint arXiv:2404.15319. The manuscript is under review in the Journal of Neural Engineering. 

- **B Aristimunha**, WHL Pinaya, RY de Camargo, Sylvain Chevallier, Alexandre Gramfort, Cédric Rommel. 
    Uncovering and improving the structure of cognitive tasks with transfer learning. Manuscript under review at Imaging NeuroScience.

**Open Software:**

- Mother of all BCI Benchmarks
   ****Aristimunha, B.****, Carrara, I., Guetschel, P., Sedlar, S., Rodrigues, P., Sosulski, J., Narayanan, D., Bjareholt, E., Quentin, B., Schirrmeister, R. T., Kobler, R., Kalunga, E., Darmet, L., Gregoire, C., Abdul Hussain, A., Gatti, R., Goncharenko, V., Thielen, J., Moreau, T., … Chevallier, S., Zenodo. [https://doi.org/10.5281/zenodo.11545401](https://doi.org/10.5281/zenodo.11545401), 2024

- Braindecode
    **Aristimunha B.**, Tibor, R., Gemein L., Gramfort, A., Rommel, C., Banville H., Sliwowskim M., Wilson, D., Theo gnassou, Pierre Gtch, Bruna Lopes, Thomas Moreau, Sara Sedlar, Marco Zamboni, Joseph Paillard, Matthieu Terris, Sylvain Chevallier, … Edward Yao., Zenodo., 2023


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

- [Jose Mauricio](https://www.linkedin.com/in/jos%C3%A9-maur%C3%ADcio-nunes-de-oliveira-junior-aa174b92/) Master Student at Federal University of ABC
- [Taha Habib](https://www.linkedin.com/in/taha-habib-a694a31b7/) Undergraduate student at Paris-Saclay University
- [Gustavo H Rodrigues](https://orcid.org/0000-0002-0922-126X) Undergraduate student at Universidade de Sao Paulo
- [Bruna Juqueira](https://www.linkedin.com/in/brunajaflopes/) Undergraduate student at Universidade de Sao Paulo
- [Alexandre Janoni](https://www.linkedin.com/in/alexandre-janoni-bayerlein-047955220/) Undergraduate student at Federal University of ABC
