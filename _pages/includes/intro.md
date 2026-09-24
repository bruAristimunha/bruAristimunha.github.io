<header class="home-intro" id="about-me">
  <p class="home-intro__field">Machine learning &amp; open science</p>
  <h1>Learning representations<br>from time series.</h1>
  <p class="home-intro__lead">I’m Bruno, a research scientist studying how models learn useful representations from time series, with brain signals as a central application.</p>
  <div class="home-intro__links">
    <a href="#-research-overview">Explore my research <span aria-hidden="true">↗</span></a>
    <a href="mailto:{{ site.author.email }}">Get in touch <span aria-hidden="true">↗</span></a>
  </div>
</header>

<div class="home-context" markdown="1">
<section markdown="1">

<h2>About me</h2>

I work at [Yneuro](https://yneuro.com/) in France and hold an honorary research appointment at [UC San Diego](https://inc.ucsd.edu/people/#Associate-Members). I earned my PhD in Computer Science jointly at [Université Paris-Saclay](https://www.universite-paris-saclay.fr/) and the [Federal University of ABC](https://www.ufabc.edu.br/), advised by [Sylvain Chevallier](https://sylvchev.github.io/), [Marie-Constance Corsi](https://marieconstance-corsi.netlify.app/) and [Raphael Y. de Camargo](https://rycamargo.github.io).

</section>
<section markdown="1">

## Research interests

My focus is **representation learning from time series**: learning structure that supports decoding, generation and transfer across subjects, sessions and datasets. I explore these questions through EEG and other neural signals, combining deep learning, Riemannian geometry and reproducible benchmarks.

</section>
</div>

## Open-source software

I build and maintain tools for learning from time series, geometric deep learning and reproducible neuroscience. Alongside these projects, I contribute to MNE-Python, MONAI, MONAI Generative and SpeechBrain.

<div class="library-showcase" data-library-showcase>
<div class="library-showcase__controls">
  <button class="library-motion-toggle" type="button" data-library-motion-toggle aria-pressed="false" hidden>Pause animations</button>
</div>
<div class="software-cards">
  <div class="software-card">
    {% include library_logo.html library="braindecode" %}
    <div class="software-card__body">
    <p class="software-card__name"><a href="https://braindecode.org">Braindecode</a></p>
    <span class="software-card__role">Lead maintainer</span>
    <p class="software-card__detail">Deep learning for EEG/MEG/brain-signal decoding in PyTorch.</p>
    <p class="software-card__meta">
      <a href="https://pepy.tech/project/braindecode"><img class="software-card__badge" src="https://pepy.tech/badge/braindecode" alt="Braindecode downloads on PyPI" height="20" loading="lazy"></a>
    </p>
    </div>
  </div>
  <div class="software-card">
    {% include library_logo.html library="moabb" %}
    <div class="software-card__body">
    <p class="software-card__name"><a href="https://moabb.neurotechx.com/docs/index.html">MOABB</a></p>
    <span class="software-card__role">Lead maintainer</span>
    <p class="software-card__detail">Mother of All BCI Benchmarks: reproducible evaluation of BCI pipelines across open datasets.</p>
    <p class="software-card__meta">
      <a href="https://pepy.tech/project/moabb"><img class="software-card__badge" src="https://pepy.tech/badge/moabb" alt="MOABB downloads on PyPI" height="20" loading="lazy"></a>
    </p>
    </div>
  </div>
  <div class="software-card">
    {% include library_logo.html library="spdlearn" %}
    <div class="software-card__body">
    <p class="software-card__name"><a href="https://spdlearn.org">SPD Learn</a></p>
    <span class="software-card__role">Creator</span>
    <p class="software-card__detail">Geometric (Riemannian/SPD) deep learning library for neural decoding through trivialization.</p>
    <p class="software-card__meta">
      <a href="https://pepy.tech/project/spd-learn"><img class="software-card__badge" src="https://pepy.tech/badge/spd-learn" alt="SPD Learn downloads on PyPI" height="20" loading="lazy"></a>
    </p>
    </div>
  </div>
  <div class="software-card">
    {% include library_logo.html library="eegdash" %}
    <div class="software-card__body">
    <p class="software-card__name"><a href="https://eegdash.org">EEG-DaSh</a></p>
    <span class="software-card__role">Creator</span>
    <p class="software-card__detail">Open data, tools, and compute resource for machine learning on neuroelectromagnetic data.</p>
    <p class="software-card__meta">
      <a href="https://pepy.tech/project/eegdash"><img class="software-card__badge" src="https://pepy.tech/badge/eegdash" alt="EEG-DaSh downloads on PyPI" height="20" loading="lazy"></a>
    </p>
    </div>
  </div>
</div>
</div>
<script src="{{ '/assets/js/library-logos.js' | relative_url }}?v={{ site.time | date: '%s' }}"></script>
