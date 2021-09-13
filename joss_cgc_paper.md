---
title: 'CGC: Working title'
tags:
  - Python
  - clusterting
  - co-/triclustering
  - Spatial
  - 
authors:
  - name: Francesco Nattino  # ^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0000-0000-0000
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Ou Ku #^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0000-0000-0000
    affiliation: "1"
  - name: Meiert W. Grootes #^[corresponding author]
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Raul
    orcid: 0000-0000-0000-0000
    affiliation: 2
  

affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University
   index: 1
 - name: Institution Name
   index: 2
 - name: Independent Researcher
   index: 3
date: 13 August 2017
bibliography: joss_cgc_paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Lorem ipsum CGC

# Statement of need

With the increase in large multi-dimensional data sets across research domains ranging from earth sciences to ....  

`cgc` was designed to be ... include link?.


# Our first section
  leaving the rest in as examole and reference


# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References