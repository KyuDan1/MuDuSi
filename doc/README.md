# Technical report

[Read the PDF](duplex-arena-technical-report.pdf) · [LaTeX source](duplex-arena-technical-report.tex) · [References](duplex-arena-technical-report.bib)

The English report documents the implemented simulator, inference/evaluation,
acoustic trajectories, B200 execution evidence, limitations, and the interfaces
still required for full-weight SLM reinforcement learning. It is a project
technical report, not a claim of conference submission or acceptance.

## Build

A standard TeX Live installation with `pdflatex` and `bibtex` is sufficient:

```bash
cd doc
pdflatex -interaction=nonstopmode -halt-on-error duplex-arena-technical-report.tex
bibtex duplex-arena-technical-report
pdflatex -interaction=nonstopmode -halt-on-error duplex-arena-technical-report.tex
pdflatex -interaction=nonstopmode -halt-on-error duplex-arena-technical-report.tex
```

The unmodified `neurips_2025.sty` is from the
[official NeurIPS 2025 style archive](https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip),
linked by the [conference formatting instructions](https://neurips.cc/Conferences/2025/CallForPapers).
Its upstream attribution is preserved. The report uses the `preprint` option;
no conference acceptance footer is added. Source evidence is in
[`docs/B200_VERIFICATION.md`](../docs/B200_VERIFICATION.md),
[`docs/SHOWCASE.md`](../docs/SHOWCASE.md), and
[`docs/SLM_TRAINING.md`](../docs/SLM_TRAINING.md).
