# MuDuSi technical report

[Read the PDF](mudusi-technical-report.pdf) · [LaTeX source](mudusi-technical-report.tex) · [References](mudusi-technical-report.bib)

**Kyudan Jung and Jaegul Choo · KAIST AI · October 12, 2026 · Version 1.0**

Technical Report **MUDUSI-TR-2026-01**, self-published by the MuDuSi Project.
[Canonical paper page](https://kyudan1.github.io/MuDuSi/paper/)

The English report describes MuDuSi (Multi-agent Duplex Simulator) and documents the implemented simulator, inference/evaluation,
acoustic trajectories, B200 execution evidence, limitations, and the interfaces
still required for full-weight SLM reinforcement learning. It is a project
technical report, not a claim of conference submission or acceptance.

## Build

A standard TeX Live installation with `pdflatex` and `bibtex` is sufficient:

```bash
cd doc
pdflatex -interaction=nonstopmode -halt-on-error mudusi-technical-report.tex
bibtex mudusi-technical-report
pdflatex -interaction=nonstopmode -halt-on-error mudusi-technical-report.tex
pdflatex -interaction=nonstopmode -halt-on-error mudusi-technical-report.tex
```

The unmodified `neurips_2026.sty` is from the
[official NeurIPS 2026 style archive](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip),
linked by the [conference formatting instructions](https://neurips.cc/Conferences/2026/CallForPapers).
Its upstream attribution is preserved. The report uses the `preprint` option;
no conference acceptance footer is added. Source evidence is in
[`docs/B200_VERIFICATION.md`](../docs/B200_VERIFICATION.md),
[`docs/SHOWCASE.md`](../docs/SHOWCASE.md), and
[`docs/SLM_TRAINING.md`](../docs/SLM_TRAINING.md).
