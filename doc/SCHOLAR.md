# Citation and discoverability

MuDuSi's technical report has a stable citation independent of search-engine indexing:

- **Title:** MuDuSi: A Multi-Agent Acoustic Simulator for Full-Duplex Speech Language Models
- **Authors:** Kyudan Jung; Jaegul Choo, in that order. Both affiliations: **KAIST AI**.
- **Publication:** MuDuSi Project, Technical Report **MUDUSI-TR-2026-01**, version **1.0**, **October 12, 2026**.
- **Canonical report page:** <https://kyudan1.github.io/MuDuSi/paper/>
- **Full-text PDF:** <https://kyudan1.github.io/MuDuSi/paper/mudusi-technical-report.pdf>
- **Citation files:** [BibTeX](../CITATION.bib) and [CFF](../CITATION.cff).

MuDuSi Project identifies the self-publishing project, not the authors' institutional affiliation. The report is not presented as an accepted NeurIPS paper. No DOI or arXiv identifier is assigned in this metadata.

## GitHub citation behavior

The root CFF file describes the software and points `preferred-citation` to the report. GitHub supports CFF `report` as a preferred citation and maps it to BibTeX `@techreport`; its repository citation feature reads a root `CITATION.cff` on the default branch. The software version is 0.1.0, while the separately versioned report is 1.0. [GitHub citation documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files)

## Google Scholar preparation

The hosting design follows these requirements from Google's guidance: a dedicated report URL with a visible complete abstract, plain links to searchable PDF text, public crawler access, bibliographic HTML metadata, and a PDF smaller than 5 MB. Report metadata includes the title, author names, publication date, publishing institution/project and report number. `citation_pdf_url` points to the PDF in the same directory as its HTML abstract. [Google Scholar inclusion guidelines](https://scholar.google.com/intl/en/scholar/inclusion.html)

For this report, the intended HTML values are:

```html
<meta name="citation_title" content="MuDuSi: A Multi-Agent Acoustic Simulator for Full-Duplex Speech Language Models">
<meta name="citation_author" content="Kyudan Jung">
<meta name="citation_author" content="Jaegul Choo">
<meta name="citation_publication_date" content="2026/10/12">
<meta name="citation_technical_report_institution" content="MuDuSi Project">
<meta name="citation_technical_report_number" content="MUDUSI-TR-2026-01">
<meta name="citation_pdf_url" content="https://kyudan1.github.io/MuDuSi/paper/mudusi-technical-report.pdf">
```

The public deployment should be checked without authentication: the page and PDF must load, the abstract must be readable, and title/author metadata must match the PDF. A valid CFF file, an accessible PDF, or successful deployment does **not** prove Scholar inclusion. Its crawler and parser make that decision. Check later by searching Scholar for the exact report title, rather than inferring inclusion from a `site:` result count. [Google Scholar troubleshooting](https://scholar.google.com/intl/en/scholar/inclusion.html#troubleshooting)

The reported status must remain **indexing not confirmed** until the report appears in Scholar search. Bibliographic citation does not need to wait for indexing. A separate ResearchGate or Zenodo upload is not part of this hosting workflow; a future DOI registration, repository deposit, or arXiv submission would be an optional additional publication step. Add any resulting identifier only after it has actually been issued.

## Maintaining the record

Keep the author order, title, date, version, and report number consistent across the PDF, HTML metadata, CFF, BibTeX, and README. Retain published versions when substantive report revisions occur, and avoid silently changing author or publication metadata. If a permanent URL changes, preserve a redirect to the report rather than sending readers to an unrelated homepage.

Before publishing a metadata change, validate `CITATION.cff` against the [official CFF 1.2.0 schema](https://raw.githubusercontent.com/citation-file-format/citation-file-format/1.2.0/schema.json). Recheck the deployed HTML and searchable PDF independently; schema validation does not test hosting or indexing.
