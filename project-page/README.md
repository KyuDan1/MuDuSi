# MuDuSi project page

A buildless project page with three real-model conversation clips, simulation and trajectory-collection explanations, and the technical report.

```bash
python -m http.server 18767 --directory project-page --bind 127.0.0.1
```

Open http://localhost:18767/. Publish the contents of this directory under the chosen project-page URL. All media and fonts are local; no tracking scripts are included.

The visual design follows https://kyudan1.github.io/: Newsreader, Red Hat Mono, a cream background and a forest-green watercolor header. Font licenses are included in `assets/`. The watercolor image comes from the author’s personal website. Video provenance is documented in `../doc/media/provenance.json` and `../docs/SHOWCASE.md`.
