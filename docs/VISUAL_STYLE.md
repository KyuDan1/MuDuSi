# MuDuSi visual style

MuDuSi uses restrained pastel accents across the paper, GitHub header, project page, and videos. The colors identify agents and connect those surfaces; typography and readable contrast carry the information. Avoid decorative gradients, glow effects, and extra brand stamps above headings.

The canonical color tokens are in [`configs/visual-theme.json`](../configs/visual-theme.json). The 3D frontend and title-card renderer import this file directly; the report, SVG banner, and static site mirror its values.

## Shared palette

| Token | Hex | Role |
|---|---|---|
| Sage | `#B7D7C0` | Agent 1 / `persona`; primary accent |
| Lavender | `#C9C0E8` | Agent 2 / `moshi`; secondary accent |
| Apricot | `#E8C6AA` | Agent 3 / `custom`; third-agent accent |
| Cream | `#FAF8F2` | Light presentation and banner background |
| Forest | `#28483D` | Headings, text on pale fills, badge background |
| Muted green | `#66786F` | Secondary text and restrained connecting lines on light backgrounds |
| Video dark | `#111B18` | Dark video and 3D presentation background |

Agent colors follow stable simulation slot IDs, not model architecture or dialogue role. For example, a second PersonaPlex instance in the `moshi` slot remains lavender. Apply the same mapping to scene agents, activity lanes, speech bubbles, and paper diagrams whenever those elements identify a particular agent. Pair color with a label, number, or shape so identity never depends on hue alone.

The connected circles in the GitHub banner represent these three slots. They illustrate shared interaction; they do not claim three model architectures, measured network topology, or a specific conversational outcome.

## Contrast and surfaces

Use forest text on pastel or cream surfaces. On dark video backgrounds, use cream body text and pastel agent accents. Muted green is intended for light surfaces; do not reuse it for small text on dark video backgrounds. Keep normal text at a readable size and check actual foreground/background contrast after introducing transparency.

The paper keeps a white page for printing and uses the shared accents in figures and restrained highlights. Videos use a dark background to separate the 3D scene, captions, and activity indicators. This difference is deliberate: matching accent identities matters more than giving print and video the same background.

Badges use a dark forest or muted-green background with light lettering. Live CI status keeps its native success/failure color so branding does not hide status. Pastels should not become low-contrast text or a light badge fill behind white text.

## GitHub implementation

GitHub README rendering does not provide a custom site stylesheet. The header therefore uses [a small source SVG](../doc/media/mudusi-banner.svg), ordinary Markdown, and compact badges. The SVG uses system fonts, includes a title and description, and has a descriptive image alt in the README. It requires no external font or generated bitmap asset.

Keep the README usable without its banner: links, setup commands, feature descriptions, and citation text remain regular document content. Do not embed demo videos back into the README; link to the project page for playback.

## Updating visual assets

When changing a semantic color, update its uses together in the frontend, exported videos, project-page styles, and technical-report figures. Rebuild generated assets from their source rather than modifying screenshots. Inspect a two-agent and a three-agent view, a speech caption, and a paper figure after the change; verify labels and audio provenance remain intact. Video exports should change presentation only, preserving source speech, selected time intervals, and audio/caption alignment.
