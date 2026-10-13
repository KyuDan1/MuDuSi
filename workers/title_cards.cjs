/* Render local, typographic chapter cards without external assets or services. */
const fs = require("node:fs");
const path = require("node:path");
const theme = require("../configs/visual-theme.json");
const { chromium } = require(
  path.resolve(
    process.env.ARENA_PLAYWRIGHT || "work/browser/node_modules/playwright",
  ),
);
const [config, fontPath, outputDirectory] = process.argv.slice(2);
if (!config || !fontPath || !outputDirectory)
  throw Error(
    "Usage: node workers/title_cards.cjs CARDS.json FONT.woff2 OUTPUT_DIR",
  );
const escape = (s) =>
  String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
(async () => {
  const font = fs.readFileSync(fontPath).toString("base64");
  const cards = JSON.parse(fs.readFileSync(config, "utf8"));
  fs.mkdirSync(outputDirectory, { recursive: true });
  const browser = await chromium.launch({
    executablePath:
      process.env.ARENA_CHROMIUM ||
      path.resolve(
        "work/playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell",
      ),
    args: ["--no-sandbox"],
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1280, height: 720 },
      deviceScaleFactor: 1,
    });
    for (const card of cards) {
      const colors = [theme.sage, theme.lavender, theme.apricot];
      const points =
        card.count === 9
          ? Array.from({length:9}, (_, i) => [905 + (i % 3) * 80, 255 + Math.floor(i / 3) * 80])
          : card.count === 2
          ? [
              [955, 275],
              [1055, 425],
            ]
          : [
              [910, 280],
              [1095, 310],
              [995, 470],
            ];
      const graphic = points
        .map(
          ([x, y], i) =>
            `<circle cx="${x}" cy="${y}" r="${card.count === 9 ? 31 : 90 + i * 8}" fill="none" stroke="${colors[i % colors.length]}" stroke-opacity=".12"/><circle cx="${x}" cy="${y}" r="${card.count === 9 ? 20 : 42}" fill="${colors[i % colors.length]}" fill-opacity=".07" stroke="${colors[i % colors.length]}" stroke-opacity=".7"/><circle cx="${x}" cy="${y}" r="7" fill="${colors[i % colors.length]}"/>`,
        )
        .join("");
      const lines = (card.count === 9 ? [] : points)
        .slice(1)
        .map(
          ([x, y]) =>
            `<path d="M${points[0][0]},${points[0][1]} L${x},${y}" stroke="#9cb1a6" stroke-opacity=".35" stroke-dasharray="3 6"/>`,
        )
        .join("");
      await page.setContent(`<!doctype html><style>
      @font-face{font-family:Film;src:url(data:font/woff2;base64,${font})}*{box-sizing:border-box}body{margin:0;background:${theme.videoBackground};color:${theme.cream};font-family:Film,sans-serif;width:1280px;height:720px;overflow:hidden}.bg{position:absolute;inset:0;background:radial-gradient(ellipse at 85% 45%,#24382e80,transparent 50%)}svg{position:absolute;inset:0}.content{position:absolute;top:216px;left:70px;width:850px}h1{font-size:57px;letter-spacing:-2px;line-height:1.25;margin:25px 0 24px;font-weight:400}.subtitle{font-size:18px;color:#9db0a4;letter-spacing:.1px}.accent{height:3px;width:64px;background:${escape(card.accent)};margin-top:40px}.chapters{position:absolute;left:70px;right:70px;bottom:79px;display:flex;gap:48px;border-top:1px solid #33443b;padding-top:27px}.chapter{color:#728b7e;font-size:12px;letter-spacing:1px}.chapter.active{color:${escape(card.accent)}}.chapter b{font-size:16px;margin-right:12px;font-weight:400}.foot{position:absolute;left:70px;bottom:32px;color:#637d6e;font-size:10px;letter-spacing:1.5px}
      </style><div class="bg"></div><svg width="1280" height="720">${lines}${graphic}</svg><div class="content"><h1>${escape(card.title)}</h1><div class="subtitle">${escape(card.subtitle)}</div><div class="accent"></div></div><div class="chapters">${[
        ["01", "FRIENDS"],
        ["02", "CLERK + CUSTOMER"],
        ["03", "THREE AGENTS"],
        ["04", "NINE ROOMS"],
      ]
        .map(
          ([n, label]) =>
            `<div class="chapter ${card.number === n ? "active" : ""}"><b>${n}</b>${label}</div>`,
        )
        .join(
          "",
        )}</div><div class="foot">ACTUAL GENERATED AUDIO / WORD-TIMED ASR REPLAY</div>`);
      await page.evaluate(() => document.fonts.ready);
      await page.screenshot({
        path: path.join(outputDirectory, path.basename(card.file)),
      });
    }
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
