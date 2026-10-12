/* Deterministic browser frames + real recorded audio, encoded in one bounded job. */
const fs = require("node:fs");
const path = require("node:path");
const { spawn, execFileSync } = require("node:child_process");
const { once } = require("node:events");
const { chromium } = require(
  path.resolve(
    process.env.ARENA_PLAYWRIGHT || "work/browser/node_modules/playwright",
  ),
);
const options = Object.fromEntries(
  process.argv
    .slice(2)
    .reduce(
      (a, v, i, xs) => (i % 2 ? a : [...a, [v.replace(/^--/, ""), xs[i + 1]]]),
      [],
    ),
);
if (!options.recording || !options.output)
  throw new Error(
    "Use --recording RUN_DIR --output FILE.mp4 [--start 0 --seconds 60 --fps 24]",
  );
const python = path.resolve(process.env.ARENA_PYTHON || ".venv/bin/python");
const ffmpeg =
  process.env.ARENA_FFMPEG ||
  execFileSync(
    python,
    ["-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
    { encoding: "utf8" },
  ).trim();
const output = path.resolve(options.output);
fs.mkdirSync(path.dirname(output), { recursive: true });
const port = Number(options.port || 18766);
const origin = `http://127.0.0.1:${port}`;
const serverLog = fs.openSync(`${output}.server.log`, "w");
const server = spawn(
  python,
  [
    "-m",
    "duplex_arena",
    "serve",
    "--host",
    "127.0.0.1",
    "--port",
    String(port),
    "--replay",
    path.resolve(options.recording),
  ],
  { stdio: ["ignore", serverLog, serverLog] },
);
let browser, encoder;
async function main() {
  for (let i = 0; i < 120; i++) {
    if (server.exitCode !== null) throw new Error("Replay server exited");
    try {
      if ((await fetch(`${origin}/api/health`)).ok) break;
    } catch {}
    await new Promise((r) => setTimeout(r, 500));
  }
  const data = await (await fetch(`${origin}/api/presentation`)).json();
  const start = Number(options.start || 0),
    fps = Number(options.fps || 24);
  const seconds = Math.min(
    Number(options.seconds || data.duration),
    data.duration - start,
  );
  if (!(start >= 0 && seconds > 0 && fps > 0))
    throw new Error("Invalid clip bounds");
  const audioFile = `${output}.mix.wav`;
  fs.writeFileSync(
    audioFile,
    Buffer.from(
      await (await fetch(`${origin}/api/presentation/audio.wav`)).arrayBuffer(),
    ),
  );
  browser = await chromium.launch({
    headless: true,
    executablePath:
      process.env.ARENA_CHROMIUM ||
      path.resolve(
        "work/playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell",
      ),
    args: [
      "--no-sandbox",
      "--use-gl=angle",
      "--use-angle=swiftshader",
      "--enable-unsafe-swiftshader",
      "--disable-dev-shm-usage",
    ],
  });
  const page = await browser.newPage({
    viewport: {
      width: Number(options.width || 1280),
      height: Number(options.height || 720),
    },
    deviceScaleFactor: 1,
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(`${origin}/?showcase&capture`);
  await page.waitForFunction(() => window.arenaReady);
  await page.evaluate(() => document.fonts.ready);
  await page.waitForFunction(n => document.querySelectorAll(".world-label").length === n, data.participants.length);
  await page.waitForTimeout(1000);
  if (await page.locator(".fallback-stage").count())
    throw new Error("3D renderer fell back to 2D");
  const encodeLog = fs.openSync(`${output}.encode.log`, "w");
  encoder = spawn(
    ffmpeg,
    [
      "-y",
      "-hide_banner",
      "-f",
      "image2pipe",
      "-framerate",
      String(fps),
      "-vcodec",
      "mjpeg",
      "-i",
      "pipe:0",
      "-ss",
      String(start),
      "-i",
      audioFile,
      "-map",
      "0:v:0",
      "-map",
      "1:a:0",
      "-t",
      String(seconds),
      "-c:v",
      "libx264",
      "-threads",
      "4",
      "-preset",
      "fast",
      "-crf",
      "18",
      "-pix_fmt",
      "yuv420p",
      "-af",
      "loudnorm=I=-18:TP=-1.5:LRA=11",
      "-ar",
      "48000",
      "-c:a",
      "aac",
      "-b:a",
      "192k",
      "-movflags",
      "+faststart",
      output,
    ],
    { stdio: ["pipe", "ignore", encodeLog] },
  );
  encoder.stdin.on("error", () => {});
  const finished = once(encoder, "close");
  const count = Math.ceil(seconds * fps);
  for (let i = 0; i < count; i++) {
    await page.evaluate((t) => window.arenaSeek(t), start + i / fps);
    const jpeg = await page.screenshot({ type: "jpeg", quality: 94 });
    if (encoder.exitCode !== null)
      throw new Error("Video encoder exited early");
    if (!encoder.stdin.write(jpeg)) await once(encoder.stdin, "drain");
    if (i === Math.floor(count / 3))
      await page.screenshot({ path: `${output}.png` });
    if (i % (fps * 5) === 0)
      console.log(`Rendered ${i}/${count} frames (${(i / fps).toFixed(1)}s)`);
  }
  encoder.stdin.end();
  const [exitCode] = await finished;
  if (exitCode !== 0 || errors.length)
    throw new Error(`Encoder=${exitCode}, browser=${errors.join("; ")}`);
  fs.writeFileSync(
    `${output}.json`,
    JSON.stringify(
      {
        recording: options.recording,
        start,
        seconds,
        fps,
        frames: count,
        width: Number(options.width || 1280),
        height: Number(options.height || 720),
        participants: data.participants,
        audio: data.audio_kind,
        mastering: "loudnorm -18 LUFS / -1.5 dBTP",
        captions: data.caption_kind,
        browser_errors: errors,
        job_id: process.env.SLURM_JOB_ID,
      },
      null,
      2,
    ),
  );
  console.log(`Saved ${output}`);
}
main()
  .catch((e) => {
    console.error(e);
    process.exitCode = 1;
  })
  .finally(async () => {
    if (browser) await browser.close();
    if (encoder && encoder.exitCode === null) encoder.kill("SIGTERM");
    server.kill("SIGTERM");
  });
