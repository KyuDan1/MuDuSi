/* Nine synchronized rooms + a single explicitly focused audio room, encoded in one bounded job. */
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
if (!options.matrix || !options.output)
  throw new Error(
    "Use --matrix MATRIX_DIR --output FILE.mp4 [--start 0 --seconds 60 --fps 24]",
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
    "--matrix",
    path.resolve(options.matrix),
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
  const data = await (await fetch(`${origin}/api/matrix`)).json();
  const start = Number(options.start || 0),
    fps = Number(options.fps || 24);
  const seconds = Math.min(
    Number(options.seconds || data.duration),
    data.duration - start,
  );
  if (!(start >= 0 && seconds > 0 && fps > 0))
    throw new Error("Invalid clip bounds");
  const focusSeconds = Number(options["focus-seconds"] || 6);
  const fixedFocus = options["audio-room"] ? data.rooms.findIndex(r => r.id === options["audio-room"]) : -1;
  if (!(focusSeconds > 0) || (options["audio-room"] && fixedFocus < 0)) throw new Error("Invalid audio focus");
  const audioFile = `${output}.focus.wav`;
  const roomAudio = [];
  for (const room of data.rooms) {
    const response = await fetch(`${origin}/api/matrix/audio/${encodeURIComponent(room.id)}.wav`);
    if (!response.ok) throw new Error(`Missing audio: ${room.id}`);
    const file = `${output}.${room.id}.wav`;
    fs.writeFileSync(file, Buffer.from(await response.arrayBuffer()));
    roomAudio.push(file);
  }
  // Select, never mix, independent rooms at their original common recording time.
  execFileSync(python, ["-c", `
import sys,wave,json
import numpy as np
paths=json.loads(sys.argv[1]);duration=float(sys.argv[3]);focus=float(sys.argv[4]);fixed=int(sys.argv[5]);sr=24000
tracks=[]
for path in paths:
 with wave.open(path) as w:
  assert w.getnchannels()==1 and w.getsampwidth()==2 and w.getframerate()==sr
  tracks.append(np.frombuffer(w.readframes(w.getnframes()),dtype='<i2'))
out=np.zeros(round(duration*sr),dtype='<i2')
step=max(1,round(focus*sr))
for lo in range(0,len(out),step):
 hi=min(len(out),lo+step);index=fixed if fixed>=0 else (lo//step)%len(tracks);source=tracks[index]
 end=min(hi,len(source))
 if end>lo:out[lo:end]=source[lo:end]
 # 5 ms boundary ramps avoid clicks; no cross-room audio overlap.
 if fixed<0:
  n=min(120,(hi-lo)//2)
  if lo:out[lo:lo+n]=(out[lo:lo+n].astype(float)*np.linspace(0,1,n)).astype('<i2')
  if hi<len(out):out[hi-n:hi]=(out[hi-n:hi].astype(float)*np.linspace(1,0,n)).astype('<i2')
with wave.open(sys.argv[2],'wb') as w:
 w.setparams((1,2,sr,0,'NONE','not compressed'));w.writeframes(out.tobytes())
`, JSON.stringify(roomAudio), audioFile, String(data.duration), String(focusSeconds), String(fixedFocus)]);
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
      width: Number(options.width || 1920),
      height: Number(options.height || 1080),
    },
    deviceScaleFactor: 1,
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  const viewParams = new URLSearchParams({ matrix: "", capture: "", focusSeconds: String(focusSeconds) });
  for (const key of ["title", "subtitle"])
    if (options[key]) viewParams.set(key, options[key]);
  await page.goto(`${origin}/?${viewParams}`);
  await page.waitForFunction(() => window.matrixReady);
  await page.evaluate(() => document.fonts.ready);
  await page.waitForFunction(
    (n) => document.querySelectorAll(".matrix-scene").length === n,
    data.rooms.filter(r => !r.error && r.recording?.participants.length === 2).length,
  );
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
    await page.evaluate((t) => window.matrixSeek(t.time, t.focus), {time: start + i / fps, focus: fixedFocus < 0 ? undefined : fixedFocus});
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
        recording: options.matrix,
        title: options.title,
        subtitle: options.subtitle,
        start,
        seconds,
        fps,
        frames: count,
        width: Number(options.width || 1920),
        height: Number(options.height || 1080),
        rooms: data.rooms.map(r => ({id:r.id,left:r.left,right:r.right})),
        focus_seconds: focusSeconds,
        fixed_audio_room: options["audio-room"] || null,
        audio: "one focused room at original timestamp; no cross-room mixing",
        mastering: "loudnorm -18 LUFS / -1.5 dBTP",
        captions: "offline word-timed emitted-audio ASR",
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
