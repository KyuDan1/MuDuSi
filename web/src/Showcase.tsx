import { agentColors } from "./theme";
import { useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import Arena from "./Arena";
import type { Snapshot } from "./types";
import "./showcase.css";

type Word = { start: number; end: number; word: string };
type Recording = {
  frames: Snapshot[];
  participants: string[];
  words: Record<string, Word[]>;
  duration: number;
  summary: { deadline_lateness_p95_ms: number };
};
declare global {
  interface Window {
    arenaSeek?: (t: number) => Promise<void>;
    arenaReady?: boolean;
  }
}
const colors = agentColors;
export default function Showcase() {
  const [record, setRecord] = useState<Recording>();
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState("");
  const audio = useRef<HTMLAudioElement>(null);
  const params = new URLSearchParams(location.search);
  const capture = params.has("capture");
  useEffect(() => {
    fetch("/api/presentation")
      .then((r) => {
        if (!r.ok)
          throw new Error("Open a completed model recording with --replay.");
        return r.json();
      })
      .then(setRecord)
      .catch((e) => setError(String(e)));
  }, []);
  useEffect(() => {
    if (!record) return;
    window.arenaSeek = async (t: number) => {
      flushSync(() =>
        setTime(Math.max(0, Math.min(t, record.duration - 0.001))),
      );
      await new Promise<void>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
      );
    };
    window.arenaReady = true;
    return () => {
      delete window.arenaSeek;
      window.arenaReady = false;
    };
  }, [record]);
  useEffect(() => {
    if (!playing) return;
    let frame: number;
    const animate = () => {
      if (audio.current) setTime(audio.current.currentTime);
      frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [playing]);
  if (!record)
    return (
      <div className="showcase-loading">
        {error || "Loading real model conversation…"}
      </div>
    );
  const original =
    record.frames[Math.min(record.frames.length - 1, Math.floor(time / 0.08))];
  const state = {
    ...original,
    time,
    running: true,
    agents: original.agents.filter((a) => record.participants.includes(a.id)),
    links: original.links.filter(
      (l) =>
        record.participants.includes(l.source) &&
        record.participants.includes(l.target),
    ),
  };
  const speaking = state.agents.filter((a) => a.speaking).length;
  const captions: Record<string, string> = {};
  for (const id of record.participants) {
    const words = record.words[id] || [];
    let spoken = words.filter((w) => w.start <= time && w.end >= time - 5);
    let phraseStart = 0;
    for (let i = 1; i < spoken.length; i++) {
      if (spoken[i].start - spoken[i - 1].end > 1.2) phraseStart = i;
    }
    spoken = spoken.slice(phraseStart);
    // Let the last phrase linger briefly, but never reveal future words.
    const last = spoken.at(-1);
    captions[id] =
      last && time - last.end < 1.8
        ? spoken
            .slice(-16)
            .map((w) => w.word.trim())
            .join(" ")
        : "";
  }
  const allUnderruns = state.agents.reduce(
    (n, a) => n + (a.worker?.underrun_frames || 0),
    0,
  );
  return (
    <main className="showcase">
      <header className="showcase-header">
        <div>
          <h1>
            {params.get("title") ||
              (state.agents.length === 2
                ? "Two friends"
                : "Three-way conversation")}
          </h1>
          <p>
            {params.get("subtitle") ||
              (state.agents.length === 2
                ? "A conversation between friends"
                : "Three people, one conversation")}{" "}
            <span>Listen · speak · overlap</span>
          </p>
        </div>
        <div className="showcase-time">
          <b>
            {time.toFixed(2).padStart(5, "0")}
            <small>s</small>
          </b>
          <span>24 kHz PCM · 80 ms ticks</span>
          <em className={speaking > 1 ? "overlap" : ""}>
            {speaking > 1
              ? `${speaking} VOICES OVERLAPPING`
              : speaking === 1
                ? "SPEAKING + LISTENING"
                : "LISTENING"}
          </em>
        </div>
      </header>
      <section className="showcase-world">
        <Arena
          state={state}
          selected=""
          onSelect={() => {}}
          waves
          links
          cameraKey={0}
          cinematic
        />
      </section>
      <div className="showcase-captions">
        {state.agents.map(
          (a) =>
            captions[a.id] && (
              <div
                key={a.id}
                className={`showcase-bubble bubble-${a.id}`}
                style={{ "--agent": colors[a.id] } as React.CSSProperties}
              >
                <small>{a.name.toUpperCase()} · AUDIO ASR</small>
                {captions[a.id]}
              </div>
            ),
        )}
      </div>
      <div className="showcase-legend">
        REAL GENERATED AUDIO <i /> WORD-TIMED ASR REPLAY <i /> SLOW ORBIT
      </div>
      <footer className="showcase-footer">
        <div className="showcase-tracks">
          {state.agents.map((a) => (
            <div
              key={a.id}
              className="showcase-track"
              style={{ "--agent": colors[a.id] } as React.CSSProperties}
            >
              <div className="track-title">
                <b>{a.name}</b>
                <span>
                  TX {a.tx_db.toFixed(0)} dB · RX {a.rx_db.toFixed(0)} dB
                </span>
              </div>
              <div className="track-activity">
                {record.frames
                  .filter((_, i) => i % 5 === 0)
                  .map((f, i) => (
                    <i
                      key={i}
                      style={{
                        opacity: f.agents.find((x) => x.id === a.id)?.speaking
                          ? 0.9
                          : 0.12,
                        background: i * 0.4 <= time ? colors[a.id] : "#52605e",
                      }}
                    />
                  ))}
                <strong
                  style={{ left: `${(time / record.duration) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
        <aside>
          <b>{state.agents.length} independent agents</b>
          <span>Speech → room acoustics → peer input</span>
          <span>
            Propagation:{" "}
            {Math.min(...state.links.map((l) => l.delay_ms)).toFixed(1)}–
            {Math.max(...state.links.map((l) => l.delay_ms)).toFixed(1)} ms
          </span>
          <span>
            Scheduler p95: {record.summary.deadline_lateness_p95_ms.toFixed(1)}{" "}
            ms late
          </span>
          <span>Buffer underruns: {allUnderruns} frames, cumulative</span>
          <small>Audio: clean speaker mix · Captions: offline ASR</small>
        </aside>
      </footer>
      {!capture && (
        <nav className="showcase-controls">
          <button
            onClick={async () => {
              if (!audio.current) return;
              if (playing) {
                audio.current.pause();
                setPlaying(false);
              } else {
                await audio.current.play();
                setPlaying(true);
              }
            }}
          >
            {playing ? "Pause" : "▶ Play with audio"}
          </button>
          <input
            aria-label="Playback time"
            type="range"
            min={0}
            max={record.duration}
            step={0.08}
            value={time}
            onChange={(e) => {
              const t = Number(e.target.value);
              setTime(t);
              if (audio.current) audio.current.currentTime = t;
            }}
          />
          <a href="/">Engineering view</a>
        </nav>
      )}
      <audio
        ref={audio}
        src="/api/presentation/audio.wav"
        onEnded={() => setPlaying(false)}
      />
    </main>
  );
}
