import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  AudioLines,
  Box,
  Check,
  ChevronDown,
  CircleHelp,
  Expand,
  Focus,
  Headphones,
  Layers3,
  Link2,
  LoaderCircle,
  Mic,
  MicOff,
  MousePointer2,
  Pause,
  Play,
  Radio,
  RotateCcw,
  Settings2,
  SkipForward,
  SlidersHorizontal,
  Sparkles,
  Volume2,
  Waves,
  X,
} from "lucide-react";
import Arena from "./Arena";
import type { Agent, Control, Snapshot } from "./types";

const colors: Record<string, string> = {
  persona: "#c9ee83",
  moshi: "#b4a0ef",
  custom: "#efbc7d",
};
const scenarioLabels: Record<string, string> = {
  roundtable: "Roundtable",
  interruption: "Interruption",
  cocktail: "Cocktail party",
  quiet: "Quiet room",
};
const scenarios = [
  {
    id: "roundtable",
    name: "Roundtable",
    description: "Three voices. A shared space.",
    icon: "◎",
  },
  {
    id: "interruption",
    name: "Interruption",
    description: "Competing turns & barge-in.",
    icon: "↯",
  },
  {
    id: "cocktail",
    name: "Cocktail party",
    description: "Noise, overlap & an obstacle.",
    icon: "✳",
  },
  {
    id: "quiet",
    name: "Quiet room",
    description: "A clean acoustic baseline.",
    icon: "◌",
  },
];
function finite(value: number | undefined, digits = 1) {
  return Number.isFinite(value) ? value!.toFixed(digits) : "—";
}
function timeLabel(t: number) {
  const min = Math.floor(t / 60);
  return `${String(min).padStart(2, "0")}:${(t % 60).toFixed(2).padStart(5, "0")}`;
}
function dbPercent(db: number) {
  return Math.max(0, Math.min(100, ((db + 80) / 80) * 100));
}
function wsUrl(path: string) {
  return `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}${path}`;
}

function useSimulation() {
  const [state, setState] = useState<Snapshot | null>(null);
  const [status, setStatus] = useState<
    "connecting" | "connected" | "disconnected"
  >("connecting");
  const [error, setError] = useState("");
  const [history, setHistory] = useState<
    Array<{ time: number; levels: Record<string, number> }>
  >([]);
  const lastTick = useRef(-1);
  const apply = useCallback((next: Snapshot) => {
    if (!next.agents || typeof next.time !== "number") return;
    setState(next);
    if (next.tick !== lastTick.current) {
      lastTick.current = next.tick;
      setHistory((previous) => {
        const base = next.time < (previous.at(-1)?.time ?? 0) ? [] : previous;
        return [
          ...base,
          {
            time: next.time,
            levels: Object.fromEntries(next.agents.map((a) => [a.id, a.rx_db])),
          },
        ].slice(-160);
      });
    }
  }, []);
  useEffect(() => {
    let closed = false,
      timer: ReturnType<typeof setTimeout>,
      socket: WebSocket | undefined;
    const connect = () => {
      if (closed) return;
      setStatus("connecting");
      fetch("/api/state")
        .then((r) => {
          if (!r.ok) throw new Error("Simulation service unavailable");
          return r.json();
        })
        .then((next) => {
          if (!closed) apply(next);
        })
        .catch(() => {});
      socket = new WebSocket(wsUrl("/ws"));
      socket.onopen = () => {
        if (!closed) {
          setStatus("connected");
          setError("");
        }
      };
      socket.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          apply(data.state ?? data);
        } catch {
          setError("The simulation sent an unreadable update.");
        }
      };
      socket.onerror = () => socket?.close();
      socket.onclose = () => {
        if (!closed) {
          setStatus("disconnected");
          timer = setTimeout(connect, 2500);
        }
      };
    };
    connect();
    return () => {
      closed = true;
      clearTimeout(timer);
      socket?.close();
    };
  }, [apply]);
  const control = useCallback(
    async (command: Control) => {
      try {
        const response = await fetch("/api/control", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(command),
        });
        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          throw new Error(
            typeof body.detail === "string"
              ? body.detail
              : "That change could not be applied.",
          );
        }
        apply(await response.json());
        setError("");
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Unable to reach the simulation.",
        );
      }
    },
    [apply],
  );
  return { state, status, error, setError, control, history };
}

function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: () => void;
  label: string;
}) {
  return (
    <button
      className={`toggle ${checked ? "checked" : ""}`}
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={onChange}
    >
      <span />
    </button>
  );
}
function Meter({ value, color }: { value: number; color: string }) {
  return (
    <div className="meter-track">
      <div style={{ width: `${dbPercent(value)}%`, background: color }} />
    </div>
  );
}
function Portrait({ agent, small = false }: { agent: Agent; small?: boolean }) {
  return (
    <div
      className={`portrait ${small ? "small" : ""}`}
      style={
        { "--agent": agent.color || colors[agent.id] } as React.CSSProperties
      }
    >
      <span className="portrait-antenna" />
      <span className="portrait-head">
        <i />
        <i />
      </span>
      <span className="portrait-body" />
    </div>
  );
}
function SignalChart({
  history,
  selected,
  color,
}: {
  history: Array<{ time: number; levels: Record<string, number> }>;
  selected: string;
  color: string;
}) {
  const points = history
    .map(
      (p, i) =>
        `${(i / Math.max(1, history.length - 1)) * 500},${78 - dbPercent(p.levels[selected] ?? -80) * 0.7}`,
    )
    .join(" ");
  return (
    <div className="signal-chart">
      <div className="signal-scale">
        <span>0</span>
        <span>−40</span>
        <span>−80</span>
      </div>
      <svg
        viewBox="0 0 500 85"
        preserveAspectRatio="none"
        role="img"
        aria-label="Actual received RMS level history"
      >
        <defs>
          <linearGradient id="signal-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity=".2" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        {[8, 43, 78].map((y) => (
          <line
            key={y}
            x1="0"
            x2="500"
            y1={y}
            y2={y}
            stroke="#273038"
            strokeDasharray="2 5"
          />
        ))}
        {history.length > 1 && (
          <>
            <polygon
              points={`0,85 ${points} 500,85`}
              fill="url(#signal-fill)"
            />
            <polyline
              points={points}
              fill="none"
              stroke={color}
              strokeWidth="1.8"
              vectorEffect="non-scaling-stroke"
            />
          </>
        )}
      </svg>
      <span className="signal-unit">dBFS</span>
    </div>
  );
}

export default function App() {
  const { state, status, error, setError, control, history } = useSimulation();
  const [selected, setSelected] = useState("persona");
  const [waves, setWaves] = useState(true),
    [links, setLinks] = useState(true),
    [cameraKey, setCameraKey] = useState(0);
  const [monitor, setMonitor] = useState(false),
    [help, setHelp] = useState(false),
    [toast, setToast] = useState("");
  const audioContext = useRef<AudioContext | null>(null);
  const arenaRef = useRef<HTMLDivElement>(null);
  const enabled = status === "connected" && !!state;
  const agent =
    state?.agents.find((a) => a.id === selected) ?? state?.agents[0];
  const color = agent?.color || colors[agent?.id ?? "persona"] || "#c9ee83";
  const configure = (value: Omit<Control, "action">) =>
    void control({ action: "configure", ...value });
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(""), 3500);
    return () => clearTimeout(timer);
  }, [toast]);
  useEffect(() => {
    if (!monitor || !agent || !audioContext.current) return;
    const ctx = audioContext.current;
    let next = ctx.currentTime + 0.04;
    const nodes = new Set<AudioBufferSourceNode>();
    const socket = new WebSocket(
      wsUrl(`/ws/audio/${encodeURIComponent(agent.id)}`),
    );
    socket.binaryType = "arraybuffer";
    const gain = ctx.createGain();
    gain.gain.value = 0.6;
    gain.connect(ctx.destination);
    socket.onmessage = (e) => {
      if (
        !(e.data instanceof ArrayBuffer) ||
        e.data.byteLength === 0 ||
        e.data.byteLength % 4 !== 0
      )
        return;
      const pcm = new Float32Array(e.data);
      const buffer = ctx.createBuffer(
        1,
        pcm.length,
        state?.sample_rate ?? 24000,
      );
      buffer.copyToChannel(pcm, 0);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.playbackRate.value = state?.speed ?? 1;
      source.connect(gain);
      if (next > ctx.currentTime + 0.65) {
        nodes.forEach((n) => {
          try {
            n.stop();
          } catch {}
        });
        nodes.clear();
        next = ctx.currentTime + 0.025;
      }
      if (next < ctx.currentTime) next = ctx.currentTime + 0.025;
      source.start(next);
      next += buffer.duration / (state?.speed ?? 1);
      nodes.add(source);
      source.onended = () => nodes.delete(source);
    };
    socket.onerror = () => {
      setError(
        "Audio monitor connection failed. Try reconnecting the monitor.",
      );
      setMonitor(false);
    };
    return () => {
      socket.close();
      nodes.forEach((n) => {
        try {
          n.stop();
        } catch {}
      });
      gain.disconnect();
    };
  }, [
    monitor,
    agent?.id,
    state?.running,
    state?.seed,
    state?.sample_rate,
    state?.speed,
    setError,
  ]);
  const toggleMonitor = async () => {
    if (monitor) {
      setMonitor(false);
      return;
    }
    try {
      audioContext.current ??= new AudioContext();
      await audioContext.current.resume();
      setMonitor(true);
    } catch {
      setError("This browser could not start audio playback.");
    }
  };
  const download = async (path: string, filename: string) => {
    try {
      const response = await fetch(path);
      if (!response.ok)
        throw new Error(
          "No recording is available yet. Run a few simulation steps first.",
        );
      const url = URL.createObjectURL(await response.blob());
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      setToast("Export downloaded");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Export failed.");
    }
  };
  const changePosition = (axis: number, value: number) => {
    if (!agent) return;
    const position = [...agent.position] as [number, number, number];
    position[axis] = value;
    configure({ agents: [{ id: agent.id, position }] });
  };
  const incoming = state?.links.filter((l) => l.target === agent?.id) ?? [];
  const hasAsr = !!state?.asr?.enabled && (state.asr.agent_id === agent?.id || !!state.replay?.enabled);
  const events = [...(state?.events ?? []), ...(state?.asr_events ?? [])].sort(
    (a, b) => a.time - b.time,
  );
  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="./" aria-label="MuDuSi home">
          <span className="brand-symbol">
            <AudioLines size={22} />
          </span>
          <span>MuDuSi</span>
          <span className="version">0.1 / LAB</span>
        </a>
        <nav aria-label="Main navigation">
          <span className="nav-active">
            <Box size={14} /> Playground
          </span>
          <button onClick={() => setHelp(true)}>
            <CircleHelp size={14} /> About the lab
          </button>
        </nav>
        <div className={`connection ${status}`}>
          <span />
          {status === "connected"
            ? "Environment connected"
            : status === "connecting"
              ? "Connecting…"
              : "Reconnecting…"}
        </div>
      </header>
      <main>
        <div className="page-heading">
          <div>
            <div className="eyebrow">FULL-DUPLEX SPEECH SIMULATION</div>
            <h1>A space for conversations.</h1>
            <p>Place agents. Shape acoustics. See what they hear.</p>
          </div>
          <div className="heading-actions">
            <span
              className={`mode-badge ${state?.mode === "live" ? "live" : ""}`}
            >
              <span />
              {state?.replay?.enabled ? "Recorded model replay" : state?.mode === "live"
                ? "Live model workers"
                : "Synthetic demo · no GPU"}
            </span>
            <button
              className="button secondary export-button"
              disabled={!state}
              onClick={() =>
                void download(
                  "/api/export",
                  `duplex-trace-${state?.seed ?? 0}.json`,
                )
              }
            >
              <ArrowDownToLine size={15} /> Export episode
            </button>
          </div>
        </div>
        {(status === "disconnected" || error || state?.error) && (
          <div className="error-banner" role="alert">
            <Radio size={16} />
            <span>
              {error ||
                state?.error ||
                "The simulation service is offline. Start the backend on port 8765; this view will reconnect automatically."}
            </span>
            {error && (
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <X size={15} />
              </button>
            )}
          </div>
        )}
        <div className="workspace">
          <aside className="control-panel panel">
            <div className="panel-title">
              <SlidersHorizontal size={15} />
              <h2>Environment</h2>
              <span className="tiny-label">01</span>
            </div>
            <section className="control-section">
              <div className="section-label">
                SCENARIO <span>4 presets</span>
              </div>
              <div className="scenario-list">
                {scenarios.map((s) => (
                  <button
                    key={s.id}
                    className={`scenario ${state?.scenario === s.id ? "active" : ""}`}
                    disabled={!enabled}
                    onClick={() => configure({ scenario: s.id })}
                  >
                    <span className="scenario-icon">{s.icon}</span>
                    <span>
                      <strong>{s.name}</strong>
                      <small>{s.description}</small>
                    </span>
                    {state?.scenario === s.id && <Check size={13} />}
                  </button>
                ))}
              </div>
            </section>
            <section className="control-section acoustics">
              <div className="section-label">
                ACOUSTICS <Waves size={12} />
              </div>
              <label className="slider-label" htmlFor="noise">
                Ambient noise{" "}
                <span>{state ? `${finite(state.noise_db, 0)} dBFS` : "—"}</span>
              </label>
              <input
                id="noise"
                type="range"
                min="-90"
                max="-10"
                step="1"
                value={state?.noise_db ?? -55}
                disabled={!enabled}
                onChange={(e) =>
                  configure({ noise_db: Number(e.target.value) })
                }
              />
              <div className="slider-ends">
                <span>Quiet</span>
                <span>Busy</span>
              </div>
              <label className="slider-label" htmlFor="reverb">
                Wall reflection{" "}
                <span>{state ? finite(state.reverb, 2) : "—"}</span>
              </label>
              <input
                id="reverb"
                type="range"
                min="0"
                max="0.8"
                step="0.01"
                value={state?.reverb ?? 0.15}
                disabled={!enabled}
                onChange={(e) => configure({ reverb: Number(e.target.value) })}
              />
              <div className="slider-ends">
                <span>Dry</span>
                <span>Reflective</span>
              </div>
            </section>
            <section className="control-section">
              <div className="section-label">
                VIEW LAYERS <Layers3 size={12} />
              </div>
              <div className="toggle-row">
                <span>
                  <Waves size={14} /> Acoustic waves
                </span>
                <Toggle
                  checked={waves}
                  onChange={() => setWaves(!waves)}
                  label="Show acoustic waves"
                />
              </div>
              <div className="toggle-row">
                <span>
                  <Link2 size={14} /> Propagation paths
                </span>
                <Toggle
                  checked={links}
                  onChange={() => setLinks(!links)}
                  label="Show propagation paths"
                />
              </div>
            </section>
            <div className="room-info">
              <div>
                <Box size={13} />
                <span>Enclosed room</span>
              </div>
              <strong>
                {state
                  ? `${state.room.width} × ${state.room.depth} × ${state.room.height} m`
                  : "Waiting for environment"}
              </strong>
              <p>
                Distance attenuation · delay
                <br />
                Occlusion · wall reflection
              </p>
            </div>
            <div className="seed-row">
              <span>EPISODE SEED</span>
              <code>{state?.seed ?? "—"}</code>
              <button
                title="Start a new seeded episode"
                aria-label="New episode seed"
                disabled={!enabled}
                onClick={() =>
                  void control({
                    action: "reset",
                    seed: Math.floor(Math.random() * 100000),
                  })
                }
              >
                <RotateCcw size={12} />
              </button>
            </div>
          </aside>
          <section
            className="viewport panel"
            ref={arenaRef}
            aria-label="Three dimensional acoustic environment"
          >
            <div className="scene-top">
              <div>
                <span
                  className={`live-indicator ${state?.running ? "running" : ""}`}
                />
                <span>
                  {state?.running
                    ? "SIMULATION RUNNING"
                    : state?.done
                      ? "EPISODE COMPLETE"
                      : "SIMULATION PAUSED"}
                </span>
              </div>
              <div className="scene-top-right">
                <span>
                  {state
                    ? (scenarioLabels[state.scenario] ?? state.scenario)
                    : "Initializing"}
                </span>
                <span className="scene-divider" />
                <span>PERSPECTIVE</span>
              </div>
            </div>
            <div className="canvas-wrap">
              {state ? (
                <Arena
                  state={state}
                  selected={agent?.id ?? selected}
                  onSelect={setSelected}
                  waves={waves}
                  links={links}
                  cameraKey={cameraKey}
                />
              ) : (
                <div className="loading-scene">
                  <div className="loading-orbit">
                    <AudioLines size={30} />
                  </div>
                  <h3>Preparing the arena</h3>
                  <p>Waiting for the simulation service.</p>
                  <LoaderCircle className="spin" size={18} />
                </div>
              )}
            </div>
            <div className="scene-tools">
              <button
                title="Reset camera"
                aria-label="Reset camera"
                onClick={() => setCameraKey((k) => k + 1)}
              >
                <Focus size={17} />
              </button>
              <button
                title="Full screen arena"
                aria-label="Full screen arena"
                onClick={() => {
                  if (document.fullscreenElement)
                    void document.exitFullscreen();
                  else
                    void arenaRef.current
                      ?.requestFullscreen()
                      .catch(() =>
                        setToast("Full screen is unavailable in this browser."),
                      );
                }}
              >
                <Expand size={16} />
              </button>
            </div>
            <div className="scene-legend">
              <span>
                <i style={{ background: "#c9ee83" }} /> PersonaPlex
              </span>
              <span>
                <i style={{ background: "#b4a0ef" }} /> Moshi
              </span>
              <span>
                <i style={{ background: "#efbc7d" }} /> Custom SLM
              </span>
            </div>
            <div className="scene-caption">
              <MousePointer2 size={11} /> Drag to orbit · scroll to zoom · click
              an agent
            </div>
            <div className="visual-note">
              <Waves size={11} /> Wavefronts slowed for visibility; audio delay
              remains physical.
            </div>
            <div className="transport">
              <div className="transport-actions">
                <button
                  className={`run-button ${state?.running ? "is-running" : ""}`}
                  disabled={!enabled}
                  onClick={() =>
                    void control({
                      action: state?.done
                        ? "reset"
                        : state?.running
                          ? "pause"
                          : "play",
                    })
                  }
                >
                  {state?.running ? (
                    <Pause size={15} fill="currentColor" />
                  ) : (
                    <Play size={15} fill="currentColor" />
                  )}
                  {state?.done
                    ? "New episode"
                    : state?.running
                      ? "Pause"
                      : "Run simulation"}
                </button>
                <button
                  className="icon-button"
                  title="Advance one frame"
                  aria-label="Advance one frame"
                  disabled={!enabled || state?.running || state?.done}
                  onClick={() => void control({ action: "step" })}
                >
                  <SkipForward size={17} />
                </button>
                <button
                  className="icon-button"
                  title="Reset this episode"
                  aria-label="Reset this episode"
                  disabled={!enabled}
                  onClick={() => void control({ action: "reset" })}
                >
                  <RotateCcw size={15} />
                </button>
              </div>
              <div className="transport-clock">
                <span>{timeLabel(state?.time ?? 0)}</span>
                <small>SIM TIME</small>
              </div>
              <div className="speed-control">
                <select
                  aria-label="Simulation playback speed"
                  value={state?.speed ?? 1}
                  disabled={!enabled}
                  onChange={(e) => configure({ speed: Number(e.target.value) })}
                >
                  <option value={0.5} disabled={state?.mode === "live"}>
                    0.5×
                  </option>
                  <option value={1}>1.0×</option>
                  <option value={2} disabled={state?.mode === "live"}>
                    2.0×
                  </option>
                </select>
                <ChevronDown size={11} />
              </div>
            </div>
          </section>
          <aside className="inspector panel">
            <div className="panel-title">
              <Activity size={15} />
              <h2>Agent inspector</h2>
              <span className="tiny-label">02</span>
            </div>
            <div
              className="agent-tabs"
              role="tablist"
              aria-label="Select agent"
            >
              {state?.agents.map((a) => (
                <button
                  key={a.id}
                  role="tab"
                  aria-selected={agent?.id === a.id}
                  className={agent?.id === a.id ? "active" : ""}
                  onClick={() => setSelected(a.id)}
                  style={
                    {
                      "--agent": a.color || colors[a.id],
                    } as React.CSSProperties
                  }
                >
                  <span />
                  {a.id === "persona" ? "Persona" : a.name.split(" ")[0]}
                </button>
              ))}
            </div>
            {agent ? (
              <>
                <div className="agent-identity">
                  <Portrait agent={agent} />
                  <div>
                    <h3>{agent.name}</h3>
                    <span>{agent.model || "Synthetic speech adapter"}</span>
                    <div className="agent-state" style={{ color }}>
                      <i />
                      {agent.muted
                        ? "Muted"
                        : agent.speaking
                          ? agent.listening
                            ? "Speaking + listening"
                            : "Speaking"
                          : agent.listening
                            ? "Listening"
                            : "Idle"}
                    </div>
                  </div>
                </div>
                <div className="audio-status">
                  <div className="level-row">
                    <span>
                      <Mic size={12} /> Transmit
                    </span>
                    <strong>
                      {finite(agent.tx_db)}
                      <small> dBFS</small>
                    </strong>
                  </div>
                  <Meter value={agent.tx_db} color={color} />
                  <div className="level-row">
                    <span>
                      <Headphones size={12} /> Receive
                    </span>
                    <strong>
                      {finite(agent.rx_db)}
                      <small> dBFS</small>
                    </strong>
                  </div>
                  <Meter value={agent.rx_db} color="#8caeaa" />
                </div>
                <div className="transcript-box">
                  <div className="section-label">
                    {agent.transcript_kind === "asr"
                      ? "ASR TRANSCRIPT"
                      : agent.transcript_kind === "model"
                        ? "MODEL TRANSCRIPT"
                        : "SCRIPTED CAPTION"}
                    <span className="caption-tag">
                      {state?.replay?.enabled ? "RECORDED" : agent.transcript_kind === "scripted" ? "DEMO" : "LIVE"}
                    </span>
                  </div>
                  <p>{agent.transcript || "No utterance in this frame."}</p>
                  <div className="asr-status">
                    <span />
                    {hasAsr
                      ? state?.asr?.error
                        ? "Receiver ASR error"
                        : state?.asr?.ready
                          ? state?.replay?.enabled ? "Offline receiver ASR" : "Receiver ASR connected"
                          : "Receiver ASR loading…"
                      : "ASR not connected"}
                  </div>
                  {hasAsr && (
                    <div className="receiver-asr">
                      <div className="section-label">
                        RECEIVER ASR
                        <span>
                          {state?.replay?.enabled ? "OFFLINE" : state?.asr?.lag_seconds !== undefined
                            ? `${finite(state.asr.lag_seconds)}s lag`
                            : "WINDOWED"}
                        </span>
                      </div>
                      <p>
                        {agent.asr_transcript ||
                          state?.asr?.error ||
                          "Waiting for a recognized speech window."}
                      </p>
                    </div>
                  )}
                </div>
                <div className="inspector-section">
                  <div className="section-label">
                    LISTENING PATHS <span>{incoming.length} peers</span>
                  </div>
                  {incoming.map((link) => (
                    <div className="path-row" key={link.source}>
                      <span
                        className="path-dot"
                        style={{ background: colors[link.source] ?? "#aaa" }}
                      />
                      <span>
                        {state?.agents.find((a) => a.id === link.source)
                          ?.name ?? link.source}
                      </span>
                      <span className={link.occluded ? "occluded" : ""}>
                        {link.occluded ? "Blocked · " : ""}
                        {finite(link.gain_db)} dB
                      </span>
                    </div>
                  ))}
                </div>
                {state?.mode === "live" && agent.worker && (
                  <div className="worker-status">
                    <span>
                      Worker {agent.worker.ready ? "ready" : "waiting"}
                    </span>
                    <span>
                      {agent.worker.underrun_frames ?? 0} underrun frames
                    </span>
                  </div>
                )}
                <div className="inspector-section position-section">
                  <div className="section-label">
                    POSITION <span>meters</span>
                  </div>
                  {[
                    { label: "X", axis: 0, min: -5.5, max: 5.5 },
                    { label: "Z", axis: 2, min: -4, max: 4 },
                  ].map(({ label, axis, min, max }) => (
                    <div className="position-row" key={axis}>
                      <label htmlFor={`pos-${axis}`}>{label}</label>
                      <input
                        id={`pos-${axis}`}
                        type="range"
                        min={min}
                        max={max}
                        step=".1"
                        value={agent.position[axis]}
                        disabled={!enabled}
                        onChange={(e) =>
                          changePosition(axis, Number(e.target.value))
                        }
                      />
                      <output>{finite(agent.position[axis])}</output>
                    </div>
                  ))}
                </div>
                <div className="inspector-actions">
                  <button
                    className={monitor ? "monitor active" : "monitor"}
                    disabled={!enabled}
                    onClick={() => void toggleMonitor()}
                  >
                    <Headphones size={14} />
                    {monitor ? "Monitoring receiver" : "Hear receiver"}
                  </button>
                  <button
                    className={agent.muted ? "mute active" : "mute"}
                    title={agent.muted ? "Unmute agent" : "Mute agent"}
                    aria-label={agent.muted ? "Unmute agent" : "Mute agent"}
                    disabled={!enabled}
                    onClick={() =>
                      configure({
                        agents: [{ id: agent.id, muted: !agent.muted }],
                      })
                    }
                  >
                    {agent.muted ? <MicOff size={15} /> : <Mic size={15} />}
                  </button>
                  <button
                    title="Download receiver WAV"
                    aria-label="Download receiver WAV"
                    onClick={() =>
                      void download(
                        `/api/audio/${encodeURIComponent(agent.id)}.wav`,
                        `${agent.id}-receiver.wav`,
                      )
                    }
                  >
                    <ArrowDownToLine size={15} />
                  </button>
                </div>
                <div className="audio-disclosure">
                  {state?.mode === "demo"
                    ? "Demo audio uses synthetic tones, not model speech."
                    : "Audio is the selected agent’s received mixture."}
                </div>
              </>
            ) : (
              <div className="inspector-empty">
                Agent telemetry appears when the environment connects.
              </div>
            )}
          </aside>
        </div>
        <div className="telemetry-row">
          <section className="timeline panel">
            <div className="bottom-panel-title">
              <h2>
                <AudioLines size={15} /> Conversation timeline
              </h2>
              <span>
                <span className="small-dot" />{" "}
                {state?.asr?.enabled
                  ? "Turns + receiver ASR"
                  : state?.mode === "demo"
                    ? "Scripted events"
                    : "Environment events"}
              </span>
            </div>
            <div className="timeline-body">
              {state && events.length ? (
                <>
                  {events
                    .slice(-8)
                    .reverse()
                    .map((event, i) => {
                      const speaker = state.agents.find(
                        (a) => a.id === event.agent_id,
                      );
                      return (
                        <div
                          className="timeline-event"
                          key={`${event.id}-${i}`}
                        >
                          <time>{timeLabel(event.time)}</time>
                          <span
                            className="event-dot"
                            style={{
                              background:
                                speaker?.color ||
                                colors[event.agent_id] ||
                                "#7494a0",
                            }}
                          />
                          <strong>
                            {speaker?.name ?? event.agent_id ?? "Environment"}
                          </strong>
                          <span className="event-text">
                            {event.text || event.type}
                          </span>
                          <small>{event.kind ?? event.type}</small>
                        </div>
                      );
                    })}
                </>
              ) : (
                <div className="timeline-empty">
                  <Radio size={21} />
                  <div>
                    <strong>The room is listening.</strong>
                    <span>
                      Run the simulation to see turns, interruptions, and
                      acoustic events.
                    </span>
                  </div>
                </div>
              )}
            </div>
          </section>
          <section className="signal-panel panel">
            <div className="bottom-panel-title">
              <h2>
                <Volume2 size={14} /> Receiver signal
              </h2>
              <span style={{ color }}>{agent?.name ?? "—"}</span>
            </div>
            <SignalChart
              history={history}
              selected={agent?.id ?? selected}
              color={color}
            />
            <div className="signal-footer">
              <span>RMS envelope · actual mixed audio</span>
              <strong>
                {state ? `${(state.sample_rate / 1000).toFixed(0)} kHz` : "—"}
              </strong>
            </div>
          </section>
        </div>
        <div className="metrics-row">
          <div className="metric">
            <span>
              <AudioLines size={13} /> ACTIVE SPEAKERS
            </span>
            <strong>
              {state?.metrics.active_speakers ?? "—"}
              <small>/ {state?.agents.length ?? 3}</small>
            </strong>
          </div>
          <div className="metric">
            <span>
              <Waves size={13} /> SPEECH OVERLAP
            </span>
            <strong>
              {state ? finite(state.metrics.overlap_ratio * 100, 1) : "—"}
              <small>%</small>
            </strong>
          </div>
          <div className="metric">
            <span>
              <Headphones size={13} /> MEAN RECEIVE LEVEL
            </span>
            <strong>
              {finite(state?.metrics.mean_rx_db)}
              <small>dBFS</small>
            </strong>
          </div>
          <div className="metric reward-metric">
            <span>
              <Sparkles size={13} /> EPISODE REWARD{" "}
              <span title="Synthetic proxy reward for audibility, overlap and energy. It is not a learned quality assessment.">
                <CircleHelp size={11} />
              </span>
            </span>
            <strong>
              {finite(state?.metrics.episode_reward, 3)}
              <small>proxy</small>
            </strong>
          </div>
          <div className="simulation-meta">
            <span>
              <i /> {state ? `${state.frame_ms} ms frames` : "Awaiting state"}
            </span>
            <span>Tick {state?.tick ?? 0} · reproducible seed</span>
          </div>
        </div>
        <footer>
          <span>
            MuDuSi <i>/</i> Multi-agent Duplex Simulator.
          </span>
          <span>
            {state?.replay?.enabled ? "Recorded batch experiment · controls cannot change recorded acoustics" : state?.mode === "live"
              ? "Model adapters connected"
              : "Synthetic adapters · connect real models on your GPU server"}
            <span className="footer-dot" />
          </span>
        </footer>
      </main>
      {toast && (
        <div className="toast" role="status">
          <Check size={15} />
          {toast}
        </div>
      )}
      {help && (
        <div className="modal-backdrop" onClick={() => setHelp(false)}>
          <section
            className="about-modal"
            role="dialog"
            aria-modal="true"
            aria-label="About MuDuSi"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="modal-close"
              aria-label="Close about dialog"
              onClick={() => setHelp(false)}
            >
              <X size={20} />
            </button>
            <span className="eyebrow">A WORLD THAT LISTENS</span>
            <h2>Conversation has a physical space.</h2>
            <p>
              MuDuSi lets full-duplex speech agents speak and listen in
              the same simulated room. Each receiver gets a distinct mixture
              shaped by distance, propagation delay, obstructions, background
              noise, and a first-order wall reflection.
            </p>
            <div className="about-facts">
              <div>
                <Box size={20} />
                <strong>Spatial simulation</strong>
                <span>
                  Select a robot and move it with the X/Z controls. The room
                  changes what it hears.
                </span>
              </div>
              <div>
                <Headphones size={20} />
                <strong>Observable audio</strong>
                <span>
                  Monitor a receiver, inspect real signal levels, and export its
                  audio plus the episode trace.
                </span>
              </div>
              <div>
                <Settings2 size={20} />
                <strong>Built for experiments</strong>
                <span>
                  Seeds reproduce episodes. Scenario presets expose overlap and
                  acoustic challenges.
                </span>
              </div>
            </div>
            <p className="about-disclosure">
              The local demo generates synthetic tones and scripted captions.
              PersonaPlex and Moshi labels describe intended adapters; these
              models are not loaded in demo mode. Proxy rewards are diagnostics,
              not a validated speech-quality score. Visual wavefronts are
              deliberately slowed; actual audio uses physical propagation
              delays.
            </p>
            <button className="button primary" onClick={() => setHelp(false)}>
              Back to the arena
            </button>
          </section>
        </div>
      )}
    </div>
  );
}
