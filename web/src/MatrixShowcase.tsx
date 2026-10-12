import { Fragment, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { Canvas, useThree } from "@react-three/fiber";
import { PerspectiveCamera, RoundedBox, View } from "@react-three/drei";
import type { Agent, Snapshot } from "./types";
import "./matrix.css";

type Word = { start: number; end: number; word: string };
type MatrixRoom = {
  id: string; left: string; right: string; error?: string;
  recording?: { frames: Snapshot[]; participants: string[]; words: Record<string, Word[]>;
    duration: number; summary?: { deadline_lateness_p95_ms?: number }; };
};
type MatrixData = { duration: number; model_order: string[]; rooms: MatrixRoom[];
  concurrency?: { completed?: boolean; simultaneous_room_count?: number; start_spread_ms?: number; playback_rate?: number } };
declare global { interface Window {
  matrixReady?: boolean;
  matrixSeek?: (time: number, focusRoomIndex?: number) => Promise<void>;
} }
const sage = "#B7D7C0", lavender = "#C9C0E8";

function caption(words: Word[] = [], time: number) {
  let visible = words.filter(w => w.start <= time && w.end >= time - 4);
  let boundary = 0;
  for (let i = 1; i < visible.length; i++) if (visible[i].start - visible[i - 1].end > 1.2) boundary = i;
  visible = visible.slice(boundary);
  return visible.length && time - visible[visible.length - 1].end < 1.8
    ? visible.slice(-11).map(w => w.word.trim()).join(" ") : "";
}

function PairRobot({ agent, partner, color, time }: { agent: Agent; partner: Agent; color: string; time: number }) {
  const yaw = Math.atan2(partner.position[0] - agent.position[0], partner.position[2] - agent.position[2]);
  const bob = agent.speaking ? Math.sin(time * 8) * .035 : 0;
  return <group position={[agent.position[0], 0, agent.position[2]]}>
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, .015, 0]}>
      <ringGeometry args={[.6, .63, 24]} /><meshBasicMaterial color={color} transparent opacity={agent.speaking ? .9 : .3} />
    </mesh>
    {agent.speaking && [0, .5].map(offset => {
      const phase = (time * .7 + offset) % 1;
      return <mesh key={offset} rotation={[-Math.PI / 2, 0, 0]} position={[0, .02, 0]} scale={1 + phase * 2.4}>
        <ringGeometry args={[.63, .65, 32]} /><meshBasicMaterial color={color} transparent opacity={(1 - phase) * .45} depthWrite={false} />
      </mesh>;
    })}
    <group rotation={[0, yaw, 0]} position={[0, agent.position[1] - 1.2 + bob, 0]}>
      <RoundedBox args={[.55, .43, .42]} radius={.10} smoothness={2} position={[0, .58, 0]}><meshStandardMaterial color={color} roughness={.55} /></RoundedBox>
      <RoundedBox args={[.8, .61, .56]} radius={.14} smoothness={2} position={[0, 1.14, 0]}><meshStandardMaterial color={color} roughness={.45} /></RoundedBox>
      <RoundedBox args={[.62, .32, .045]} radius={.065} smoothness={2} position={[0, 1.15, .285]}><meshStandardMaterial color="#14221c" /></RoundedBox>
      {[-.16, .16].map(x => <mesh key={x} position={[x, 1.19, .315]}><sphereGeometry args={[.037, 8, 6]} /><meshBasicMaterial color={color} /></mesh>)}
      <mesh position={[0, 1.07, .316]}><boxGeometry args={[agent.speaking ? .2 : .09, .025, .012]} /><meshBasicMaterial color={color} /></mesh>
      {[-.34, .34].map(x => <mesh key={x} position={[x, .6, 0]} rotation={[0, 0, x > 0 ? -.12 : .12]}><capsuleGeometry args={[.075, .24, 3, 8]} /><meshStandardMaterial color={color} /></mesh>)}
      {[-.16, .16].map(x => <mesh key={x} position={[x, .21, .015]}><boxGeometry args={[.15, .3, .21]} /><meshStandardMaterial color={color} /></mesh>)}
    </group>
  </group>;
}
function PairScene({ agents, time }: { agents: Agent[]; time: number }) {
  const centerX = agents.reduce((sum, a) => sum + a.position[0], 0) / 2;
  const centerZ = agents.reduce((sum, a) => sum + a.position[2], 0) / 2;
  const distance = Math.max(2.5, Math.hypot(agents[0].position[0] - agents[1].position[0], agents[0].position[2] - agents[1].position[2]));
  const angle = .10 + Math.sin(time * .025) * .1;
  const radius = Math.max(4.8, distance * 1.6);
  return <>
    <PerspectiveCamera makeDefault fov={30} position={[centerX + Math.sin(angle) * radius, 2.8, centerZ + Math.cos(angle) * radius]} onUpdate={c => c.lookAt(centerX, .65, centerZ)} />
    <color attach="background" args={["#111B18"]} />
    <ambientLight intensity={1.4} /><directionalLight position={[2, 7, 4]} intensity={2} />
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[centerX, -.03, centerZ]}><planeGeometry args={[distance + 4, 5]} /><meshStandardMaterial color="#22352D" /></mesh>
    <gridHelper args={[10, 16, "#53695b", "#384E42"]} position={[centerX, -.015, centerZ]} />
    {agents.map((a, i) => <PairRobot key={a.id} agent={a} partner={agents[1 - i]} color={i === 0 ? sage : lavender} time={time} />)}
  </>;
}
function RenderPulse({ time, focus }: { time: number; focus: number }) {
  const invalidate = useThree(s => s.invalidate);
  useEffect(() => { invalidate(); }, [time, focus, invalidate]);
  return null;
}
function Room({ room, time, focused, choose }: { room: MatrixRoom; time: number; focused: boolean; choose: () => void }) {
  const record = room.recording;
  const frame = record?.frames[Math.min(record.frames.length - 1, Math.floor(time / .08))];
  const agents = frame?.agents.filter(a => record!.participants.includes(a.id)).sort((a, b) => record!.participants.indexOf(a.id) - record!.participants.indexOf(b.id)) ?? [];
  const ended = record ? time >= record.duration : false;
  const status = room.error ? "RUN FAILED" : ended ? "RECORDING ENDED" : agents.filter(a => a.speaking).length > 1 ? "OVERLAP" : "LISTENING / SPEAKING";
  return <article className={`matrix-room${focused ? " focused" : ""}`} data-room={room.id} aria-label={`${room.left} with ${room.right}`}>
    <button className="matrix-room-heading" onClick={choose} aria-pressed={focused}><span>{room.left} <b>↔</b> {room.right}</span><em>{focused ? "AUDIO ON" : status}</em></button>
    {agents.length === 2 && !room.error ? <View className="matrix-scene"><PairScene agents={agents} time={Math.min(time, record!.duration)} /></View> : <div className="matrix-room-error">{room.error || "No completed recording"}</div>}
    <div className="matrix-room-captions">
      {agents.map((a, i) => {
        const text = caption(record?.words?.[a.id], time);
        return <div key={a.id} className={`matrix-caption${a.speaking && !ended ? " speaking" : ""}`} style={{ "--speaker": i ? lavender : sage } as React.CSSProperties}>
          <span className="matrix-speaker">{i ? room.right : room.left}<i>{a.speaking && !ended ? "SPEAKING" : "LISTENING"}</i></span>
          <p>{text || (room.error ? "Unavailable" : " ")}</p>
          <small>TX {a.tx_db.toFixed(0)} · RX {a.rx_db.toFixed(0)} dBFS</small>
        </div>;
      })}
    </div>
    <span className="matrix-room-timing">{frame?.links.find(l => record?.participants.includes(l.source) && record?.participants.includes(l.target))?.delay_ms.toFixed(1) ?? "—"} ms path · {record?.summary?.deadline_lateness_p95_ms?.toFixed(1) ?? "—"} ms tick p95</span>
  </article>;
}
export default function MatrixShowcase() {
  const [data, setData] = useState<MatrixData>();
  const [time, setTime] = useState(0), [focus, setFocus] = useState(0), [playing, setPlaying] = useState(false), [error, setError] = useState("");
  const [audioSource, setAudioSource] = useState<{room: string; url: string}>();
  const audio = useRef<HTMLAudioElement>(null), clock = useRef({ start: 0, offset: 0 });
  const params = new URLSearchParams(location.search), capture = params.has("capture");
  const focusSeconds = Math.max(.1, Number(params.get("focusSeconds") || 6));
  useEffect(() => { fetch("/api/matrix").then(r => { if (!r.ok) throw new Error("Open recorded rooms with --matrix."); return r.json(); }).then(setData).catch(e => setError(String(e))); }, []);
  useEffect(() => {
    if (!data) return;
    window.matrixSeek = async (t, selected) => {
      flushSync(() => { setTime(Math.max(0, Math.min(t, data.duration - .001))); setFocus(selected ?? Math.floor(t / focusSeconds) % data.rooms.length); });
      await new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
    };
    window.matrixReady = true;
    return () => { window.matrixReady = false; delete window.matrixSeek; };
  }, [data, focusSeconds]);
  useEffect(() => {
    if (!playing || !data) return;
    clock.current = { start: performance.now(), offset: time };
    let request: number;
    const tick = () => { const t = clock.current.offset + (performance.now() - clock.current.start) / 1000; if (t >= data.duration) { setPlaying(false); return; } setTime(t); request = requestAnimationFrame(tick); };
    request = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(request);
  // Time is sampled only when playback starts; changing it must not reset the clock.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, data]);
  useEffect(() => {
    if (!data) return;
    const controller = new AbortController();
    let url: string | undefined;
    const id = data.rooms[focus].id;
    // A complete Blob makes WAV seeking reliable even when an API lacks HTTP Range support.
    fetch(`/api/matrix/audio/${encodeURIComponent(id)}.wav`, { signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error(`Audio unavailable: ${id}`); return response.blob(); })
      .then(blob => { if (controller.signal.aborted) return; url = URL.createObjectURL(blob); setAudioSource({room: id, url}); })
      .catch(e => { if (!controller.signal.aborted) setError(String(e)); });
    return () => { controller.abort(); if (url) URL.revokeObjectURL(url); };
  }, [focus, data]);
  useEffect(() => {
    const element = audio.current;
    if (!element || !data || audioSource?.room !== data.rooms[focus].id) return;
    const synchronize = () => {
      const current = playing ? clock.current.offset + (performance.now() - clock.current.start) / 1000 : time;
      element.currentTime = Math.min(current, data.rooms[focus].recording?.duration ?? current);
      if (playing) element.play().catch(e => { setError(`Audio playback: ${e}`); setPlaying(false); });
      else element.pause();
    };
    if (element.readyState >= 1) synchronize();
    else element.addEventListener("loadedmetadata", synchronize, { once: true });
    return () => element.removeEventListener("loadedmetadata", synchronize);
  // The audio clock runs independently between room changes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus, playing, data, audioSource]);
  if (!data) return <div className="matrix-loading">{error || "Loading nine recorded rooms…"}</div>;
  const room = data.rooms[focus];
  return <main className="matrix-showcase">
    <header className="matrix-header"><div><h1>Nine rooms. Eighteen voices.</h1><p>MuDuSi · All ordered pairs of Moshi, PersonaPlex and Raon</p></div><div className="matrix-clock"><strong>{time.toFixed(2)}<small> s</small></strong><span>Simulation time · 80 ms audio ticks</span>{data.concurrency?.completed && data.concurrency.simultaneous_room_count === 9 && <span>9 concurrent rooms · {data.concurrency.start_spread_ms?.toFixed(1)} ms start spread</span>}</div></header>
    <div className="matrix-axis"><span>LEFT ↓ / RIGHT →</span>{data.model_order.map(name => <b key={name}>{name}</b>)}</div>
    <section className="matrix-board" aria-label="Nine independent acoustic rooms">
      <Canvas className="matrix-canvas" frameloop="demand" dpr={1} gl={{ antialias: true, alpha: true }}><View.Port /><RenderPulse time={time} focus={focus} /></Canvas>
      {data.model_order.map(left => <Fragment key={left}><div className="matrix-row-label">{left}</div>{data.model_order.map(right => {
        const index = data.rooms.findIndex(r => r.left === left && r.right === right);
        return index < 0 ? <div className="matrix-room-error" key={right}>Missing room</div> : <Room key={right} room={data.rooms[index]} time={time} focused={focus === index} choose={() => setFocus(index)} />;
      })}</Fragment>)}
    </section>
    <footer className="matrix-footer"><div><span className="matrix-audio-dot" />Listening to <strong>{room?.left} ↔ {room?.right}</strong><span className="matrix-isolation">Other rooms are silent in the soundtrack · Rooms do not hear each other</span></div><span>{data.concurrency?.playback_rate ?? 1}× collection · Normal-speed audio · Offline ASR</span></footer>
    {!capture && <div className="matrix-controls"><button onClick={() => setPlaying(!playing)}>{playing ? "Pause" : "Play selected room"}</button><input aria-label="Recording time" type="range" min="0" max={data.duration - .01} step=".08" value={time} onChange={e => { const t = Number(e.target.value); setTime(t); clock.current = { start: performance.now(), offset: t }; if (audio.current) audio.current.currentTime = t; }} /><span>{error}</span></div>}
    <audio ref={audio} src={audioSource?.url} preload="metadata" />
  </main>;
}
