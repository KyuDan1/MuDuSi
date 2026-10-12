import theme, { agentColor } from "./theme";
import { Component, useEffect, useMemo, useRef, type ReactNode } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import {
  ContactShadows,
  Grid,
  Html,
  Line,
  OrbitControls,
  RoundedBox,
} from "@react-three/drei";
import * as THREE from "three";
import type { Agent, Snapshot } from "./types";

const colorOf = (agent: Agent) => agentColor(agent.id, agent.color);

function SoundWave({
  color,
  offset,
  active,
  running,
  time,
}: {
  color: string;
  offset: number;
  active: boolean;
  running: boolean;
  time?: number;
}) {
  const ref = useRef<THREE.Mesh>(null);
  const phase = useRef(offset);
  useFrame((_, delta) => {
    if (!ref.current) return;
    if (running) phase.current = (phase.current + delta * 0.32) % 1;
    const p = time === undefined ? phase.current : (offset + time * 0.32) % 1;
    ref.current.scale.setScalar(0.75 + p * 5.8);
    (ref.current.material as THREE.MeshBasicMaterial).opacity = active
      ? Math.pow(1 - p, 1.7) * 0.56
      : 0;
  });
  return (
    <mesh
      ref={ref}
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, 0.07 + offset * 0.02, 0]}
    >
      <ringGeometry args={[0.98, 1.008, 96]} />
      <meshBasicMaterial
        color={color}
        transparent
        opacity={0}
        depthWrite={false}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

function Robot({
  agent,
  selected,
  onSelect,
  waves,
  running,
  caption,
  cinematic = false,
  time,
  yaw = 0,
}: {
  agent: Agent;
  selected: boolean;
  onSelect: () => void;
  waves: boolean;
  running: boolean;
  caption?: string;
  cinematic?: boolean;
  time?: number;
  yaw?: number;
}) {
  const robot = useRef<THREE.Group>(null);
  const color = colorOf(agent);
  useFrame(({ clock }) => {
    if (robot.current)
      robot.current.position.y =
        agent.position[1] -
        1.2 +
        (agent.speaking && running
          ? Math.sin((time ?? clock.elapsedTime) * 8) * 0.055
          : 0);
  });
  return (
    <group position={[agent.position[0], 0, agent.position[2]]}>
      {waves &&
        [0, 0.25, 0.5, 0.75].map((offset) => (
          <SoundWave
            key={offset}
            offset={offset}
            color={color}
            active={agent.speaking && !agent.muted}
            running={running}
            time={time}
          />
        ))}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.025, 0]}>
        <circleGeometry args={[0.76, 64]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={selected ? 0.15 : 0.055}
        />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.035, 0]}>
        <ringGeometry args={[0.74, selected ? 0.77 : 0.751, 64]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={selected ? 0.95 : 0.4}
        />
      </mesh>
      <group
        ref={robot}
        rotation={[0, yaw, 0]}
        onClick={(e) => {
          e.stopPropagation();
          onSelect();
        }}
        onPointerOver={() => {
          document.body.style.cursor = "pointer";
        }}
        onPointerOut={() => {
          document.body.style.cursor = "auto";
        }}
      >
        <RoundedBox
          args={[0.64, 0.48, 0.52]}
          radius={0.16}
          smoothness={4}
          position={[0, 0.58, 0]}
          castShadow
        >
          <meshStandardMaterial
            color={color}
            roughness={0.43}
            metalness={0.12}
          />
        </RoundedBox>
        <RoundedBox
          args={[0.89, 0.64, 0.65]}
          radius={0.19}
          smoothness={4}
          position={[0, 1.17, 0]}
          castShadow
        >
          <meshStandardMaterial
            color={color}
            roughness={0.32}
            metalness={0.18}
          />
        </RoundedBox>
        <RoundedBox
          args={[0.68, 0.36, 0.095]}
          radius={0.1}
          smoothness={4}
          position={[0, 1.18, 0.324]}
        >
          <meshStandardMaterial
            color="#131c23"
            roughness={0.21}
            metalness={0.25}
          />
        </RoundedBox>
        {[-0.17, 0.17].map((x) => (
          <mesh
            key={x}
            position={[x, 1.22, 0.379]}
            scale={[1, agent.speaking ? 1.3 : 1, 1]}
          >
            <sphereGeometry args={[0.037, 16, 12]} />
            <meshBasicMaterial color={color} />
          </mesh>
        ))}
        <mesh position={[0, 1.095, 0.382]}>
          <boxGeometry args={[agent.speaking ? 0.15 : 0.09, 0.022, 0.008]} />
          <meshBasicMaterial color={color} />
        </mesh>
        {[-1, 1].map((side) => (
          <group key={side}>
            <mesh
              position={[side * 0.5, 1.19, 0]}
              rotation={[0, 0, Math.PI / 2]}
            >
              <cylinderGeometry args={[0.125, 0.125, 0.105, 24]} />
              <meshStandardMaterial
                color="#5c6971"
                metalness={0.55}
                roughness={0.3}
              />
            </mesh>
            {agent.listening && (
              <mesh
                position={[side * 0.56, 1.19, 0]}
                rotation={[0, Math.PI / 2, 0]}
              >
                <torusGeometry args={[0.116, 0.019, 8, 32]} />
                <meshBasicMaterial color={color} transparent opacity={0.75} />
              </mesh>
            )}
            <RoundedBox
              args={[0.15, 0.33, 0.19]}
              radius={0.07}
              smoothness={3}
              position={[side * 0.43, 0.64, 0]}
              rotation={[0, 0, side * 0.22]}
              castShadow
            >
              <meshStandardMaterial color={color} roughness={0.5} />
            </RoundedBox>
            <RoundedBox
              args={[0.23, 0.14, 0.36]}
              radius={0.07}
              smoothness={3}
              position={[side * 0.19, 0.18, 0.06]}
              castShadow
            >
              <meshStandardMaterial
                color="#67737a"
                roughness={0.45}
                metalness={0.32}
              />
            </RoundedBox>
            <mesh position={[side * 0.19, 0.31, 0]}>
              <cylinderGeometry args={[0.067, 0.067, 0.24, 16]} />
              <meshStandardMaterial color="#56646a" />
            </mesh>
          </group>
        ))}
        <mesh position={[0, 1.61, 0]}>
          <cylinderGeometry args={[0.018, 0.018, 0.25, 12]} />
          <meshStandardMaterial
            color="#a1acb1"
            metalness={0.7}
            roughness={0.25}
          />
        </mesh>
        <mesh position={[0, 1.755, 0]}>
          <sphereGeometry args={[0.061, 16, 16]} />
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={agent.speaking ? 2 : 0.2}
          />
        </mesh>
        <mesh position={[0, 0.61, 0.266]}>
          <circleGeometry args={[0.071, 24]} />
          <meshBasicMaterial color="#25352d" />
        </mesh>
        <mesh position={[0, 0.61, 0.271]}>
          <circleGeometry args={[0.031, 24]} />
          <meshBasicMaterial color={color} />
        </mesh>
      </group>
      {caption && (
        <Html
          zIndexRange={[30, 20]}
          position={[
            agent.id === "persona" ? -1.6 : agent.id === "moshi" ? 1.6 : 0,
            agent.id === "custom" ? 2.9 : 3.6,
            0,
          ]}
          center
          style={{ pointerEvents: "none" }}
        >
          <div
            className="showcase-bubble"
            style={{ "--agent": color } as React.CSSProperties}
          >
            <small>{agent.name.toUpperCase()} · AUDIO ASR</small>
            {caption}
          </div>
        </Html>
      )}
      <Html
        zIndexRange={[10, 0]}
        position={[0, 2.25, 0]}
        center
        distanceFactor={cinematic ? undefined : 16}
        style={{ pointerEvents: "none" }}
      >
        <div
          className={`world-label ${selected ? "selected" : ""}`}
          style={{ "--agent": color } as React.CSSProperties}
        >
          <span className="world-dot" />
          {agent.name}
          <span className="world-state">
            {agent.muted
              ? "MUTED"
              : agent.speaking
                ? "SPEAKING"
                : agent.listening
                  ? "LISTENING"
                  : "IDLE"}
          </span>
        </div>
      </Html>
    </group>
  );
}

function Partitions({ state }: { state: Snapshot }) {
  // The reference environment exposes axis-aligned wall endpoints in the x/z plane.
  const obstacles = state.room.obstacles ?? [];
  return (
    <>
      {obstacles.map((obstacle, i) => {
        const raw = obstacle as Record<string, any>;
        const x = Number(raw.x ?? raw.position?.[0] ?? 0);
        const start = Number(
          raw.z_min ?? raw.z1 ?? raw.start?.[1] ?? raw.z?.[0] ?? -3.8,
        );
        const end = Number(
          raw.z_max ?? raw.z2 ?? raw.end?.[1] ?? raw.z?.[1] ?? -0.8,
        );
        const height = Number(raw.height ?? raw.size?.[1] ?? 2.2);
        const depth = Math.max(
          0.1,
          Number(raw.depth ?? raw.size?.[2] ?? Math.abs(end - start)),
        );
        const z = Number(raw.position?.[2] ?? (start + end) / 2);
        return (
          <group key={i} position={[x, 0, z]}>
            <RoundedBox
              args={[0.12, height, depth]}
              radius={0.04}
              position={[0, height / 2, 0]}
              castShadow
              receiveShadow
            >
              <meshStandardMaterial
                color="#485564"
                roughness={0.65}
                metalness={0.15}
                transparent
                opacity={0.77}
              />
            </RoundedBox>
            <mesh position={[0, 0.08, 0]}>
              <boxGeometry args={[0.32, 0.15, depth + 0.18]} />
              <meshStandardMaterial color="#293640" />
            </mesh>
            <Html
              zIndexRange={[10, 0]}
              position={[0, height + 0.35, 0]}
              center
              distanceFactor={18}
            >
              <span className="partition-label">ACOUSTIC PARTITION</span>
            </Html>
          </group>
        );
      })}
    </>
  );
}

function Scene({
  state,
  selected,
  onSelect,
  waves,
  links,
  cameraKey,
  cinematic = false,
  captions = {},
  faceToFace = false,
}: {
  state: Snapshot;
  selected: string;
  onSelect: (id: string) => void;
  waves: boolean;
  links: boolean;
  cameraKey: number;
  cinematic?: boolean;
  captions?: Record<string, string>;
  faceToFace?: boolean;
}) {
  const { camera, invalidate } = useThree();
  useEffect(() => { if (cinematic) invalidate(); }, [cinematic, state.time, invalidate]);
  useEffect(() => {
    camera.position.set(10, 10.5, 13);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }, [camera, cameraKey]);
  useFrame(() => {
    if (cinematic) {
      const angle = 0.12 + state.time * 0.007;
      camera.position.set(Math.sin(angle) * 11, 6.7, Math.cos(angle) * 11);
      camera.lookAt(0, 1, 0.7);
    }
  });
  const { width, depth } = state.room;
  const uniqueLinks = state.links.filter((link) => link.source < link.target);
  return (
    <>
      <color attach="background" args={[theme.videoBackground]} />
      <fog attach="fog" args={[theme.videoBackground, 22, 43]} />
      <ambientLight intensity={0.85} />
      <hemisphereLight args={["#d6e9ff", "#445341", 1.4]} />
      <directionalLight
        position={[4, 12, 6]}
        intensity={2.8}
        castShadow
        shadow-mapSize={cinematic ? [512, 512] : [2048, 2048]}
        shadow-camera-left={-10}
        shadow-camera-right={10}
        shadow-camera-top={10}
        shadow-camera-bottom={-10}
        shadow-bias={-0.0002}
      />
      <pointLight position={[-5, 5, -4]} intensity={22} color="#aac6ff" />
      <RoundedBox
        args={[width + 0.6, 0.32, depth + 0.6]}
        radius={0.13}
        position={[0, -0.2, 0]}
        receiveShadow
        castShadow
      >
        <meshStandardMaterial
          color="#27332f"
          roughness={0.75}
          metalness={0.08}
        />
      </RoundedBox>
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, -0.032, 0]}
        receiveShadow
      >
        <planeGeometry args={[width, depth]} />
        <meshStandardMaterial color="#202b29" roughness={0.9} />
      </mesh>
      <Grid
        position={[0, -0.016, 0]}
        args={[width, depth]}
        cellSize={1}
        sectionSize={3}
        cellColor="#59665c"
        sectionColor="#778674"
        cellThickness={0.4}
        sectionThickness={0.75}
        fadeDistance={28}
        fadeStrength={1}
        infiniteGrid={false}
      />
      <Line
        points={[
          [-width / 2, 0.005, -depth / 2],
          [width / 2, 0.005, -depth / 2],
          [width / 2, 0.005, depth / 2],
          [-width / 2, 0.005, depth / 2],
          [-width / 2, 0.005, -depth / 2],
        ]}
        color="#829374"
        lineWidth={0.8}
        transparent
        opacity={0.5}
      />
      <mesh position={[0, 0.18, -depth / 2 - 0.12]} castShadow>
        <boxGeometry args={[width + 0.5, 0.44, 0.14]} />
        <meshStandardMaterial color="#344139" />
      </mesh>
      <mesh position={[-width / 2 - 0.12, 0.18, 0]} castShadow>
        <boxGeometry args={[0.14, 0.44, depth + 0.3]} />
        <meshStandardMaterial color="#344139" />
      </mesh>
      <Html
        zIndexRange={[10, 0]}
        position={[-width / 2 - 0.3, 0.03, depth / 2 + 0.55]}
        center
        distanceFactor={20}
      >
        <span className="dimension-label">−6, +4.5</span>
      </Html>
      <Html
        zIndexRange={[10, 0]}
        position={[width / 2 + 0.1, 0.03, depth / 2 + 0.6]}
        center
        distanceFactor={20}
      >
        <span className="dimension-label">{width.toFixed(1)} m</span>
      </Html>
      <Html
        zIndexRange={[10, 0]}
        position={[-width / 2 - 0.65, 0.03, -depth / 2]}
        center
        distanceFactor={20}
      >
        <span className="dimension-label">{depth.toFixed(1)} m</span>
      </Html>
      <Partitions state={state} />
      {links &&
        uniqueLinks.map((link) => {
          const a = state.agents.find((a) => a.id === link.source),
            b = state.agents.find((a) => a.id === link.target);
          if (!a || !b) return null;
          return (
            <group key={link.source + link.target}>
              <Line
                points={[
                  [a.position[0], 0.09, a.position[2]],
                  [b.position[0], 0.09, b.position[2]],
                ]}
                color={link.occluded ? "#e6a366" : "#6c8981"}
                lineWidth={1}
                dashed
                dashSize={0.12}
                gapSize={0.1}
                transparent
                opacity={0.55}
              />
              {!cinematic && (
                <Html
                  zIndexRange={[10, 0]}
                  position={[
                    (a.position[0] + b.position[0]) / 2,
                    0.13,
                    (a.position[2] + b.position[2]) / 2,
                  ]}
                  center
                  distanceFactor={22}
                >
                  <span className="link-label">
                    {link.distance.toFixed(1)} m · {link.delay_ms.toFixed(1)} ms
                    {link.occluded ? " · OCCLUDED" : ""}
                  </span>
                </Html>
              )}
            </group>
          );
        })}
      {state.agents.map((agent) => (
        <Robot
          key={agent.id}
          agent={agent}
          selected={selected === agent.id}
          onSelect={() => onSelect(agent.id)}
          waves={waves}
          running={state.running}
          caption={captions[agent.id]}
          cinematic={cinematic}
          time={cinematic ? state.time : undefined}
          yaw={faceToFace && state.agents.length === 2
            ? Math.atan2(
                state.agents.find((other) => other.id !== agent.id)!.position[0] - agent.position[0],
                state.agents.find((other) => other.id !== agent.id)!.position[2] - agent.position[2],
              )
            : 0}
        />
      ))}
      <ContactShadows
        position={[0, -0.04, 0]}
        opacity={0.35}
        scale={20}
        blur={2.5}
        far={6}
        resolution={256}
        frames={1}
      />
      {!cinematic && (
        <OrbitControls
          key={cameraKey}
          makeDefault
          minDistance={8}
          maxDistance={28}
          minPolarAngle={0.2}
          maxPolarAngle={Math.PI / 2.2}
          target={[0, 0, 0]}
          enableDamping
          dampingFactor={0.07}
        />
      )}
    </>
  );
}

function Fallback({
  state,
  selected,
  onSelect,
}: {
  state: Snapshot;
  selected: string;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="fallback-stage">
      <div className="fallback-grid">
        {state.agents.map((a) => (
          <button
            key={a.id}
            onClick={() => onSelect(a.id)}
            className={`fallback-agent ${selected === a.id ? "active" : ""}`}
            style={
              {
                left: `${50 + (a.position[0] / state.room.width) * 85}%`,
                top: `${50 + (a.position[2] / state.room.depth) * 85}%`,
                "--agent": colorOf(a),
              } as React.CSSProperties
            }
          >
            <span>{a.name.slice(0, 1)}</span>
            <strong>{a.name}</strong>
            <small>
              {a.speaking ? "Speaking" : a.listening ? "Listening" : "Idle"}
            </small>
          </button>
        ))}
      </div>
      <span className="fallback-notice">
        2D compatibility view · WebGL unavailable
      </span>
    </div>
  );
}

class SceneBoundary extends Component<
  { children: ReactNode; fallback: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

export default function Arena(props: {
  state: Snapshot;
  selected: string;
  onSelect: (id: string) => void;
  waves: boolean;
  links: boolean;
  cameraKey: number;
  cinematic?: boolean;
  captions?: Record<string, string>;
  faceToFace?: boolean;
}) {
  const hasGL = useMemo(() => {
    try {
      const canvas = document.createElement("canvas");
      const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
      const supported = !!gl;
      gl?.getExtension("WEBGL_lose_context")?.loseContext();
      return supported;
    } catch {
      return false;
    }
  }, []);
  const fallback = (
    <Fallback
      state={props.state}
      selected={props.selected}
      onSelect={props.onSelect}
    />
  );
  if (!hasGL) return fallback;
  return (
    <SceneBoundary fallback={fallback}>
      <Canvas
        frameloop={props.cinematic ? "demand" : "always"}
        shadows
        camera={{ position: [10, 10.5, 13], fov: 42, near: 0.1, far: 80 }}
        dpr={props.cinematic ? 1 : [1, 1.75]}
        gl={{ antialias: true, alpha: false }}
      >
        <Scene {...props} />
      </Canvas>
    </SceneBoundary>
  );
}
