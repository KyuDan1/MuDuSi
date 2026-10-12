export type Agent = {
  id: string;
  name: string;
  model: string;
  color: string;
  position: [number, number, number];
  speaking: boolean;
  listening: boolean;
  tx_db: number;
  rx_db: number;
  activity: number;
  transcript: string;
  transcript_kind: "scripted" | "model" | "asr";
  asr_transcript?: string;
  worker?: {
    ready?: boolean;
    connected?: boolean;
    error?: string;
    underrun_frames?: number;
    queued_output_samples?: number;
  };
  reward: number;
  cumulative_reward: number;
  muted?: boolean;
};
export type Link = {
  source: string;
  target: string;
  distance: number;
  delay_ms: number;
  gain_db: number;
  occluded: boolean;
  level_db: number;
};
export type ArenaEvent = {
  id: string | number;
  time: number;
  agent_id: string;
  type: string;
  text: string;
  kind?: string;
};
export type Snapshot = {
  time: number;
  tick: number;
  seed: number;
  scenario: string;
  mode: "demo" | "live";
  sample_rate: number;
  frame_ms: number;
  room: {
    width: number;
    depth: number;
    height: number;
    obstacles?: Array<Record<string, unknown>>;
  };
  noise_db: number;
  reverb: number;
  agents: Agent[];
  links: Link[];
  events: ArenaEvent[];
  metrics: {
    overlap_ratio: number;
    mean_rx_db: number;
    episode_reward: number;
    active_speakers: number;
  };
  rewards: Record<
    string,
    { audibility: number; overlap: number; energy: number; total: number }
  >;
  done: boolean;
  running: boolean;
  speed: number;
  error?: string | null;
  asr?: {
    enabled: boolean;
    agent_id?: string;
    ready?: boolean;
    error?: string;
    lag_seconds?: number;
    dropped_windows?: number;
  };
  asr_events?: ArenaEvent[];
};
export type Control = {
  action: "play" | "pause" | "step" | "reset" | "configure";
  speed?: number;
  seed?: number;
  scenario?: string;
  noise_db?: number;
  reverb?: number;
  agents?: Array<{
    id: string;
    position?: [number, number, number];
    muted?: boolean;
  }>;
};
