export type ProjectStatus = "draft" | "queued" | "running" | "completed" | "failed";

export type PromptBox = {
  object_id: number;
  box: [number, number, number, number];
};

export type CalibrationPair = {
  video: [number, number];
  pitch: [number, number];
};

export type Project = {
  id: string;
  owner_id: string;
  title: string;
  match_label: string;
  team_a: string;
  team_b: string;
  source_path: string | null;
  normalized_video_path: string | null;
  mask_manifest_path: string | null;
  foreground_video_path: string | null;
  metrics_path: string | null;
  prompts: PromptBox[];
  calibration: CalibrationPair[];
  slurm_job_id: string | null;
  status: ProjectStatus;
  error_message: string | null;
  created_at: string;
  updated_at: string;
};

export type TrajectoryPoint = {
  frame: number;
  time: number;
  bbox: [number, number, number, number];
  foot: [number, number];
  pitch: [number, number];
  area: number;
  speed_kmh: number;
};

export type Track = {
  id: string;
  project_id: string;
  object_id: number;
  role: "player" | "referee";
  team: string;
  jersey_number: number | null;
  player_name: string | null;
  dominant_color: [number, number, number] | null;
  trajectory: TrajectoryPoint[];
  speed_series: number[];
  metrics: {
    distance_m: number;
    average_speed_kmh: number;
    max_speed_kmh: number;
    ocr_confidence: number;
    occlusion_count: number;
    occlusion_frames: number[];
    area_recovery_ratio: number;
    recovery_frames: number | null;
    max_centroid_jump_px: number;
    id_retained: boolean;
  };
};

export type RleMask = { size: [number, number]; counts: number[] };
export type MaskManifest = {
  fps: number;
  width: number;
  height: number;
  frames: Array<{ index: number; objects: Record<string, RleMask> }>;
};
