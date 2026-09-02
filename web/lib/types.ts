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
  calibration_path: string | null;
  analysis_mode: "manual_sam" | "auto_all";
  stage: string;
  progress: number;
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
  pitch: [number, number] | null;
  smoothed_pitch?: [number, number] | null;
  area: number;
  speed_kmh: number | null;
};

export type DetectionPoint = {
  frame: number;
  time: number;
  bbox: [number, number, number, number];
};

export type Track = {
  id: string;
  project_id: string;
  object_id: number;
  role: "player" | "goalkeeper" | "referee";
  team: string;
  jersey_number: number | null;
  player_name: string | null;
  dominant_color: [number, number, number] | null;
  detections?: DetectionPoint[];
  trajectory: TrajectoryPoint[];
  speed_series: Array<number | null>;
  mask_path: string | null;
  first_frame: number | null;
  last_frame: number | null;
  detector_confidence: number | null;
  auto_roster_id: number | null;
  roster_id: number | null;
  identity_source: "automatic" | "manual" | "unidentified";
  identity_confidence: number;
  metrics: {
    distance_m: number;
    average_speed_kmh: number | null;
    max_speed_kmh: number | null;
    metric_calibration_available: boolean;
    mask_coverage_ratio?: number;
    occlusion_count: number;
    occlusion_frames: number[];
    area_recovery_ratio: number;
    recovery_frames: number | null;
    max_centroid_jump_px: number;
    id_retained: boolean;
    mask_model_tier?: "base_plus" | "large";
    mask_refinement_status?: "base_ready" | "queued" | "running" | "large_ready" | "failed";
    mask_refinement_error?: string | null;
  };
};

export type RleMask = {
  size: [number, number];
  counts: number[];
  bbox?: [number, number, number, number];
};
export type MaskManifest = {
  fps: number;
  width: number;
  height: number;
  frames: Array<{ index: number; objects: Record<string, RleMask> }>;
};

export type TrackMaskManifest = {
  track_id: number;
  fps: number;
  width: number;
  height: number;
  first_frame: number;
  last_frame: number;
  model_tier?: "base_plus" | "large";
  frames: Array<{ index: number; rle: RleMask }>;
};

export type RosterPlayer = {
  id: number;
  team: string;
  squad_number: number;
  player_name: string;
  position: string;
};
