"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { decodeRleCrop, nearestFrameValue, projectMaskCrop, smallestTrackAt } from "@/lib/masks";
import type { Track, TrackMaskManifest } from "@/lib/types";

const COLORS = [[184, 255, 98], [117, 216, 255], [255, 189, 102], [206, 134, 255], [255, 123, 115], [80, 227, 178]];
type BoxPoint = { frame: number; bbox: [number, number, number, number] };

export function MaskVideoPlayer({
  videoUrl,
  manifest,
  tracks,
  selectedId,
  showMasks,
  showTrajectory,
  onSelect,
  onTime,
}: {
  videoUrl: string;
  manifest: TrackMaskManifest | null;
  tracks: Track[];
  selectedId: number | null;
  showMasks: boolean;
  showTrajectory: boolean;
  onSelect: (objectId: number) => void;
  onTime: (time: number) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [videoDimensions, setVideoDimensions] = useState({ width: 1280, height: 720 });
  const dimensions = manifest ? { width: manifest.width, height: manifest.height } : videoDimensions;
  const [frameIndex, setFrameIndex] = useState(0);
  const fps = manifest?.fps ?? 15;
  const trackMap = useMemo(() => new Map(tracks.map((track) => [track.object_id, track])), [tracks]);
  const maskByFrame = useMemo(() => new Map(manifest?.frames.map((frame) => [frame.index, frame.rle]) ?? []), [manifest]);
  const detectionsByFrame = useMemo(() => {
    const indexed = new Map<number, Array<{ track: Track; point: BoxPoint }>>();
    for (const track of tracks) {
      for (const point of track.detections ?? track.trajectory) {
        const items = indexed.get(point.frame) ?? [];
        items.push({ track, point });
        indexed.set(point.frame, items);
      }
    }
    return indexed;
  }, [tracks]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    const maskFrame = selectedId !== null ? nearestFrameValue(maskByFrame, frameIndex) : null;
    if (showMasks && maskFrame) {
      const mask = decodeRleCrop(maskFrame.value);
      const image = context.createImageData(mask.width, mask.height);
      const color = COLORS[Math.abs((selectedId ?? 1) - 1) % COLORS.length];
      for (let pixel = 0; pixel < mask.pixels.length; pixel += 1) {
        if (!mask.pixels[pixel]) continue;
        const offset = pixel * 4;
        image.data[offset] = color[0];
        image.data[offset + 1] = color[1];
        image.data[offset + 2] = color[2];
        image.data[offset + 3] = 120;
      }
      const offscreen = document.createElement("canvas");
      offscreen.width = mask.width;
      offscreen.height = mask.height;
      offscreen.getContext("2d")?.putImageData(image, 0, 0);
      const selectedTrack = selectedId === null ? undefined : trackMap.get(selectedId);
      const sourceBox = (selectedTrack?.detections ?? selectedTrack?.trajectory)?.find((point) => point.frame === maskFrame.frame)?.bbox;
      const targetBox = (selectedTrack?.detections ?? selectedTrack?.trajectory)?.find((point) => point.frame === frameIndex)?.bbox;
      const projected = projectMaskCrop(mask, sourceBox, targetBox);
      context.drawImage(offscreen, projected.x, projected.y, projected.width, projected.height);
    }

    if (showTrajectory) {
      const trajectoryTracks = selectedId === null ? tracks : [trackMap.get(selectedId)].filter((track): track is Track => Boolean(track));
      for (const track of trajectoryTracks) {
        const selected = track.object_id === selectedId;
        context.strokeStyle = selected ? "#b8ff62" : "rgba(184,255,98,.22)";
        context.lineWidth = selected ? 3 : 1;
        context.lineCap = "round";
        context.lineJoin = "round";
        let path: Array<{ frame: number; foot: [number, number] }> = [];
        let previousFrame: number | null = null;
        const drawPath = () => {
          if (path.length < 2) return;
          context.beginPath();
          path.forEach((point, index) => index ? context.lineTo(...point.foot) : context.moveTo(...point.foot));
          context.stroke();
        };
        for (const point of track.trajectory.filter((item) => item.frame <= frameIndex)) {
          if (previousFrame !== null && point.frame - previousFrame > 15) { drawPath(); path = []; }
          path.push({ frame: point.frame, foot: point.foot });
          previousFrame = point.frame;
        }
        drawPath();
      }
    }

    for (const { track, point } of detectionsByFrame.get(frameIndex) ?? []) {
      const color = COLORS[Math.abs(track.object_id - 1) % COLORS.length];
      context.strokeStyle = selectedId === track.object_id ? "#b8ff62" : `rgba(${color.join(",")},.72)`;
      context.lineWidth = selectedId === track.object_id ? 3 : 1.25;
      context.strokeRect(point.bbox[0], point.bbox[1], point.bbox[2] - point.bbox[0], point.bbox[3] - point.bbox[1]);
      const labelY = Math.max(0, point.bbox[1] - 20);
      context.font = "700 13px sans-serif";
      context.fillStyle = "rgba(5,12,10,.86)";
      context.fillRect(point.bbox[0], labelY, 46, 18);
      context.fillStyle = `rgb(${color.join(",")})`;
      context.fillText(`ID ${track.object_id}`, point.bbox[0] + 5, labelY + 14);
    }
  }, [detectionsByFrame, frameIndex, maskByFrame, selectedId, showMasks, showTrajectory, trackMap, tracks]);

  useEffect(() => draw(), [draw]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !("requestVideoFrameCallback" in video)) return;
    let callbackId = 0;
    const updateFrame: VideoFrameRequestCallback = (_now, metadata) => {
      setFrameIndex(Math.round(metadata.mediaTime * fps));
      onTime(metadata.mediaTime);
      callbackId = video.requestVideoFrameCallback(updateFrame);
    };
    callbackId = video.requestVideoFrameCallback(updateFrame);
    return () => video.cancelVideoFrameCallback(callbackId);
  }, [fps, onTime, videoUrl]);

  function syncTime() {
    const time = videoRef.current?.currentTime ?? 0;
    setFrameIndex(Math.round(time * fps));
    onTime(time);
  }

  function readVideoSize() {
    const video = videoRef.current;
    if (video?.videoWidth && video.videoHeight && !manifest) {
      setVideoDimensions({ width: video.videoWidth, height: video.videoHeight });
    }
  }

  function click(event: React.PointerEvent<HTMLDivElement>) {
    const rect = event.currentTarget.parentElement?.getBoundingClientRect() ?? event.currentTarget.getBoundingClientRect();
    // Native controls occupy the lower edge of the video. Ignore only that
    // strip; the rest of the stage remains a reliable track hit target even
    // when the browser reports the parent/canvas as the event target.
    if ((event.clientY - rect.top) / rect.height > 0.86) return;
    const x = ((event.clientX - rect.left) / rect.width) * dimensions.width;
    const y = ((event.clientY - rect.top) / rect.height) * dimensions.height;
    const selected = smallestTrackAt(tracks, frameIndex, x, y);
    if (selected) {
      event.preventDefault();
      videoRef.current?.pause();
      onSelect(selected.object_id);
    }
  }

  return (
    <div className="video-stage">
      <video ref={videoRef} src={videoUrl} crossOrigin="anonymous" controls playsInline onLoadedMetadata={readVideoSize} onLoadedData={syncTime} onTimeUpdate={syncTime} onSeeked={syncTime} onEnded={syncTime} />
      <canvas ref={canvasRef} width={dimensions.width} height={dimensions.height} />
      <div className="video-hit-area" onPointerDown={click} aria-hidden="true" />
    </div>
  );
}
