import * as tus from "tus-js-client";
import type { SupabaseClient } from "@supabase/supabase-js";
import { supabaseConfig } from "./supabase/config";

export const MAX_VIDEO_BYTES = 50 * 1024 * 1024;

export async function uploadVideo(
  supabase: SupabaseClient,
  file: File,
  objectPath: string,
  onProgress: (percent: number) => void,
) {
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  if (!token) throw new Error("The session has expired.");
  const { url, key } = supabaseConfig();

  await new Promise<void>((resolve, reject) => {
    const upload = new tus.Upload(file, {
      endpoint: `${url}/storage/v1/upload/resumable`,
      headers: { authorization: `Bearer ${token}`, apikey: key },
      metadata: {
        bucketName: "videos",
        objectName: objectPath,
        contentType: "video/mp4",
        cacheControl: "3600",
      },
      chunkSize: 6 * 1024 * 1024,
      removeFingerprintOnSuccess: true,
      onProgress: (sent, total) => onProgress(Math.round((sent / total) * 100)),
      onError: reject,
      onSuccess: () => resolve(),
    });
    upload.start();
  });
}
