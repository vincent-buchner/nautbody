import { onScopeDispose, shallowRef } from "vue";

/**
 * `@picovoice/web-voice-processor` reaches for `window.AudioContext` and
 * `navigator.mediaDevices`, so it must never be evaluated on the server. See
 * the guard in loadWebVoiceProcessor().
 */
type WebVoiceProcessorClass =
  (typeof import("@picovoice/web-voice-processor"))["WebVoiceProcessor"];

let cachedWebVoiceProcessor: WebVoiceProcessorClass | null = null;

async function loadWebVoiceProcessor(): Promise<WebVoiceProcessorClass> {
  // `import.meta.client` is statically false in the server build, so Rollup
  // folds this branch and drops the import() below — otherwise Nitro is left
  // holding an external it cannot resolve, since the package is not shipped
  // in .output/server/node_modules. Reaching it server-side is a bug.
  if (!import.meta.client) {
    throw new Error("[useMicrophone] microphone capture is browser-only");
  }

  cachedWebVoiceProcessor ??= (await import("@picovoice/web-voice-processor"))
    .WebVoiceProcessor;
  return cachedWebVoiceProcessor;
}

interface MicrophoneInfo {
  sample_rate: number;
  channel_count: number;
}

interface UseMicrophoneParams {
  sample_rate: number;
  channel_count: number;
  frame_length: number;
  onStart: (info: MicrophoneInfo) => void;
  onEnd: () => void;
  onRestart: (info: MicrophoneInfo) => void;
  onAudio: (frame: ArrayBuffer) => void;
}

const DEFAULT_PARAMS: UseMicrophoneParams = {
  sample_rate: 16_000,
  channel_count: 1,
  frame_length: 512,
  onStart: () => {},
  onEnd: () => {},
  onRestart: () => {},
  onAudio: () => {},
};

/**
 * Microphone capture built on `@picovoice/web-voice-processor`.
 *
 * The library owns the AudioWorklet, the resampling worker and the WASM
 * resampler — it emits Int16Array frames of `frame_length` samples at
 * `sample_rate`. This composable owns the lifecycle and formats those frames
 * for the backend via convertInt16ArrayToFloat32.
 *
 *   const mic = useMicrophone({ onAudio: (bytes) => ws.send(bytes) });
 *   await mic.start();
 *   await mic.end();
 *
 * Note: because `WebVoiceProcessor` is a process-wide singleton with one
 * microphone, this composable is not designed for two instances running
 * concurrently.
 */
const useMicrophone = (params: Partial<UseMicrophoneParams>) => {
  // NOTE: We have to destructure it this way in order to take defaults
  // from the potential undefined properties
  const {
    sample_rate,
    channel_count,
    frame_length,
    onStart,
    onEnd,
    onRestart,
    onAudio,
  } = {
    ...DEFAULT_PARAMS,
    ...params,
  };

  const running = shallowRef(false);

  const engine = {
    onmessage: (event: MessageEvent) => {
      const frame = event.data?.inputFrame;
      if (frame instanceof Int16Array && frame.length > 0) {
        onAudio(convertInt16ArrayToFloat32(frame));
      }
    },
  };

  const doStart = async (notify: boolean): Promise<void> => {
    if (running.value) return;

    // The library hardcodes `numberOfChannels = 1`, so anything else would be
    // silently ignored and emit a frame layout we never asked for.
    if (channel_count !== 1) {
      throw new Error(
        `[useMicrophone] channel_count ${channel_count} is unsupported — ` +
          `@picovoice/web-voice-processor captures mono only`,
      );
    }

    const wvp = await loadWebVoiceProcessor();
    wvp.setOptions({
      outputSampleRate: sample_rate,
      frameLength: frame_length,
    });

    try {
      await wvp.subscribe(engine);
    } catch (err) {
      running.value = false;
      throw err;
    }

    running.value = true;
    if (notify) {
      onStart({ sample_rate, channel_count });
    }
  };

  const doEnd = async (notify: boolean): Promise<void> => {
    if (!running.value) return;
    running.value = false;

    try {
      const wvp = await loadWebVoiceProcessor();
      await wvp.unsubscribe(engine);
    } catch (err) {
      console.warn("[useMicrophone] releasing capture failed:", err);
    }

    if (notify) onEnd();
  };

  const start = () => doStart(true);

  const end = () => doEnd(true);

  /** Tears down and starts again, firing only `onRestart`. */
  const restart = async (): Promise<void> => {
    await doEnd(false);
    await doStart(false);
    onRestart({ sample_rate, channel_count });
  };

  onScopeDispose(() => {
    void doEnd(true);
  });

  return { start, end, restart, running };
};

function convertInt16ArrayToFloat32(frame: Int16Array) {
  return Float32Array.from(frame, (s) => s / 32768).buffer;
}

export { useMicrophone };
export type { MicrophoneInfo, UseMicrophoneParams };
