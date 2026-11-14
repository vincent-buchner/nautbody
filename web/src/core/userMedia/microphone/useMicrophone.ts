import { onScopeDispose, shallowRef } from "vue";

import { toWire } from "./toWire";
import type { SampleFormat } from "./toWire";

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
    throw new Error(
      "[useMicrophone] microphone capture is browser-only",
    );
  }

  cachedWebVoiceProcessor ??= (await import("@picovoice/web-voice-processor"))
    .WebVoiceProcessor;
  return cachedWebVoiceProcessor;
}

/** What the lifecycle callbacks receive. */
interface MicrophoneInfo {
  sample_rate: number;
  channel_count: number;
  sample_format: SampleFormat;
}

interface UseMicrophoneParams {
  /** Output sample rate in Hz. Must match the backend's SAMPLE_RATE. */
  sample_rate: number;
  /** Channels per frame. The library captures mono, so only 1 is valid. */
  channel_count: number;
  /** Byte layout of each emitted frame. */
  sample_format: SampleFormat;
  onStart: (info: MicrophoneInfo) => void;
  onEnd: () => void;
  onRestart: (info: MicrophoneInfo) => void;
  /** One wire-ready frame per VAD window. */
  onAudio: (frame: ArrayBufferLike) => void;
}

const DEFAULT_PARAMS: UseMicrophoneParams = {
  sample_rate: 16_000,
  channel_count: 1,
  sample_format: "float32",
  onStart: () => {},
  onEnd: () => {},
  onRestart: () => {},
  onAudio: () => {},
};

/**
 * Silero's analysis window, not a free parameter: the backend reinterprets
 * each incoming chunk under VAD_CHUNK_SIZE = 512 (gateway/cli/main.py:20),
 * so a different frame size would feed the VAD sliced audio.
 */
const FRAME_LENGTH = 512;

/**
 * Microphone capture built on `@picovoice/web-voice-processor`.
 *
 * The library owns the AudioWorklet, the resampling worker and the WASM
 * resampler — it emits Int16Array frames of `frame_length` samples at
 * `sample_rate`. This composable owns the lifecycle and formats those frames
 * for the wire via `toWire`.
 *
 *   const mic = useMicrophone({ onAudio: (bytes) => ws.send(bytes) });
 *   await mic.start();
 *   await mic.end();
 *
 * Note: because `WebVoiceProcessor` is a process-wide singleton with one
 * microphone, this composable is not designed for two instances running
 * concurrently.
 */
const useMicrophone = (params: Partial<UseMicrophoneParams> = {}) => {
  const {
    sample_rate,
    channel_count,
    sample_format,
    onStart,
    onEnd,
    onRestart,
    onAudio,
  } = { ...DEFAULT_PARAMS, ...params };

  const running = shallowRef(false);

  const engine = {
    onmessage: (event: MessageEvent) => {
      const frame = event.data?.inputFrame;
      if (frame instanceof Int16Array && frame.length > 0) {
        onAudio(toWire(frame, sample_format));
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

    // Must run before subscribe(): without forceUpdate, setOptions only takes
    // effect on engines subscribed afterwards.
    const wvp = await loadWebVoiceProcessor();
    wvp.setOptions({
      outputSampleRate: sample_rate,
      frameLength: FRAME_LENGTH,
    });

    try {
      await wvp.subscribe(engine);
    } catch (err) {
      running.value = false;
      // The library maps getUserMedia failures to named WvpErrors
      // (PermissionError, DeviceMissingError, DeviceReadError).
      throw err;
    }

    running.value = true;
    if (notify) {
      onStart({ sample_rate, channel_count, sample_format });
    }
  };

  const doEnd = async (notify: boolean): Promise<void> => {
    if (!running.value) return;
    running.value = false;

    try {
      // doStart() already cached the module, so this cannot re-import or fail.
      // The engine leaves the set synchronously on entry, so frames stop
      // arriving immediately even though teardown continues asynchronously.
      const wvp = await loadWebVoiceProcessor();
      await wvp.unsubscribe(engine);
    } catch (err) {
      console.warn("[useMicrophone] releasing capture failed:", err);
    }

    if (notify) onEnd();
  };

  const start = (): Promise<void> => doStart(true);

  const end = (): Promise<void> => doEnd(true);

  /** Tears down and starts again, firing only `onRestart`. */
  const restart = async (): Promise<void> => {
    await doEnd(false);
    await doStart(false);
    onRestart({ sample_rate, channel_count, sample_format });
  };

  onScopeDispose(() => {
    void doEnd(true);
  });

  return { start, end, restart, running };
};

export { useMicrophone };
export type { MicrophoneInfo, SampleFormat, UseMicrophoneParams };
