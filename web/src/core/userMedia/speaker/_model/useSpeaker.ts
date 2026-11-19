import { onScopeDispose } from "vue";

interface UseSpeakerParams {
  sample_rate: number;
  channels: number;
  bytes_per_sample: number;
}

const DEFAULT_USE_SPEAKER_PARAMS: UseSpeakerParams = {
  sample_rate: 24_000,
  channels: 1,
  bytes_per_sample: Float32Array.BYTES_PER_ELEMENT,
};

/**
 * Plays raw PCM clips received from the conversation socket.
 *
 * `AudioContext` is browser-only and the autoplay policy keeps a context
 * created outside a user gesture suspended, so the context is created lazily
 * by `unlock()` — call it synchronously from the connect click — and reused
 * for every clip. Each `AudioBuffer` carries its own sample rate, so the
 * context can stay at the device default (often 48 kHz) and the browser
 * resamples the 24 kHz clip on playback.
 */
const useSpeaker = (params: Partial<UseSpeakerParams>) => {
  const { sample_rate, channels, bytes_per_sample } = {
    ...DEFAULT_USE_SPEAKER_PARAMS,
    ...params,
  };

  let context: AudioContext | null = null;
  let source: AudioBufferSourceNode | null = null;

  function ensureContext(): AudioContext {
    context ??= new AudioContext();
    return context;
  }

  /**
   * Creates/resumes the context while the caller still has user activation.
   * Must run before the first `await` of the click handler — Safari otherwise
   * leaves playback blocked even though the context exists.
   */
  function unlock(): void {
    const ctx = ensureContext();
    if (ctx.state === "suspended") void ctx.resume();
  }

  /** Stops the current clip, if any. Safe to call when nothing is playing. */
  function stop(): void {
    if (!source) return;
    const node = source;
    source = null;
    node.onended = null;
    try {
      node.stop();
    } catch {
      // stop() throws if the node already ended; there is nothing left to do.
    }
    node.disconnect();
  }

  async function play(bytes: ArrayBuffer): Promise<void> {
    if (bytes.byteLength === 0) return;
    if (bytes.byteLength % bytes_per_sample !== 0) {
      console.warn(
        `[speaker] dropping ${bytes.byteLength}-byte frame: not float32 PCM`,
      );
      return;
    }

    const ctx = ensureContext();
    if (ctx.state === "suspended") await ctx.resume();

    // One clip per assistant response, so replace rather than mix. This also
    // silences a stale response: an interruption cancels generation, but the
    // backend cannot recall bytes it already sent.
    // TODO: Make this more responsive to the interrupt event that comes back
    stop();

    const samples = new Float32Array(bytes);
    const buffer = ctx.createBuffer(channels, samples.length, sample_rate);
    buffer.copyToChannel(samples, 0);

    const node = ctx.createBufferSource();
    node.buffer = buffer;
    node.connect(ctx.destination);
    node.onended = () => {
      if (source === node) source = null;
    };
    source = node;
    node.start();
  }

  onScopeDispose(() => {
    stop();
    void context?.close();
    context = null;
  });

  return { unlock, play, stop };
};

export { useSpeaker };
export type { UseSpeakerParams };
