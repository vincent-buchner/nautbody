import { onScopeDispose } from "vue";

/**
 * Backend TTS wire contract: `ChatterboxTTS.generate_audio` returns
 * `torch.from_numpy(wav).unsqueeze(0).numpy().tobytes()` — mono float32 LE at
 * S3GEN_SR = 24_000 Hz — and the gateway sends that whole buffer as a single
 * binary WebSocket frame (service.py: handle_assistant_speaking_stopped).
 */
const DEFAULT_SAMPLE_RATE = 24_000;
const CHANNELS = 1;
const BYTES_PER_SAMPLE = Float32Array.BYTES_PER_ELEMENT;

interface UseSpeakerParams {
  /** Sample rate of the incoming PCM. Must match the backend's TTS output. */
  sample_rate?: number;
}

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
const useSpeaker = (params: UseSpeakerParams = {}) => {
  const sampleRate = params.sample_rate ?? DEFAULT_SAMPLE_RATE;

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
    if (bytes.byteLength % BYTES_PER_SAMPLE !== 0) {
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
    stop();

    const samples = new Float32Array(bytes);
    const buffer = ctx.createBuffer(CHANNELS, samples.length, sampleRate);
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
