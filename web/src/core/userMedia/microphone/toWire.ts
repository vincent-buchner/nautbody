type SampleFormat = "int16" | "float32";

/**
 * -32768 maps to exactly -1.0; +32767 lands just under +1.0, which is the
 * correct asymmetry for two's-complement int16.
 */
const INT16_FULL_SCALE = 32768;

/**
 * Turn one resampled frame into the byte layout the peer expects.
 *
 * `@picovoice/web-voice-processor` only ever emits Int16Array, while the
 * backend reads `np.frombuffer(chunk, dtype=np.float32)`
 * (start_conversation.py:98) — so the default path scales back into [-1, 1).
 * Sending the raw int16 bytes would be reinterpreted as garbage floats that
 * still satisfy `len % 4 == 0`, i.e. it would fail silently.
 *
 *   "float32" -> 4 bytes per sample (2048 for a 512-sample frame)
 *   "int16"   -> 2 bytes per sample (1024 for a 512-sample frame)
 *
 * The `slice()` on the int16 path is load-bearing: structured clone hands
 * over a typed array's entire backing buffer, so a frame that is a view into
 * a larger buffer would otherwise be sent wholesale along with its
 * neighbours.
 */
function toWire(frame: Int16Array, format: SampleFormat): ArrayBufferLike {
  if (format === "int16") return frame.slice().buffer;

  const out = new Float32Array(frame.length);
  for (let i = 0; i < frame.length; i++) {
    out[i] = frame[i] / INT16_FULL_SCALE;
  }
  return out.buffer;
}

export { toWire };
export type { SampleFormat };
