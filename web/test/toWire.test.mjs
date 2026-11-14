import assert from "node:assert/strict";
import { test } from "node:test";

import { toWire } from "../src/core/userMedia/microphone/toWire.ts";

const FRAME_LENGTH = 512;

test("float32 frames match the backend's 512-sample contract", () => {
  const bytes = toWire(new Int16Array(FRAME_LENGTH), "float32");

  assert.equal(
    bytes.byteLength,
    FRAME_LENGTH * 4,
    "512 samples x 4 bytes = 2048, the size start_conversation.py reinterprets",
  );
  assert.equal(new Float32Array(bytes).length, FRAME_LENGTH);
  assert.ok(
    new Float32Array(bytes).every((v) => v === 0),
    "digital silence stays silence",
  );
});

test("int16 full scale maps onto [-1, 1] without wrapping", () => {
  const frame = new Int16Array([0, 32767, -32768, 1, -1]);
  const out = new Float32Array(toWire(frame, "float32"));

  assert.equal(out[0], 0);
  assert.equal(out[2], -1, "-32768 is exactly -1.0");
  assert.ok(Math.abs(out[1] - 32767 / 32768) < 1e-7, "+32767 lands just under 1.0");
  assert.ok(Math.abs(out[3] - 1 / 32768) < 1e-7);
  assert.ok(Math.abs(out[4] + 1 / 32768) < 1e-7);

  for (const v of out) {
    assert.ok(v >= -1 && v <= 1, `${v} escaped [-1, 1]`);
  }
});

test("int16 output is a byte-exact pass-through at half the size", () => {
  const frame = new Int16Array(FRAME_LENGTH);
  for (let i = 0; i < frame.length; i++) frame[i] = i - 256;

  const bytes = toWire(frame, "int16");

  assert.equal(bytes.byteLength, FRAME_LENGTH * 2, "512 samples x 2 bytes");
  assert.deepEqual(new Int16Array(bytes), frame, "no rescaling, no truncation");
});

test("an int16 view into a larger buffer is not sent wholesale", () => {
  // Structured clone hands over a typed array's entire backing buffer, so a
  // frame that is a subarray must be copied out — otherwise the neighbouring
  // samples ride along and the VAD reads someone else's audio.
  const backing = new Int16Array(FRAME_LENGTH + 64);
  backing.fill(999, 0, 32); // junk ahead of the view

  const frame = backing.subarray(32, 32 + FRAME_LENGTH);
  for (let i = 0; i < frame.length; i++) frame[i] = i;

  const bytes = toWire(frame, "int16");

  assert.equal(bytes.byteLength, FRAME_LENGTH * 2, "only the view, not the store");
  const out = new Int16Array(bytes);
  assert.equal(out[0], 0);
  assert.equal(out[FRAME_LENGTH - 1], FRAME_LENGTH - 1);
  assert.ok(!out.includes(999), "junk outside the view leaked into the frame");
});

test("an unrecognised format falls through to the float32 path", () => {
  const bytes = toWire(new Int16Array(4), "banana");

  assert.equal(bytes.byteLength, 4 * 4, "defaults to the backend's layout");
});
