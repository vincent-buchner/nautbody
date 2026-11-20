<script setup lang="ts">
import { useMicrophone } from "@@/src/core/userMedia";
import { useSpeaker } from "@@/src/core/userMedia";
import { useConversationConnection } from "../_model/useConversationConnection";

// mic.end() is a no-op while start() is in flight (running flips true only
// after subscribe()), so onClose awaits this first — otherwise disconnecting
// during the permission prompt leaks the mic.
let startPromise: Promise<void> = Promise.resolve();

const onOpen = async () => {
  speaker.unlock();
  startPromise = mic.start();
  await startPromise;
};

const onMessage = (event: MessageEvent<string | ArrayBuffer>) => {
  if (typeof event.data === "string") {
    const { event_type } = JSON.parse(event.data);
    if (event_type === "AssistantSpeakingCancelledEvent") speaker.stop();
    return;
  }

  void speaker.play(event.data).catch((error) => {
    console.error("[speaker] playback failed:", error);
  });
};

const onClose = async (event?: CloseEvent) => {
  await startPromise.catch(() => {});
  void mic.end();
  speaker.stop();

  if (event !== undefined) console.warn(`Closed WS: ${event.reason}`);
};

const onError = () => {
  void mic.end();
  speaker.stop();
};

const { connectionStatus, connect, disconnect, send } =
  useConversationConnection({
    handlers: {
      onOpen,
      onMessage,
      onClose,
      onError,
    },
  });

const mic = useMicrophone({
  sample_rate: 16_000,
  channel_count: 1,
  onStart: () => console.log("[mic] started"),
  onEnd: () => console.log("[mic] stopped"),
  onRestart: () => console.log("[mic] restarted"),
  // Frames arrive already laid out for the wire (float32 @ 16 kHz, 512
  // samples), so this is a straight handoff to the socket.
  onAudio: (frame: ArrayBuffer) => {
    send(frame);
  },
});

// TTS clips arrive as one binary frame per response: float32 LE @ 24 kHz mono.
const speaker = useSpeaker({ sample_rate: 24_000 });

const canConnect = computed(() => connectionStatus.value === "disconnected");
const canDisconnect = computed(() => connectionStatus.value !== "disconnected");
</script>

<template>
  <div>
    <button @click="connect()" :disabled="!canConnect">
      Connect to Web Socket
    </button>
  </div>
  <div>
    <button @click="disconnect()" :disabled="!canDisconnect">
      Disconnect from Web Socket
    </button>
  </div>
</template>
