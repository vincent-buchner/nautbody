<script setup lang="ts">
import { useMicrophone } from "@@/src/core/userMedia/microphone/useMicrophone";
import { useSpeaker } from "@@/src/core/userMedia/speaker/useSpeaker";
import { useConversationConnection } from "../_model/useConversationConnection";

const onOpen = () => {
  speaker.unlock();
  mic.start();
};

const onMessage = (event: MessageEvent<string | ArrayBuffer>) => {
  if (typeof event.data === "string") {
    const { event_type, body } = JSON.parse(event.data);
    if (event_type === "AssistantSpeakingCancelledEvent") speaker.stop();
    return;
  }

  void speaker.play(event.data).catch((error) => {
    console.error("[speaker] playback failed:", error);
  });
};

const onClose = () => {
  void mic.end();
  speaker.stop();
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
  sample_format: "float32",
  onStart: () => console.log("[mic] started"),
  onEnd: () => console.log("[mic] stopped"),
  onRestart: () => console.log("[mic] restarted"),
  // Frames arrive already laid out for the wire (float32 @ 16 kHz, 512
  // samples), so this is a straight handoff to the socket.
  onAudio: (frame: ArrayBufferLike) => {
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
