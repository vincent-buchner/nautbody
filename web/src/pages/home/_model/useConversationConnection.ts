import { ref, onScopeDispose } from "vue";

type WebSocketConnectionStatus = "connected" | "pending" | "disconnected";

interface UseConversationHandlers {
  onOpen: () => void;
  // FIXME: Get type from contract
  onMessage: (event: MessageEvent<string | ArrayBuffer>) => void;
  onClose: () => void;
  onError: () => void;
}

interface UseConversationConnectionParams {
  handlers: UseConversationHandlers;
}

const useConversationConnection = ({
  handlers,
}: UseConversationConnectionParams) => {
  // FIXME: Pull from contract
  const url = "http://127.0.0.1:8000/conversation/connect";

  let ws: WebSocket | null = null;
  const connectionStatus = ref<WebSocketConnectionStatus>("disconnected");

  const connect = () => {
    if (connectionStatus.value === "connected") return;
    try {
      ws = new WebSocket(url);
    } catch (err: unknown) {
      throw new Error(`Failed to connect to voice websocket, ${err}`);
    }
    ws.binaryType = "arraybuffer";
    connectionStatus.value = "pending";

    ws.onopen = () => {
      connectionStatus.value = "connected";
      handlers.onOpen();
    };
    ws.onmessage = handlers.onMessage;
    ws.onerror = handlers.onError;
    ws.onclose = () => {
      try {
        handlers.onClose();
      } finally {
        disconnect();
      }
    };
  };

  const send = (payload: string | ArrayBufferLike) => {
    if (ws === null || ws.readyState !== WebSocket.OPEN) return false;

    ws.send(payload);
    return true;
  };

  const disconnect = () => {
    const socket = ws;
    ws = null;
    if (socket === null) return;
    connectionStatus.value = "disconnected";

    // NOTE: Ugly line of code, but just reset all the handlers
    socket.onopen = socket.onmessage = socket.onclose = socket.onerror = null;

    if (socket.readyState < WebSocket.CLOSING) {
      socket.close();
    }
  };

  onScopeDispose(disconnect);

  return {
    connectionStatus,
    connect,
    send,
    disconnect,
  };
};

export { useConversationConnection };
