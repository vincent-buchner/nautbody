import { ref, readonly, onScopeDispose } from "vue";

type WebSocketConnectionStatus = "connected" | "pending" | "disconnected";

interface UseConversationHandlers {
  onOpen: () => Promise<void>;
  // FIXME: Get type from contract
  onMessage: (event: MessageEvent<string | ArrayBuffer>) => void;
  onClose: (event?: CloseEvent) => void;
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

  const _teardown = (event?: CloseEvent) => {
    ws = null;
    connectionStatus.value = "disconnected";
    handlers.onClose(event);
  };

  const connect = () => {
    if (connectionStatus.value !== "disconnected") return;
    try {
      ws = new WebSocket(url);
    } catch (err: unknown) {
      throw new Error(`Failed to connect to voice websocket, ${err}`);
    }
    ws.binaryType = "arraybuffer";
    connectionStatus.value = "pending";

    const socket = ws;
    ws.onopen = async () => {
      try {
        await handlers.onOpen();
      } catch {
        disconnect();
        return;
      }

      if (ws !== socket) return;
      connectionStatus.value = "connected";
    };
    ws.onmessage = (event) => {
      if (ws === socket) handlers.onMessage(event);
    };
    ws.onerror = () => {
      if (ws === socket) handlers.onError();
    };

    socket.onclose = (event) => {
      if (ws !== socket) return;
      _teardown(event);
    };
  };

  const send = (payload: ArrayBuffer) => {
    if (ws === null || ws.readyState !== WebSocket.OPEN) return false;

    ws.send(payload);
    return true;
  };

  const disconnect = () => {
    const socket = ws;
    ws = null;
    if (socket === null) return;

    _teardown();

    if (socket.readyState < WebSocket.CLOSING) {
      socket.close();
    }
  };

  onScopeDispose(disconnect);

  return {
    connectionStatus: readonly(connectionStatus),
    connect,
    send,
    disconnect,
  };
};

export { useConversationConnection };
