"""WebSocket client for the LingBot-VA inference server.

Mirrors lingbot-va/evaluation/robotwin/websocket_client_policy.py. Disables ping
so long inference calls don't trip the keepalive."""

import logging
import time
from typing import Dict, Optional, Tuple

import websockets.sync.client

from msgpack_numpy import Packer, unpackb


class PolicyClient:
    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = None,
                 api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=None,
                    close_timeout=10,
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except Exception as e:
                logging.info(f"Still waiting for server... (Error: {e})")
                time.sleep(5)

    def infer(self, obs: Dict) -> Dict:
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)

    def reset(self, prompt: str) -> None:
        self.infer({"reset": True, "prompt": prompt})

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass
