import logging
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AnyStr, Callable, Dict, Optional, Text, Union

from websocket_server import WebsocketServer

from . import blocks
from . import sources as src
from .inference import StreamingInference

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ClientState:
    """Represents the state of a connected client."""

    audio_source: src.WebSocketAudioSource
    inference: StreamingInference


class WebSocketStreamingServer:
    """Handles real-time speaker diarization inference for multiple audio sources over WebSocket.

    This handler manages WebSocket connections from multiple clients, processing
    audio streams and performing speaker diarization in real-time.

    Parameters
    ----------
    pipeline_class : type
        Pipeline class
    pipeline_config : blocks.PipelineConfig
        Pipeline configuration
    host : str, optional
        WebSocket server host, by default "127.0.0.1"
    port : int, optional
        WebSocket server port, by default 7007
    key : Union[str, Path], optional
        SSL key file path for secure WebSocket
    certificate : Union[str, Path], optional
        SSL certificate file path for secure WebSocket
    """

    def __init__(
        self,
        pipeline_class: type,
        pipeline_config: blocks.PipelineConfig,
        host: Text = "127.0.0.1",
        port: int = 7007,
        key: Optional[Union[Text, Path]] = None,
        certificate: Optional[Union[Text, Path]] = None,
    ):
        # Pipeline configuration
        self.pipeline_class = pipeline_class
        self.pipeline_config = pipeline_config

        # Server configuration
        self.host = host
        self.port = port
        self.uri = f"{host}:{port}"
        self._clients: Dict[Text, ClientState] = {}

        # Initialize WebSocket server
        self.server = WebsocketServer(host, port, key=key, cert=certificate)
        self._setup_server_handlers()

    def _setup_server_handlers(self) -> None:
        """Configure WebSocket server event handlers."""
        self.server.set_fn_new_client(self._on_connect)
        self.server.set_fn_client_left(self._on_disconnect)
        self.server.set_fn_message_received(self._on_message_received)

    def _create_client_state(self, client_id: Text) -> ClientState:
        """Create and initialize state for a new client.

        Parameters
        ----------
        client_id : Text
            Unique client identifier

        Returns
        -------
        ClientState
            Initialized client state object
        """
        # Create a new pipeline instance with the same config
        # This ensures each client has its own state while sharing model weights
        pipeline = self.pipeline_class(self.pipeline_config)

        audio_source = src.WebSocketAudioSource(
            uri=f"{self.uri}:{client_id}",
            sample_rate=self.pipeline_config.sample_rate,
        )

        inference = StreamingInference(
            pipeline=pipeline,
            source=audio_source,
            # The following variables are fixed for a client
            batch_size=1,
            do_profile=False, # for minimal latency
            do_plot=False,
            show_progress=False,
            progress_bar=None,
        )

        return ClientState(audio_source=audio_source, inference=inference)

    def _on_connect(self, client: Dict[Text, Any], server: WebsocketServer) -> None:
        """Handle new client connection.

        Parameters
        ----------
        client : Dict[Text, Any]
            Client information dictionary
        server : WebsocketServer
            WebSocket server instance
        """
        client_id = client["id"]
        logger.info(f"New client connected: {client_id}")

        if client_id in self._clients:
            return

        try:
            self._clients[client_id] = self._create_client_state(client_id)

            # Setup RTTM response hook
            self._clients[client_id].inference.attach_hooks(
                lambda ann_wav: self.send(client_id, ann_wav[0].to_rttm())
            )

            # Start inference
            self._clients[client_id].inference()
            logger.info(f"Started inference for client: {client_id}")

            # Send ready notification to client
            self.send(client_id, "READY")
        except Exception as e:
            logger.error(f"Failed to initialize client {client_id}: {e}")
            self.close(client_id)

    def _on_disconnect(self, client: Dict[Text, Any], server: WebsocketServer) -> None:
        """Handle client disconnection.

        Parameters
        ----------
        client : Dict[Text, Any]
            Client information dictionary
        server : WebsocketServer
            WebSocket server instance
        """
        client_id = client["id"]
        logger.info(f"Client disconnected: {client_id}")
        self.close(client_id)

    def _on_message_received(
        self, client: Dict[Text, Any], server: WebsocketServer, message: AnyStr
    ) -> None:
        """Process incoming client messages.

        Parameters
        ----------
        client : Dict[Text, Any]
            Client information dictionary
        server : WebsocketServer
            WebSocket server instance
        message : AnyStr
            Received message data
        """
        client_id = client["id"]

        if client_id not in self._clients:
            return

        try:
            self._clients[client_id].audio_source.process_message(message)
        except (socket.error, ConnectionError) as e:
            logger.warning(f"Client {client_id} disconnected: {e}")
            self.close(client_id)
        except Exception as e:
            logger.error(f"Error processing message from client {client_id}: {e}")
            # Don't close the connection for non-connection related errors
            # This allows the client to retry sending the message

    def send(self, client_id: Text, message: AnyStr) -> None:
        """Send a message to a specific client.

        Parameters
        ----------
        client_id : Text
            Target client identifier
        message : AnyStr
            Message to send
        """
        if not message:
            return

        client = next((c for c in self.server.clients if c["id"] == client_id), None)

        if client is None:
            return

        try:
            self.server.send_message(client, message)
        except (socket.error, ConnectionError) as e:
            logger.warning(
                f"Client {client_id} disconnected while sending message: {e}"
            )
            self.close(client_id)
        except Exception as e:
            logger.error(f"Failed to send message to client {client_id}: {e}")

    def run(self) -> None:
        """Start the WebSocket server."""
        logger.info(f"Starting WebSocket server on {self.uri}")
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                self.server.run_forever()
                break  # If server exits normally, break the retry loop
            except (socket.error, ConnectionError) as e:
                logger.warning(f"WebSocket connection error: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(
                        f"Attempting to restart server (attempt {retry_count + 1}/{max_retries})"
                    )
                else:
                    logger.error("Max retry attempts reached. Server shutting down.")
            except Exception as e:
                logger.error(f"Fatal server error: {e}")
                break
            finally:
                self.close_all()

    def close(self, client_id: Text) -> None:
        """Close a specific client's connection and cleanup resources.

        Parameters
        ----------
        client_id : Text
            Client identifier to close
        """
        if client_id not in self._clients:
            return

        try:
            # Clean up pipeline state using built-in reset method
            client_state = self._clients[client_id]
            client_state.inference.pipeline.reset()

            # Close audio source and remove client
            client_state.audio_source.close()
            del self._clients[client_id]

            # Try to send a close frame to the client
            client = next((c for c in self.server.clients if c["id"] == client_id), None)

            if client is None:
                return

            try:
                self.server.send_message(client, "CLOSE")
            except (socket.error, ConnectionError) as e:
                logger.warning(
                    f"Client {client_id} disconnected while sending message: {e}"
                )
                self.close(client_id)
            except Exception as e:
                logger.error(f"Failed to send message to client {client_id}: {e}")

            logger.info(
                f"Closed connection and cleaned up state for client: {client_id}"
            )
        except Exception as e:
            logger.error(f"Error closing client {client_id}: {e}")
            # Ensure client is removed from dictionary even if cleanup fails
            self._clients.pop(client_id, None)

    def close_all(self) -> None:
        """Shutdown the server and cleanup all client connections."""
        logger.info("Shutting down server...")
        try:
            for client_id in self._clients.keys():
                self.close(client_id)
            if self.server is not None:
                self.server.shutdown_gracefully()
            logger.info("Server shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
