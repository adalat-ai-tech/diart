from dataclasses import dataclass
from pathlib import Path
from typing import Union, Text, Optional, AnyStr, Dict, Any, Callable
import logging
from websocket_server import WebsocketServer

from . import blocks
from . import sources as src
from .inference import StreamingInference
from .progress import ProgressBar, RichProgressBar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClientState:
    """Represents the state of a connected client."""
    audio_source: src.WebSocketAudioSource
    inference: StreamingInference


class StreamingInferenceHandler:
    """Handles real-time speaker diarization inference for multiple audio sources over WebSocket.

    This handler manages WebSocket connections from multiple clients, processing
    audio streams and performing speaker diarization in real-time.

    Parameters
    ----------
    pipeline : blocks.Pipeline
        Configured speaker diarization pipeline
    sample_rate : int, optional
        Audio sample rate in Hz, by default 16000
    host : str, optional
        WebSocket server host, by default "127.0.0.1"
    port : int, optional
        WebSocket server port, by default 7007
    key : Union[str, Path], optional
        SSL key file path for secure WebSocket
    certificate : Union[str, Path], optional
        SSL certificate file path for secure WebSocket
    batch_size : int, optional
        Number of inputs to process at once, by default 1
    do_profile : bool, optional
        Enable processing time profiling, by default True
    do_plot : bool, optional
        Enable real-time prediction plotting, by default False
    show_progress : bool, optional
        Display progress bar, by default True
    progress_bar : ProgressBar, optional
        Custom progress bar implementation
    output : bool, optional
        Enable output saving, by default False
    """

    def __init__(
        self,
        pipeline: blocks.Pipeline,
        sample_rate: int = 16000,
        host: Text = "127.0.0.1",
        port: int = 7007,
        key: Optional[Union[Text, Path]] = None,
        certificate: Optional[Union[Text, Path]] = None,
        batch_size: int = 1,
        do_profile: bool = True,
        do_plot: bool = False,
        show_progress: bool = True,
        progress_bar: Optional[ProgressBar] = None,
        output: bool = False
    ):
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.do_profile = do_profile
        self.do_plot = do_plot
        self.show_progress = show_progress
        self.progress_bar = progress_bar or RichProgressBar()
        self.output = output

        # Server configuration
        self.uri = f"{host}:{port}"
        self.sample_rate = sample_rate
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
        audio_source = src.WebSocketAudioSource(
            uri=f"{self.uri}:{client_id}",
            sample_rate=self.sample_rate
        )

        inference = StreamingInference(
            pipeline=self.pipeline,
            source=audio_source,
            batch_size=self.batch_size,
            do_profile=self.do_profile,
            do_plot=self.do_plot,
            show_progress=self.show_progress,
            progress_bar=self.progress_bar
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

        if client_id not in self._clients:
            try:
                client_state = self._create_client_state(client_id)
                self._clients[client_id] = client_state

                # Setup RTTM response hook
                client_state.inference.attach_hooks(
                    lambda ann_wav: self.send(client_id, ann_wav[0].to_rttm())
                )

                # Start inference
                client_state.inference()
                logger.info(f"Started inference for client: {client_id}")
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
        self,
        client: Dict[Text, Any],
        server: WebsocketServer,
        message: AnyStr
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
        if client_id in self._clients:
            try:
                self._clients[client_id].audio_source.process_message(message)
            except Exception as e:
                logger.error(f"Error processing message from client {client_id}: {e}")

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

        client = next(
            (c for c in self.server.clients if c["id"] == client_id),
            None
        )
        
        if client is not None:
            try:
                self.server.send_message(client, message)
            except Exception as e:
                logger.error(f"Failed to send message to client {client_id}: {e}")

    def run(self) -> None:
        """Start the WebSocket server."""
        logger.info(f"Starting WebSocket server on {self.uri}")
        try:
            self.server.run_forever()
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.close_all()

    def close(self, client_id: Text) -> None:
        """Close a specific client's connection and cleanup resources.

        Parameters
        ----------
        client_id : Text
            Client identifier to close
        """
        if client_id in self._clients:
            try:
                self._clients[client_id].audio_source.close()
                del self._clients[client_id]
                logger.info(f"Closed connection for client: {client_id}")
            except Exception as e:
                logger.error(f"Error closing client {client_id}: {e}")

    def close_all(self) -> None:
        """Shutdown the server and cleanup all client connections."""
        logger.info("Shutting down server...")
        try:
            for client_id in list(self._clients.keys()):
                self.close(client_id)
            if self.server is not None:
                self.server.shutdown_gracefully()
            logger.info("Server shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
