from pathlib import Path
from typing import Union, Text, Optional, AnyStr, Dict, Any
from websocket_server import WebsocketServer

from . import blocks
from . import sources as src
from .inference import StreamingInference
from .progress import ProgressBar, RichProgressBar, TQDMProgressBar


class StreamingInferenceHandler:
    """Handles inference for multiple audio sources.

    Parameters
    ----------
    pipeline: StreamingPipeline
        Configured speaker diarization pipeline.
    batch_size: int
        Number of inputs to send to the pipeline at once.
        Defaults to 1.
    do_profile: bool
        If True, compute and report the processing time of the pipeline.
        Defaults to True.
    do_plot: bool
        If True, draw predictions in a moving plot.
        Defaults to False.
    show_progress: bool
        If True, show a progress bar.
        Defaults to True.
    progress_bar: Optional[diart.progress.ProgressBar]
        Progress bar.
        If description is not provided, set to 'Streaming <source uri>'.
        Defaults to RichProgressBar().
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
        # StreamingInference params
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.do_profile = do_profile
        self.do_plot = do_plot
        self.show_progress = show_progress
        self.progress_bar = progress_bar

        # Client-specific params
        # to handle audio streams and their inference
        self.uri = f"{host}:{port}"
        self.sample_rate = sample_rate
        self.sources: Dict[Text, src.WebSocketAudioSource] = {}
        self.inferences: Dict[Text, StreamingInference] = {}

        # Server init
        self.server = WebsocketServer(host, port, key=key, cert=certificate)
        self.server.set_fn_new_client(self._on_connect)
        self.server.set_fn_message_received(self._on_message_received)

    def _on_connect(
        self,
        client: Dict[Text, Any],
        server: WebsocketServer,
    ):
        client_id = client["id"]

        # Ensure the client has an associated WebSocketAudioSource
        if client_id not in self.sources:

            # Source init
            self.sources[client_id] = src.WebSocketAudioSource(
                uri=f"{self.uri}:{client_id}",
                sample_rate=self.sample_rate,
            )

            # Run online inference
            self.inferences[client_id] = StreamingInference(
                pipeline=self.pipeline,
                source=self.sources[client_id],
                batch_size=self.batch_size,
                do_profile=self.do_profile,
                do_plot=self.do_plot,
                show_progress=self.show_progress,
            )

            # # Write to disk if required
            # if self.output is not None:
            #     inference.attach_observers(
            #         RTTMWriter(audio_source.uri, args.output / f"{audio_source.uri}.rttm")
            #     )

            # Send back responses as RTTM text lines
            self.inferences[client_id].attach_hooks(lambda ann_wav: self.send(
                client_id=client_id,
                message=ann_wav[0].to_rttm(),
            ))

            # Run server and pipeline
            self.inferences[client_id]()

    def _on_message_received(
        self,
        client: Dict[Text, Any],
        server: WebsocketServer,
        message: AnyStr,
    ):
        client_id = client["id"]
        # Pass the message to the respective WebSocketAudioSource
        self.sources[client_id].process_message(message)

    def send(self, client_id: Text, message: AnyStr):
        """Send a message to a specific client."""
        client = next(
            (c for c in self.server.clients if c["id"] == client_id), None
        )
        if client is not None and len(message) > 0:
            self.server.send_message(client, message)

    def run(self):
        """Starts the WebSocket server."""
        self.server.run_forever()

    def close(self, client_id: Text):
        """Closes audio stream of a specific client"""
        if client_id in self.sources:
            self.sources[client_id].close()

    def close_all(self):
        """Shuts down the server gracefully, invoking close on each audio source."""
        if self.server is not None:
            for client_id in self.sources.keys():
                self.close(client_id)
            self.server.shutdown_gracefully()

