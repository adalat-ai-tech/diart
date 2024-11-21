from abc import ABC, abstractmethod
from pathlib import Path
from queue import SimpleQueue
from typing import Text, Optional, AnyStr, Dict, Any, Union, Tuple

import numpy as np
import sounddevice as sd
import torch
from einops import rearrange
from rx.subject import Subject
from torchaudio.io import StreamReader
from websocket_server import WebsocketServer

from . import utils
from .audio import FilePath, AudioLoader


class AudioSource(ABC):
    """Represents a source of audio that can start streaming via the `stream` property.

    Parameters
    ----------
    uri: Text
        Unique identifier of the audio source.
    sample_rate: int
        Sample rate of the audio source.
    """

    def __init__(self, uri: Text, sample_rate: int):
        self.uri = uri
        self.sample_rate = sample_rate
        self.stream = Subject()

    @property
    def duration(self) -> Optional[float]:
        """The duration of the stream if known. Defaults to None (unknown duration)."""
        return None

    @abstractmethod
    def read(self):
        """Start reading the source and yielding samples through the stream."""
        pass

    @abstractmethod
    def close(self):
        """Stop reading the source and close all open streams."""
        pass


class FileAudioSource(AudioSource):
    """Represents an audio source tied to a file.

    Parameters
    ----------
    file: FilePath
        Path to the file to stream.
    sample_rate: int
        Sample rate of the chunks emitted.
    padding: (float, float)
        Left and right padding to add to the file (in seconds).
        Defaults to (0, 0).
    block_duration: int
        Duration of each emitted chunk in seconds.
        Defaults to 0.5 seconds.
    """

    def __init__(
        self,
        file: FilePath,
        sample_rate: int,
        padding: Tuple[float, float] = (0, 0),
        block_duration: float = 0.5,
    ):
        super().__init__(Path(file).stem, sample_rate)
        self.loader = AudioLoader(self.sample_rate, mono=True)
        self._duration = self.loader.get_duration(file)
        self.file = file
        self.resolution = 1 / self.sample_rate
        self.block_size = int(np.rint(block_duration * self.sample_rate))
        self.padding_start, self.padding_end = padding
        self.is_closed = False

    @property
    def duration(self) -> Optional[float]:
        # The duration of a file is known
        return self.padding_start + self._duration + self.padding_end

    def read(self):
        """Send each chunk of samples through the stream"""
        waveform = self.loader.load(self.file)

        # Add zero padding at the beginning if required
        if self.padding_start > 0:
            num_pad_samples = int(np.rint(self.padding_start * self.sample_rate))
            zero_padding = torch.zeros(waveform.shape[0], num_pad_samples)
            waveform = torch.cat([zero_padding, waveform], dim=1)

        # Add zero padding at the end if required
        if self.padding_end > 0:
            num_pad_samples = int(np.rint(self.padding_end * self.sample_rate))
            zero_padding = torch.zeros(waveform.shape[0], num_pad_samples)
            waveform = torch.cat([waveform, zero_padding], dim=1)

        # Split into blocks
        _, num_samples = waveform.shape
        chunks = rearrange(
            waveform.unfold(1, self.block_size, self.block_size),
            "channel chunk sample -> chunk channel sample",
        ).numpy()

        # Add last incomplete chunk with padding
        if num_samples % self.block_size != 0:
            last_chunk = (
                waveform[:, chunks.shape[0] * self.block_size :].unsqueeze(0).numpy()
            )
            diff_samples = self.block_size - last_chunk.shape[-1]
            last_chunk = np.concatenate(
                [last_chunk, np.zeros((1, 1, diff_samples))], axis=-1
            )
            chunks = np.vstack([chunks, last_chunk])

        # Stream blocks
        for i, waveform in enumerate(chunks):
            try:
                if self.is_closed:
                    break
                self.stream.on_next(waveform)
            except BaseException as e:
                self.stream.on_error(e)
                break
        self.stream.on_completed()
        self.close()

    def close(self):
        self.is_closed = True


class MicrophoneAudioSource(AudioSource):
    """Audio source tied to a local microphone.

    Parameters
    ----------
    block_duration: int
        Duration of each emitted chunk in seconds.
        Defaults to 0.5 seconds.
    device: int | str | (int, str) | None
        Device identifier compatible for the sounddevice stream.
        If None, use the default device.
        Defaults to None.
    """

    def __init__(
        self,
        block_duration: float = 0.5,
        device: Optional[Union[int, Text, Tuple[int, Text]]] = None,
    ):
        # Use the lowest supported sample rate
        sample_rates = [16000, 32000, 44100, 48000]
        best_sample_rate = None
        for sr in sample_rates:
            try:
                sd.check_input_settings(device=device, samplerate=sr)
            except Exception:
                pass
            else:
                best_sample_rate = sr
                break
        super().__init__(f"input_device:{device}", best_sample_rate)

        # Determine block size in samples and create input stream
        self.block_size = int(np.rint(block_duration * self.sample_rate))
        self._mic_stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            latency=0,
            blocksize=self.block_size,
            callback=self._read_callback,
            device=device,
        )
        self._queue = SimpleQueue()

    def _read_callback(self, samples, *args):
        self._queue.put_nowait(samples[:, [0]].T)

    def read(self):
        self._mic_stream.start()
        while self._mic_stream:
            try:
                while self._queue.empty():
                    if self._mic_stream.closed:
                        break
                self.stream.on_next(self._queue.get_nowait())
            except BaseException as e:
                self.stream.on_error(e)
                break
        self.stream.on_completed()
        self.close()

    def close(self):
        self._mic_stream.stop()
        self._mic_stream.close()


class WebSocketAudioSource(AudioSource):
    """Represents a source of audio coming from the network using the WebSocket protocol.

    Parameters
    ----------
    sample_rate: int
        Sample rate of the chunks emitted.
    """

    def __init__(
        self,
        uri: str,
        sample_rate: int,
    ):
        # FIXME sample_rate is not being used, this can be confusing and lead to incompatibilities.
        #  I would prefer the client to send a JSON with data and sample rate, then resample if needed
        super().__init__(uri, sample_rate)

    def process_message(self, message: AnyStr):
        """Decode and process an incoming audio message."""
        # Send decoded audio to pipeline
        self.stream.on_next(utils.decode_audio(message))

    def read(self):
        """Starts running the websocket server and listening for audio chunks"""
        pass

    def close(self):
        """Complete the audio stream for this client."""
        self.stream.on_completed()


class WebSocketAudioHandler:
    """
    Handles a WebSocket server and manages audio streams from multiple client connections.
    """

    def __init__(
        self,
        sample_rate: int,
        host: Text = "127.0.0.1",
        port: int = 7007,
        key: Optional[Union[Text, Path]] = None,
        certificate: Optional[Union[Text, Path]] = None,
    ):
        self.server = WebsocketServer(host, port, key=key, cert=certificate)
        self.server.set_fn_message_received(self._on_message_received)

        self.clients: Dict[Text, WebSocketAudioSource] = {}
        self.uri = f"{host}:{port}"
        self.sample_rate = sample_rate

    def _on_message_received(
        self,
        client: Dict[Text, Any],
        server: WebsocketServer,
        message: AnyStr,
    ):
        client_id = client["id"]

        # Ensure the client has an associated WebSocketAudioSource
        if client_id not in self.clients:
            self.clients[client_id] = WebSocketAudioSource(
                uri=f"{self.uri}:{client_id}",
                sample_rate=self.sample_rate,
            )

        # Pass the message to the respective WebSocketAudioSource
        self.clients[client_id].process_message(message)

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
        if client_id in self.clients:
            self.clients[client_id].close()

    def close_all(self):
        """Shuts down the server gracefully, invoking close on each audio source."""
        if self.server is not None:
            for client_id in self.clients.keys():
                self.close(client_id)
            self.server.shutdown_gracefully()


class TorchStreamAudioSource(AudioSource):
    def __init__(
        self,
        uri: Text,
        sample_rate: int,
        streamer: StreamReader,
        stream_index: Optional[int] = None,
        block_duration: float = 0.5,
    ):
        super().__init__(uri, sample_rate)
        self.block_size = int(np.rint(block_duration * self.sample_rate))
        self._streamer = streamer
        self._streamer.add_basic_audio_stream(
            frames_per_chunk=self.block_size,
            stream_index=stream_index,
            format="fltp",
            sample_rate=self.sample_rate,
        )
        self.is_closed = False

    def read(self):
        for item in self._streamer.stream():
            try:
                if self.is_closed:
                    break
                # shape (samples, channels) to (1, samples)
                chunk = np.mean(item[0].numpy(), axis=1, keepdims=True).T
                self.stream.on_next(chunk)
            except BaseException as e:
                self.stream.on_error(e)
                break
        self.stream.on_completed()
        self.close()

    def close(self):
        self.is_closed = True


class AppleDeviceAudioSource(TorchStreamAudioSource):
    def __init__(
        self,
        sample_rate: int,
        device: str = "0:0",
        stream_index: int = 0,
        block_duration: float = 0.5,
    ):
        uri = f"apple_input_device:{device}"
        streamer = StreamReader(device, format="avfoundation")
        super().__init__(uri, sample_rate, streamer, stream_index, block_duration)
