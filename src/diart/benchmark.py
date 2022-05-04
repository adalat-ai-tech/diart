import argparse
from pathlib import Path

import torch
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

import diart.operators as dops
import diart.sources as src
from diart.pipelines import OnlineSpeakerDiarization, PipelineConfig
from diart.sinks import RTTMWriter

# Define script arguments
parser = argparse.ArgumentParser()
parser.add_argument("root", type=str, help="Directory with audio files <conversation>.(wav|flac|m4a|...)")
parser.add_argument("--reference", type=str, help="Directory with RTTM files <conversation>.rttm")
parser.add_argument("--step", default=0.5, type=float, help="Source sliding window step in seconds. Defaults to 0.5")
parser.add_argument("--latency", default=0.5, type=float, help="System latency in seconds. Defaults to 0.5")
parser.add_argument("--tau", default=0.5, type=float, help="Activity threshold tau active in [0,1]. Defaults to 0.5")
parser.add_argument("--rho", default=0.3, type=float, help="Speech ratio threshold rho update in [0,1]. Defaults to 0.3")
parser.add_argument("--delta", default=1, type=float, help="Maximum distance threshold delta new in [0,2]. Defaults to 1")
parser.add_argument("--gamma", default=3, type=float, help="Parameter gamma for overlapped speech penalty. Defaults to 3")
parser.add_argument("--beta", default=10, type=float, help="Parameter beta for overlapped speech penalty. Defaults to 10")
parser.add_argument("--max-speakers", default=20, type=int, help="Maximum number of identifiable speakers. Defaults to 20")
parser.add_argument("--batch-size", default=32, type=int, help="For segmentation and embedding pre-calculation. If lower than 2, run fully online and estimate real-time latency. Defaults to 32")
parser.add_argument("--output", type=str, help="Output directory to store RTTM files. Defaults to `root`")
parser.add_argument("--gpu", dest="gpu", action="store_true", help="Add this flag to run on GPU")
args = parser.parse_args()

args.root = Path(args.root)
assert args.root.is_dir(), "Root argument must be a directory"
if args.reference is not None:
    args.reference = Path(args.reference)
    assert args.reference.is_dir(), "Reference argument must be a directory"
args.output = args.root if args.output is None else Path(args.output)
args.output.mkdir(parents=True, exist_ok=True)

# Define online speaker diarization pipeline
config = PipelineConfig(
    step=args.step,
    latency=args.latency,
    tau_active=args.tau,
    rho_update=args.rho,
    delta_new=args.delta,
    gamma=args.gamma,
    beta=args.beta,
    max_speakers=args.max_speakers,
    device=torch.device("cuda") if args.gpu else None,
)
pipeline = OnlineSpeakerDiarization(config)

# Run inference
chunk_loader = src.ChunkLoader(pipeline.sample_rate, pipeline.duration, config.step)
for filepath in args.root.expanduser().iterdir():
    num_chunks = chunk_loader.num_chunks(filepath)

    # Stream fully online if batch size is 1 or lower
    source = None
    if args.batch_size < 2:
        source = src.FileAudioSource(
            filepath,
            filepath.stem,
            src.RegularAudioFileReader(pipeline.sample_rate, pipeline.duration, config.step),
            # Benchmark the processing time of a single chunk
            profile=True,
        )
        observable = pipeline.from_source(source, output_waveform=False)
    else:
        observable = pipeline.from_file(filepath, batch_size=args.batch_size, verbose=True)

    observable.pipe(
        dops.progress(f"Streaming {filepath.stem}", total=num_chunks, leave=source is None)
    ).subscribe(
        RTTMWriter(path=args.output / f"{filepath.stem}.rttm")
    )

    if source is not None:
        source.read()

# Run evaluation
if args.reference is not None:
    metric = DiarizationErrorRate(collar=0, skip_overlap=False)
    for ref_path in args.reference.iterdir():
        ref = load_rttm(ref_path).popitem()[1]
        hyp = load_rttm(args.output / ref_path.name).popitem()[1]
        metric(ref, hyp)
    print()
    metric.report(display=True)
    print()
