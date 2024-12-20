# Use NVIDIA CUDA base image
FROM docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install sudo, git, wget, gcc, g++, and other essential build tools
RUN apt-get update && \
    apt-get install -y sudo git wget build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# Install Python 3.10 using Conda
RUN conda install python=3.10

# Upgrade pip and setuptools to avoid deprecation warnings
RUN pip install --upgrade pip setuptools

# Set Python 3.11 as default by creating a symbolic link
RUN ln -sf /opt/conda/bin/python3.10 /opt/conda/bin/python && \
    ln -sf /opt/conda/bin/python3.10 /usr/bin/python

# Verify installations
RUN python --version && \
    gcc --version && \
    g++ --version && \
    pip --version && \
    conda --version

RUN git clone --branch develop-server https://github.com/adalat-ai-tech/diart.git
WORKDIR /diart

# Install diart dependencies
RUN conda install portaudio pysoundfile ffmpeg -c conda-forge
RUN pip install -e .

# Expose the port the app runs on
EXPOSE 8080

# Define environment variable to prevent Python from buffering stdout/stderr
# and writing byte code to file
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python", "-m", "diart.console.serve", "--host", "0.0.0.0", "--port", "7007", "--segmentation", "pyannote/segmentation-3.0", "--embedding", "speechbrain/spkrec-resnet-voxceleb", "--tau-active", "0.45", "--rho-update", "0.25", "--delta-new", "0.6", "--latency", "5", "--max-speakers", "3"]
