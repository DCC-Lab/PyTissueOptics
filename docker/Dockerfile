FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libgl1 \
    libxi6 \
    libsm6 \
    libfontconfig1 \
    ocl-icd-opencl-dev \
    pocl-opencl-icd \
    clinfo \
    freeglut3-dev \
    xvfb \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    libegl1-mesa-dev \
    libosmesa6-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /PyTissueOptics
RUN git clone https://github.com/DCC-Lab/PyTissueOptics.git . && \
    git checkout jlb/setup-ci && \
    pip install --no-cache-dir -r requirements.txt

ENV QT_QPA_PLATFORM=offscreen
ENV PTO_DEVICE_INDEX=0
ENV PTO_N_WORK_UNITS=128
ENV PTO_MAX_MEMORY_MB=1024

CMD ["bash"]
