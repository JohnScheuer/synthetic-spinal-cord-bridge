# Scientific Reproducibility Container for Synthetic Spinal Cord Bridge
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-12.8

RUN apt-get update && apt-get install -y \
    build-essential cmake g++-11 python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install numpy pandas matplotlib

WORKDIR /app
COPY . /app

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

CMD ["./build/spinal_bridge_engine"]
