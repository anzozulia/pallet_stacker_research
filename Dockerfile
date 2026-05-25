# Pallet packer — development + runtime image.
#
# Why this exists: we're porting the v3.12 BRKGA decoders from Numba
# (`@njit`) to Cython (`.pyx`). The build needs a controlled toolchain
# (gcc + cython + numpy) AND deterministic results require a pinned
# Python + numpy combination. Docker is the simplest way to give every
# developer + CI + production the exact same stack.
#
# Build (from repo root):
#     docker build -t pallet-packer:dev .
#
# Run a one-shot evaluation:
#     docker run --rm -v "$(pwd)":/app pallet-packer:dev \
#         python scripts/run_v310_industry_eval.py --time 10
#
# Drop into a shell with the code mounted:
#     docker run --rm -it -v "$(pwd)":/app pallet-packer:dev bash
#
# Pre-build the Cython extensions inside the image (so wheels are baked in):
#     docker run --rm -v "$(pwd)":/app pallet-packer:dev \
#         pip install --no-build-isolation -e .

FROM python:3.13-slim-bookworm

# C / C++ toolchain for Cython + Numba LLVM IR support.
# git is in case any pip dep needs VCS access.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install runtime + build dependencies. We pin numpy and numba so JIT
# results are bit-for-bit reproducible across rebuilds.
#
# numba 0.65.x is the latest stable that supports numpy 2.x.
# cython 3.x is the modern API used by scipy/scikit-learn.
RUN pip install --no-cache-dir \
        "numpy>=2.4,<2.5" \
        "numba>=0.65,<0.66" \
        "cython>=3.0,<4.0" \
        "ortools>=9.10,<10.0" \
        "setuptools>=68" \
        "wheel>=0.40"

# Copy source last so dependency layer caches well across iterations.
COPY . /app

# Default to bash; the user passes whatever command they want at run time.
CMD ["bash"]
