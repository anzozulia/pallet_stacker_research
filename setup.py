"""
setup.py — Cython extension build for pallet_packer._brkga_core.

Most metadata lives in pyproject.toml; this file exists only because
setuptools' declarative config can't yet express Cython extensions in a
fully PEP 621-compliant way (as of setuptools 75.x / Cython 3.2).

Compile flags
-------------
  -O3                aggressive optimisation
  -ffast-math        relaxed FP semantics (matches Numba's `fastmath=True`)
  -march=native      use SIMD instructions available on the build host
                     (we ship inside a Docker image with a known x86_64
                     baseline, so this is reproducible)

Compiler directives
-------------------
  boundscheck=False         skip array bounds checks (we already validate)
  wraparound=False          no negative indexing
  cdivision=True            C division semantics (no Python ZeroDivisionError)
  initializedcheck=False    skip memoryview init check
  language_level=3          modern Python syntax

Adding a new .pyx
-----------------
1. Drop it under `pallet_packer/_brkga_core/` (alongside the .py file
   it replaces).
2. Add an `Extension(...)` entry in EXTENSION_MODULES below.
3. `pip install --no-build-isolation -e .` inside Docker.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError:
    print(
        "ERROR: Cython is required to build the accelerated decoders.\n"
        "       Install with: pip install 'cython>=3.0,<4.0'\n"
        "       Or build inside the project Dockerfile.",
        file=sys.stderr,
    )
    raise


SRC = Path("pallet_packer/_brkga_core")

# Common compile + directive options shared by every extension.
EXTRA_COMPILE_ARGS = ["-O3", "-ffast-math", "-march=native"]
EXTRA_LINK_ARGS: list[str] = []

# OpenMP flags — applied only to extensions that use cython.parallel.prange
# (Phase 5: batch decoders). On Linux/macOS GCC this is -fopenmp for both
# compile and link. On clang/Apple it would be -fopenmp + -lomp; we always
# build inside Docker (debian-based GCC), so just -fopenmp.
OPENMP_COMPILE_ARGS = EXTRA_COMPILE_ARGS + ["-fopenmp"]
OPENMP_LINK_ARGS = EXTRA_LINK_ARGS + ["-fopenmp"]

COMPILER_DIRECTIVES = {
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "initializedcheck": False,
    "language_level": "3",
}

# One Extension per .pyx; sources resolved at cythonize time.
# When the port progresses, add more entries here.
EXTENSION_MODULES = [
    Extension(
        name="pallet_packer._brkga_core._cyhello",
        sources=[str(SRC / "_cyhello.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
    ),
    # Phase 1: geometric scoring primitives (find_best_wall/corner/in_slab).
    # Numba fallback in jit_primitives.py stays as the reference impl.
    Extension(
        name="pallet_packer._brkga_core.jit_primitives_cy",
        sources=[str(SRC / "jit_primitives_cy.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
    ),
    # Phase 2: constraint helpers (_check_load_on_top, _check_cog_envelope,
    # _apply_*). Numba fallback in jit_constraints.py stays as reference.
    Extension(
        name="pallet_packer._brkga_core.jit_constraints_cy",
        sources=[str(SRC / "jit_constraints_cy.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
    ),
    # Phase 3a: brkga_v3_fast helpers (find_best_dftrc + commit_ems)
    # ported to Cython so geom decoders can call them via cimport.
    Extension(
        name="pallet_packer._brkga_core.v3fast_cy",
        sources=[str(SRC / "v3fast_cy.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
    ),
    # Phase 3a/b/c/d: geometric decoders (all 4 entry points).
    # Phase 5: batch decoders use cython.parallel.prange → OpenMP flags.
    Extension(
        name="pallet_packer._brkga_core.jit_decoders_geom_cy",
        sources=[str(SRC / "jit_decoders_geom_cy.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=OPENMP_COMPILE_ARGS,
        extra_link_args=OPENMP_LINK_ARGS,
    ),
    # Phase 4: constraint-aware decoders (cstr modes 0/1/2/3/4). Cimports
    # the helpers from jit_constraints_cy + jit_primitives_cy +
    # v3fast_cy + jit_decoders_geom_cy. Same OpenMP flags for Phase 5b.
    Extension(
        name="pallet_packer._brkga_core.jit_decoders_cstr_cy",
        sources=[str(SRC / "jit_decoders_cstr_cy.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=OPENMP_COMPILE_ARGS,
        extra_link_args=OPENMP_LINK_ARGS,
    ),
]


setup(
    ext_modules=cythonize(
        EXTENSION_MODULES,
        compiler_directives=COMPILER_DIRECTIVES,
        annotate=True,  # generate .html annotations alongside .c
    ),
)
