# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""
pallet_packer._brkga_core._cyhello — Cython build smoke test.

A trivial extension that exists only to verify the build pipeline
(pyproject.toml + setup.py + Dockerfile + cythonize) works end-to-end
BEFORE we start porting the real decoders. If `python -c "from
pallet_packer._brkga_core._cyhello import add, sum_array; print(add(2, 3))"`
prints `5`, the pipeline is good.

Delete this file once the first real Cython decoder lands.
"""
import numpy as np
cimport numpy as cnp


cpdef int add(int a, int b) noexcept nogil:
    """Sanity check: pure C addition with no Python overhead."""
    return a + b


cpdef double sum_array(double[::1] arr) noexcept nogil:
    """Sanity check: typed memoryview sum, matches numpy's array.sum()."""
    cdef Py_ssize_t i, n = arr.shape[0]
    cdef double s = 0.0
    for i in range(n):
        s += arr[i]
    return s
