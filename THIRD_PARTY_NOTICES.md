# Third-party notices

Otter includes adaptations of third-party open-source code. The physical
papers cited in the source and documentation remain the primary references
for the scientific models; the notices below record software provenance and
license obligations.

## Scientific reference data

`benchmarks/reference_data/` contains digitized publication coordinates and
author-provided numerical values used only for scientific comparisons.  The
files are attributed individually in machine-readable manifests and dataset
READMEs.  Except where a manifest states a specific open license, their data
license is `NOASSERTION` and Otter's BSD-3-Clause software license does not
apply to them.  See
[`benchmarks/reference_data/README.md`](benchmarks/reference_data/README.md)
for the maintainer's distribution decision and downstream reuse notice.

## JaXRTS

The finite-temperature Geldart–Vosko and Gregori-2007 local-field-correction
implementation in `src/otter/ionic/lfc.py` was copied from
`jaxrts.ee_localfieldcorrections` in
[JaXRTS](https://github.com/JaXRTS/jaxrts), commit
`de309018194a036cf513b4156aee389501308703`. The adaptation translates JAX
quantities to NumPy atomic units and integrates the API with Otter.

Software reference:

> J. Lütgert, S. Schumacher, J. Rips, C. Qu, T. Döppner, and D. Kraus,
> “jaxrts: A Python package for simulating X-ray Thomson scattering spectra
> from dense plasmas using jax,” *Computer Physics Communications* **325**,
> 110173 (2026), DOI
> [10.1016/j.cpc.2026.110173](https://doi.org/10.1016/j.cpc.2026.110173).

JaXRTS license notice:

> BSD 3-Clause License
>
> Copyright (c) 2024 (in alphabetical order), Julian Lütgert, Samuel
> Schumacher
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice,
>    this list of conditions and the following disclaimer.
>
> 2. Redistributions in binary form must reproduce the above copyright
>    notice, this list of conditions and the following disclaimer in the
>    documentation and/or other materials provided with the distribution.
>
> 3. Neither the name of the copyright holder nor the names of its
>    contributors may be used to endorse or promote products derived from
>    this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
> ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
> LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
> CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
> SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
> INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
> CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.
