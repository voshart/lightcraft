# ORF research tools

Local, read-only file observations and optional binary reference experiments.
No compressed decoder, compressed encoder, copied decoder source or camera
tables are in this directory. Python tools are developer tools, not product
dependencies. Ordinary container inspection uses only Python's standard library.

Run from the workspace root (Python 3.10+):

```powershell
python tools/orf-research/inspect_orf.py C:\photos\Olympus --output plan/orf-research/manifest.json
```

The manifest whitelists camera/lens, geometry, CFA, bit/strip budgets and hashes.
It excludes GPS, serials, timestamps and names embedded in metadata. It includes
input basenames and raw strip prefixes: keep private manifests in `plan/`, not
in a commit. ORFs/JPEGs are read without modification. TIFF traversal and file,
entry and value budgets are bounded. Outputs are created exclusively; choose a
new output name to repeat an experiment.

Optional black-box instrument, installed separately in the ignored research area:

```powershell
python -m pip download --only-binary=:all: --dest plan/orf-research/wheels rawpy==0.27.1 numpy==2.4.3
Get-FileHash -Algorithm SHA256 plan/orf-research/wheels/*.whl
python -m pip install --no-index --find-links plan/orf-research/wheels --only-binary=:all: --target plan/orf-research/runtime rawpy==0.27.1 numpy==2.4.3
python tools/orf-research/measure_packed12.py C:\photos\high-res.orf --reference-runtime plan/orf-research/runtime --output plan/orf-research/packing.json
python tools/orf-research/probe_reference.py C:\photos\compressed.orf --reference-runtime plan/orf-research/runtime --output plan/orf-research/perturbations.json
```

Verify the exact wheel hashes in [provenance](../../docs/raw/orf12-provenance.md).
Those hashes are for Windows/CPython 3.13; record distinct hashes on another
platform. Binary wheels only: no build or inspection of decoder source.
`measure_packed12.py` proposes a narrowly scoped, observable packing hypothesis;
the reference only checks it. `probe_reference.py` runs each mutation in a child
with a 20-second timeout and at most 64 cases. Native errors are observations,
not evidence of a coding rule. Neither program reads colour calibration tables.

Check the actual Rust reader, optionally writing local u16 sensor dumps:

```powershell
cargo run -p lightcraft-raw --example orf_audit -- --sensor-dir plan/orf-research/rust-sensor C:\photos\high-res.orf
Get-FileHash -Algorithm SHA256 plan/orf-research/rust-sensor/*.u16le
```

`orf_audit` checks `probe_info == decode().info()` on successful files, reports
unsupported variants, and refuses to overwrite a dump. The sensor array includes
all borders, serialized row-major little-endian u16. Keep media, wheel/runtime
files, sensor arrays and private observations outside Git. The commit should
contain only source, synthetic tests, general format notes and provenance.

These tools currently specify container metadata and one padded packed layout.
They do not constitute an independently established compressed ORF specification.

The creator-released CC0 originals, pinned external specimen identities and
versioned reference observations are in the separate
[Olympus research corpus](https://github.com/voshart/Rust-Olympus-RAW-decoder/tree/4d29fe4886d893883a78f6099f02d8dd17d47192).
Use its Git LFS instructions and SHA-256 verifier; keep media outside this repo.
