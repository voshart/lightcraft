# ORF research provenance and acceptance gates

Task: `LR-IMP-FORMATS` / M11.3. Started 2026-10-08. Baseline:
`629e39380e296f588c64cd9c0053a8edc3528f36` of storytold/lightcraft.

## Inputs and exposure disclosure

- Read: LightCraft's existing MIT/Apache code, the user-provided photographs,
  published metadata descriptions, PR #240's description and maintainer discussion.
- Not used as inputs: PR #240's `orfc.rs`, its compressed encoder, other ORF
  decoder source, camera coefficient tables, Adobe profiles or lens profiles.
- The author is Codex. No assertion is made that an AI model's training contains
  no decoder knowledge. Instead, every new packing rule is tied to measurements
  collected before reference comparison. No compressed rule is supplied from
  remembered code or reconstructed from the rejected contribution.
- Metadata and simple packed-pair fixtures are independently generated with
  LightCraft's existing TIFF writer. They do not encode a compressed ORF stream.
- The creator has explicitly released all ten initial originals as CC0 in a
  separate [public corpus repository](https://github.com/voshart/Rust-Olympus-RAW-decoder/tree/4d29fe4886d893883a78f6099f02d8dd17d47192/corpus/voshart-olympus).
  They remain unchanged. No media, derived sensor arrays or binary instruments
  are committed to LightCraft. A public E-M5 II held-out sample now also passes
  every-sample comparison; broader camera/mode claims still need distinct evidence.
- Additional inputs: fourteen licence-verified external CC0 specimens and their
  primary corpus metadata. No third-party decoder source or corpus script was
  opened. See [source checks and findings](https://github.com/voshart/Rust-Olympus-RAW-decoder/blob/4d29fe4886d893883a78f6099f02d8dd17d47192/research/public-samples.md).

## Evidence trail

| Id | Experiment | Result | Limit |
|---|---|---|---|
| C1 | `inspect_orf.py`, eight ORFs and two JPEGs, hashed before use | Models, lenses, dimensions, CFA and strip budgets recorded | Local specimen coverage only |
| C2 | Compare E-M5 II Exif CFA in ordinary/high-resolution modes | RGGB versus GRBG | No model-table inference |
| P1 | Divide high-resolution strip count by height and width | 14,848 bytes/row; ten samples/16 bytes | One initial original, later released as CC0 |
| P2 | Count padding and compare low/high-nibble hypotheses in masked columns | 6,432,896 zero pads; sharply different border distributions | Statistics alone do not prove packing |
| P3 | Compare the Python hypothesis and final Rust reader to black-box sensor output | All three full-sensor hashes agree; 64,328,960 identical samples, including borders | Reference implementation may itself have bugs |
| P4 | Public held-out PIXLS.US 2856 | Python, final Rust reader and reference match all 64,328,960 sensor samples | Same E-M5 II mode; no broader camera claim |
| S1 | Independently generated packed and Exif fixtures | Four Bayer layouts/two byte orders, boundary values, truncations and mutations | Round trips are supporting evidence, not independent correctness |
| S2 | Synthetic maker-note crop at u64::MAX | Old unchecked crop addition overflows; checked coordinates fall back to the sensor area | Non-conforming numeric tag type |
| X1 | Isolated compressed-strip byte perturbations | Reproducible reference failures/difference extents | Does not establish compressed coding rules |

Local artifacts: `plan/orf-research/private-manifest.json`,
`packed12-reference.json`, `compressed-perturbations.json`, and the Rust audit
sensor dump. Input and output hashes make accidental specimen changes detectable.
Tools write observations with create-new semantics and open photographs read-only.
The separate public repository versions the released originals, whitelisted
observations, instrument versions and per-file comparison hashes; derived sensor
arrays and reference binaries remain local.

## External measuring instrument

The separately installed binary Python package rawpy **0.27.1** uses LibRaw
**0.22.1**. NumPy **2.4.3** supports array comparisons. They are installed only
in the gitignored local research directory; none is a Cargo dependency, shipped
asset or runtime dependency. Their source was not opened. Only sensor arrays and
geometry were read; colour matrices and camera tables were not consulted.

Pinned Windows CPython 3.13 wheel hashes (SHA-256):

| Wheel | Hash |
|---|---|
| rawpy-0.27.1-cp313-cp313-win_amd64.whl | 1d2d32bfe7df6421f214f0502d34bd599ce53156401928c4f538336f2f8b3689 |
| numpy-2.4.3-cp313-cp313-win_amd64.whl | 0a60e17a14d640f49146cb38e3f105f571318db7826d9b6fef7e4dce758faecd |

Binary-distribution sources: [rawpy on PyPI](https://pypi.org/project/rawpy/0.27.1/),
[NumPy on PyPI](https://pypi.org/project/numpy/2.4.3/).
The optional instrument is not needed to build, test or run LightCraft.

## Before a compressed decoder is acceptable

1. Independently derive the bitstream description. For each non-obvious rule,
   record specimen hash, operation, competing hypotheses and discriminating
   observations. Preserve failed hypotheses rather than retrofitting a story.
2. Review the specification separately from code. Disclose exposure to other
   implementations; a new AI session alone is not provenance evidence.
3. Implement only the approved specification, with independently supported test
   vectors. Keep unknown variants explicitly unsupported.
4. Compare every sensor sample on held-out files, stating the full dimensions,
   active crop, CFA origin and any unverified margins. Add bounded allocation,
   arithmetic, truncation and mutation coverage. Run the full project CI.
5. Keep camera colour and lens correction acceptance separate from unpacking.
   Accurate samples do not establish Lightroom rendering parity.

This work satisfies the observed padded packed-layout gate on two E-M5 II files. It does **not**
claim that the compressed specification, broader camera verification or colour
calibration gates have been passed. A rights-holder grant is an alternative to
independent derivation only for exactly the implementation covered by that grant;
the maintainers must decide whether it fits the project policy.
