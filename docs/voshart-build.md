# Voshart personal build

This branch combines the contributor's verified Olympus decoding, AI Bayer RAW denoise,
face detection and pure-Rust recognition with an upstream checkpoint. It is an experimental
personal build. Upstream acceptance is tracked separately in PRs #172, #173, #174, #360 and #384.

The fork's main branch mirrors upstream. Feature PRs stay separate; changes made to this
combined branch do not become extra changes in those PRs.

## Windows

From this checkout, run `powershell -ExecutionPolicy Bypass -File scripts/run-voshart.ps1 -Build`.
Later launches can omit `-Build`. Add `-Release` for a release build, or `-Demo` for the
procedural demo. The launcher runs ordinary Cargo builds, not the complete test suite.
`CARGO_TARGET_DIR` can select a build directory; use one directory per checkout.

No trained model weights are bundled or downloaded at startup. Install models separately
through Settings after reviewing their terms. The RawNIND download offer is disabled by
default; user-supplied denoise models remain supported. Supported ORF coding profiles and
remaining colour/lens gaps are documented in `docs/raw/orf12-compressed-measured.md`.

For the complete language fonts, use the same craft-fonts commit recorded by the workflows:
`8dcdacd5153e64560d109541a47d806f26f048c0`, set `CRAFT_FONTS_DIR` to that checkout, and build.

## Updating

Update at deliberate checkpoints: preserve the tested personal-build head, refresh the
independent contribution branches, then integrate each remaining feature and run
`cargo xtask ci` before adding the next. Once a contribution lands upstream, start the next
integration from that upstream version and omit the already accepted patch. Keep a known
working build while testing a newer checkpoint.
