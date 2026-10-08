//! Olympus ORF — uncompressed variants.
//!
//! Sources: TIFF 6.0 (the `IIRO`/`MMOR` container is a TIFF with a different magic number), the ExifTool
//! Olympus tag-name documentation (maker-note sub-directories `0x2020` CameraSettings — `0x0101/0x0102`
//! PreviewImageStart/Length — and `0x2040` ImageProcessing — `0x0100` WB_RBLevels, `0x0600` BlackLevel2,
//! `0x0612–0x0615` CropLeft/Top/Width/Height) and our own black-box analysis of CC0 samples from raw.pixls.us
//! (E-1, E-400, XZ-2):
//!
//! - 16 bits per sample: little-endian words; some bodies (E-1, E-400) store 12-bit values in the top bits (the low
//!   four bits zero in more than 99% of samples), which we shift down.
//! - 12-bit packed (XZ-2): each row is a sequence of little-endian 32-bit words read MSB-first (found by testing
//!   candidate bit orders for the smoothest image).
//! - Olympus's compressed ORF (most interchangeable-lens bodies since ~2008) is not decoded: no permissively
//!   licensed description exists. It reports [`RawError::Unsupported`]; the embedded preview still works.
//! - Exif `CFAPattern` (`0xa302`, Exif 2.32) gives the per-file Bayer layout. Without a valid tag, block means
//!   estimate only the green diagonal; the red/blue assignment remains the historical GRBG/RGGB fallback.
//! - E-M5 II High Res Shot: ten 12-bit samples in 16 bytes, five little-endian 3-byte pairs and a zero pad byte.
//!   Independently established from strip size, padding and competing nibble layouts, then checked against
//!   64,328,960 reference sensor samples. See `docs/raw/orf12-format.md` and `orf12-provenance.md`.

use super::white_from_data;
use crate::tiffraw::{Packing, read_image};
use crate::unpack::unpack_msb;
use crate::{BlackLevel, Cfa, ColorData, Mode, OpcodeLists, RawData, RawError, RawFormat, RawImage, Rect, Result};
use lightcraft_geom::Orientation;
use lightcraft_tiff::image::chunk_bytes;
use lightcraft_tiff::makernote::MakerNote;
use lightcraft_tiff::{Ifd, Tiff, Value, makernote, tags as t};
use rayon::prelude::*;

pub(crate) const CAMERA_SETTINGS: u16 = 0x2020;
const IMAGE_PROCESSING: u16 = 0x2040;
pub(crate) const PREVIEW_START: u16 = 0x0101;
pub(crate) const PREVIEW_LENGTH: u16 = 0x0102;
const WB_RB: u16 = 0x0100;
const BLACK: u16 = 0x0600;
const CROP: [u16; 4] = [0x0612, 0x0613, 0x0614, 0x0615];
const EXIF_CFA_PATTERN: u16 = 0xa302;

/// The Olympus maker note.
pub(crate) fn maker_note(bytes: &[u8], tiff: &Tiff) -> Option<MakerNote> {
    let make = tiff.find(t::MAKE).and_then(|e| e.value.as_str()).unwrap_or_default().to_string();
    let e = tiff.exif().and_then(|e| e.get(t::MAKER_NOTE))?;
    makernote::parse_makernote(bytes, e.offset, e.count() as u64, tiff.order, &make)
}

/// A maker-note sub-directory: an IFD pointer (offset relative to the note's base) in new-style notes, or the
/// IFD stored inline as an undefined blob (offsets relative to the base) in old-style ones.
pub(crate) fn sub_ifd(bytes: &[u8], mn: &MakerNote, tag: u16) -> Option<Ifd> {
    let e = mn.ifd.get(tag)?;
    let at = match &e.value {
        Value::Undefined(_) | Value::Byte(_) => e.offset,
        _ => mn.base.checked_add(mn.ifd.u64(tag)?)?,
    };
    let opts = lightcraft_tiff::ParseOptions { max_ifds: 4, max_depth: 1, follow_children: false, ..Default::default() };
    lightcraft_tiff::parse_ifd_at(bytes, at, mn.order, mn.base, false, &opts).ok().map(|(i, _)| i)
}

/// Unpack one row of 12-bit samples stored as little-endian 32-bit words read MSB-first.
pub(crate) fn unpack_row_le32_msb(src: &[u8], bits: u32, out: &mut [u16]) {
    let swapped: Vec<u8> = src
        .chunks(4)
        .flat_map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            [w[3], w[2], w[1], w[0]]
        })
        .collect();
    unpack_msb(&swapped, bits, out);
}

/// Exif 2.32: two SHORT dimensions in the TIFF byte order followed by row-major BYTE colour codes.
fn cfa_from_exif(tiff: &Tiff) -> Option<Cfa> {
    let bytes = tiff.exif()?.bytes(EXIF_CFA_PATTERN)?;
    if bytes.len() != 8 || tiff.order.read_u16(bytes, 0)? != 2 || tiff.order.read_u16(bytes, 2)? != 2 {
        return None;
    }
    let name = match bytes.get(4..8)? {
        [0, 1, 1, 2] => "RGGB",
        [2, 1, 1, 0] => "BGGR",
        [1, 0, 2, 1] => "GRBG",
        [1, 2, 0, 1] => "GBRG",
        _ => return None,
    };
    Some(Cfa::bayer_static(name))
}

/// Estimate the green diagonal using 8×8 block means to reduce sensitivity to single-pixel texture.
/// This cannot distinguish red from blue; the established no-tag fallback is still GRBG or RGGB.
pub(crate) fn cfa_from_data(d: &[u16], w: usize, a: Rect) -> Cfa {
    let Some(h) = d.len().checked_div(w) else { return Cfa::bayer_static("RGGB") };
    let a = a.clipped(w, h);
    let (x0, y0) = (a.x.saturating_add(a.width / 8).saturating_add(1) & !1, a.y.saturating_add(a.height / 8).saturating_add(1) & !1);
    let (x1, y1) = (a.x + a.width - a.width / 8, a.y + a.height - a.height / 8);
    let (mut main, mut anti) = (0u64, 0u64);
    for y in (y0..y1.saturating_sub(7)).step_by(16) {
        for x in (x0..x1.saturating_sub(7)).step_by(16) {
            let mut sums = [0u64; 4];
            for dy in 0..8 {
                for dx in 0..8 {
                    let at = (y + dy).checked_mul(w).and_then(|v| v.checked_add(x + dx));
                    if let Some(value) = at.and_then(|i| d.get(i)) {
                        sums[(dy % 2) * 2 + dx % 2] += u64::from(*value);
                    }
                }
            }
            main += sums[0].abs_diff(sums[3]);
            anti += sums[1].abs_diff(sums[2]);
        }
    }
    Cfa::bayer_static(if main < anti { "GRBG" } else { "RGGB" })
}

/// Five little-endian 12-bit pairs followed by a zero padding byte (ten samples in 16 bytes).
fn unpack_padded_pairs(src: &[u8], row: &mut [u16]) -> Result<()> {
    if !row.len().is_multiple_of(10) || src.len() != row.len() / 10 * 16 {
        return Err(RawError::Corrupt("ORF padded row length mismatch".into()));
    }
    for (group, out) in src.as_chunks::<16>().0.iter().zip(row.as_chunks_mut::<10>().0.iter_mut()) {
        let payload = group.get(..15).ok_or_else(|| RawError::Corrupt("short ORF packed group".into()))?;
        if group.get(15) != Some(&0) {
            return Err(RawError::Unsupported("ORF padded packing with nonzero padding".into()));
        }
        for (pair, target) in payload.as_chunks::<3>().0.iter().zip(out.as_chunks_mut::<2>().0.iter_mut()) {
            let [a, b, c] = pair;
            let [first, second] = target;
            *first = u16::from(*a) | (u16::from(*b & 15) << 8);
            *second = u16::from(*b >> 4) | (u16::from(*c) << 4);
        }
    }
    Ok(())
}

pub(crate) fn decode(bytes: &[u8], mode: Mode) -> Result<RawImage> {
    let tiff = Tiff::parse(bytes)?;
    let ifd0 = tiff.ifds.first().ok_or_else(|| RawError::Corrupt("ORF without IFD0".into()))?;
    let info = ifd0.image()?;
    if info.samples_per_pixel != 1 || info.sample_format != 1 {
        return Err(RawError::Unsupported("ORF without a single unsigned sensor sample per pixel".into()));
    }
    let (w, h) = (info.width as usize, info.height as usize);
    let n = w.checked_mul(h).filter(|n| *n > 0 && *n <= crate::MAX_SAMPLES).ok_or(RawError::Limit("image too large"))?;
    let chunks = info.chunks(bytes.len() as u64);
    let total = chunks.iter().try_fold(0u64, |total, c| {
        let end = c.offset.checked_add(c.len).filter(|end| *end <= bytes.len() as u64);
        if end.is_none() || c.len == 0 {
            return Err(RawError::Corrupt("ORF strip outside file".into()));
        }
        total.checked_add(c.len).ok_or_else(|| RawError::Corrupt("ORF strip lengths overflow".into()))
    })?;
    let stored_bits = total.checked_mul(8).ok_or_else(|| RawError::Corrupt("ORF stored bit count overflow".into()))?;
    let tagged_cfa = cfa_from_exif(&tiff);
    let packed_header = mode == Mode::Header && tagged_cfa.is_some();
    let (mut data, bits) = if info.compression != 1 {
        return Err(RawError::Unsupported(format!("ORF compression {}", info.compression)));
    } else if total >= (n as u64) * 2 {
        let d = read_image(bytes, &info, tiff.order, Packing::Word16)?;
        (d, 16)
    } else if stored_bits >= (n as u64) * 12 && stored_bits < (n as u64) * 13 && chunks.len() == 1 {
        let chunk = chunks.first().ok_or_else(|| RawError::Corrupt("ORF without strip".into()))?;
        let src = chunk_bytes(bytes, chunk).ok_or_else(|| RawError::Corrupt("ORF strip outside file".into()))?;
        if !src.len().is_multiple_of(h) {
            return Err(RawError::Corrupt("ORF rows do not divide strip length".into()));
        }
        let stride = src.len() / h;
        let padded_pairs = w.is_multiple_of(10) && stride == w / 10 * 16;
        if padded_pairs
            && (tiff.order != lightcraft_tiff::ByteOrder::Little
                || !matches!(info.layout, lightcraft_tiff::image::Layout::Strips { rows_per_strip } if rows_per_strip as usize == h))
        {
            return Err(RawError::Unsupported("ORF padded packing not verified for this byte order or strip layout".into()));
        }
        if padded_pairs && src.as_chunks::<16>().0.iter().any(|g| g.get(15) != Some(&0)) {
            return Err(RawError::Unsupported("ORF padded packing with nonzero padding".into()));
        }
        let mut d = if packed_header { Vec::new() } else { vec![0u16; n] };
        d.par_chunks_mut(w).enumerate().try_for_each(|(y, row)| -> Result<()> {
            let start = y.checked_mul(stride).ok_or_else(|| RawError::Corrupt("ORF row offset overflow".into()))?;
            let end = start.checked_add(stride).ok_or_else(|| RawError::Corrupt("ORF row end overflow".into()))?;
            let src = src.get(start..end).ok_or_else(|| RawError::Corrupt("short ORF row".into()))?;
            if padded_pairs {
                unpack_padded_pairs(src, row)?
            } else {
                unpack_row_le32_msb(src, 12, row)
            }
            Ok(())
        })?;
        (RawData::U16(d), 12)
    } else {
        return Err(RawError::Unsupported("Olympus compressed ORF".into()));
    };
    let RawData::U16(ref mut samples) = data else { return Err(RawError::Unsupported("float ORF".into())) };
    let bits = if bits == 16 && samples.iter().step_by(7).filter(|v| *v & 15 != 0).count() * 700 <= n {
        // 12-bit values stored in the top bits
        samples.par_iter_mut().for_each(|v| *v >>= 4);
        12
    } else if bits == 16 {
        let mx = samples.iter().step_by(31).max().copied().unwrap_or(0);
        if mx < 4096 {
            12
        } else if mx < 16384 {
            14
        } else {
            16
        }
    } else {
        bits
    };

    let mn = maker_note(bytes, &tiff);
    let ip = mn.as_ref().and_then(|m| sub_ifd(bytes, m, IMAGE_PROCESSING));
    let active = match ip.as_ref().map(|i| CROP.map(|tag| i.u64(tag).and_then(|v| usize::try_from(v).ok()))) {
        Some([Some(x), Some(y), Some(cw), Some(ch)])
            if cw > 0 && ch > 0 && x.checked_add(cw).is_some_and(|r| r <= w) && y.checked_add(ch).is_some_and(|b| b <= h) =>
        {
            Rect::new(x, y, cw, ch)
        }
        _ => Rect::new(0, 0, w, h),
    };
    let cfa = tagged_cfa.unwrap_or_else(|| cfa_from_data(samples, w, active));
    let black = match ip.as_ref().and_then(|i| i.f64s(BLACK)).as_deref() {
        Some(v @ [_, _, _, _]) => {
            let a = cfa.shifted(active.x, active.y);
            let values = a.pattern.iter().map(|&c| [v[0], (v[1] + v[2]) / 2.0, v[3]][c as usize] as f32).collect();
            BlackLevel { repeat_rows: 2, repeat_cols: 2, values, ..Default::default() }
        }
        _ => BlackLevel::uniform(0.0),
    };
    let wb = ip
        .as_ref()
        .and_then(|i| i.f64s(WB_RB))
        .filter(|v| v.len() >= 2 && v[0] > 0.0 && v[1] > 0.0)
        .map(|v| [(v[0] / 256.0) as f32, 1.0, (v[1] / 256.0) as f32]);
    let white = white_from_data(samples, bits);
    let mut metadata = lightcraft_meta::from_tiff(&tiff);
    metadata.width = Some(active.width as u32);
    metadata.height = Some(active.height as u32);
    let img = RawImage {
        format: RawFormat::Orf,
        width: w,
        height: h,
        cpp: 1,
        data,
        cfa: Some(cfa),
        bits,
        black,
        white: vec![white],
        active_area: active,
        crop: Rect::new(0, 0, active.width, active.height),
        orientation: Orientation::from_exif(ifd0.u16(t::ORIENTATION).unwrap_or(1)),
        color: ColorData::default(),
        wb_multipliers: wb,
        linearized: false,
        opcodes: OpcodeLists::default(),
        metadata,
    };
    img.validate_for(mode)?;
    Ok(img)
}

/// The large preview JPEG referenced by CameraSettings `PreviewImageStart/Length` (relative to the note base).
pub(crate) fn preview(bytes: &[u8]) -> Option<&[u8]> {
    let tiff = Tiff::parse(bytes).ok()?;
    let mn = maker_note(bytes, &tiff)?;
    let cs = sub_ifd(bytes, &mn, CAMERA_SETTINGS)?;
    let start = mn.base.checked_add(cs.u64(PREVIEW_START)?)? as usize;
    let len = cs.u64(PREVIEW_LENGTH)? as usize;
    bytes.get(start..start.checked_add(len)?.min(bytes.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lightcraft_tiff::{ByteOrder, IfdBuilder, ImageData, TiffWriter};

    fn orf_with_exif(w: u32, h: u32, bits: u16, strip: Vec<u8>, order: ByteOrder, exif: Option<IfdBuilder>) -> Vec<u8> {
        let mut ifd = IfdBuilder::new();
        ifd.set(t::IMAGE_WIDTH, Value::Long(vec![w]));
        ifd.set(t::IMAGE_LENGTH, Value::Long(vec![h]));
        ifd.set(t::BITS_PER_SAMPLE, Value::Short(vec![bits]));
        ifd.set(t::COMPRESSION, Value::Short(vec![1]));
        ifd.set(t::PHOTOMETRIC, Value::Short(vec![1]));
        ifd.set(t::MAKE, Value::Ascii("OLYMPUS IMAGING CORP.".into()));
        ifd.set_image(ImageData::Strips { rows_per_strip: h, strips: vec![strip] });
        if let Some(exif) = exif {
            ifd.set_child(t::EXIF_IFD, exif);
        }
        let mut b = TiffWriter::new(order, false).write(&[ifd]).unwrap();
        b[..4].copy_from_slice(if order == ByteOrder::Little { b"IIRO" } else { b"MMOR" });
        b
    }

    fn orf(w: u32, h: u32, bits: u16, strip: Vec<u8>) -> Vec<u8> {
        orf_with_exif(w, h, bits, strip, ByteOrder::Little, None)
    }

    fn exif_cfa(order: ByteOrder, pattern: &[u8]) -> IfdBuilder {
        let mut v = Vec::new();
        order.put_u16(&mut v, 2);
        order.put_u16(&mut v, 2);
        v.extend_from_slice(pattern);
        IfdBuilder::new().with(EXIF_CFA_PATTERN, Value::Undefined(v))
    }

    #[test]
    fn exif_cfa_controls_all_four_layouts_in_both_byte_orders() {
        assert!(RawFormat::Orf.is_supported());
        for order in [ByteOrder::Little, ByteOrder::Big] {
            for name in ["RGGB", "BGGR", "GRBG", "GBRG"] {
                let cfa = Cfa::bayer(name).unwrap();
                let mut strip = Vec::new();
                for _ in 0..16 * 16 {
                    order.put_u16(&mut strip, 1025);
                }
                let bytes = orf_with_exif(16, 16, 16, strip, order, Some(exif_cfa(order, &cfa.pattern)));
                let raw = crate::decode(&bytes).unwrap();
                assert_eq!(raw.cfa, Some(cfa));
                assert_eq!(crate::probe_info(&bytes).unwrap(), raw.info());
            }
        }
    }

    #[test]
    fn malformed_exif_cfa_is_not_used() {
        for pattern in [&[0, 1, 1][..], &[0, 1, 1, 7], &[0, 0, 1, 2], &[1, 1, 0, 2], &[0, 1, 1, 2, 0]] {
            let exif = exif_cfa(ByteOrder::Little, pattern);
            let bytes = orf_with_exif(16, 16, 16, vec![1; 512], ByteOrder::Little, Some(exif));
            assert!(cfa_from_exif(&Tiff::parse(&bytes).unwrap()).is_none());
            assert_eq!(crate::probe_info(&bytes).unwrap(), crate::decode(&bytes).unwrap().info());
        }
        let mut exif = exif_cfa(ByteOrder::Little, &[0, 1, 1, 2]);
        exif.set(EXIF_CFA_PATTERN, Value::Undefined(vec![0, 2, 0, 2, 0, 1, 1, 2]));
        let bytes = orf_with_exif(16, 16, 16, vec![1; 512], ByteOrder::Little, Some(exif));
        assert!(cfa_from_exif(&Tiff::parse(&bytes).unwrap()).is_none());
    }

    #[test]
    fn block_means_tolerate_misleading_single_pixel_texture() {
        let (w, h) = (128usize, 96usize);
        for name in ["RGGB", "GRBG"] {
            let cfa = Cfa::bayer(name).unwrap();
            let mut pixels: Vec<u16> = (0..w * h).map(|i| [1000, 2000, 3000][cfa.color_at(i % w, i / w) as usize]).collect();
            // The old point-wise score sees a false green diagonal at every sampled cell.
            for y in (12..84).step_by(16) {
                for x in (16..112).step_by(8) {
                    if name == "RGGB" {
                        pixels[y * w + x] = 2000;
                        pixels[(y + 1) * w + x + 1] = 2000;
                        pixels[y * w + x + 1] = 5000;
                    } else {
                        pixels[y * w + x + 1] = 2000;
                        pixels[(y + 1) * w + x] = 2000;
                        pixels[y * w + x] = 5000;
                    }
                }
            }
            assert_eq!(cfa_from_data(&pixels, w, Rect::new(0, 0, w, h)).name(), name);
        }
        assert_eq!(cfa_from_data(&[], 0, Rect::new(usize::MAX, usize::MAX, usize::MAX, usize::MAX)).name(), "RGGB");
        let _ = cfa_from_data(&[1000], usize::MAX, Rect::new(usize::MAX, usize::MAX, usize::MAX, usize::MAX));
    }

    fn padded_samples(pixels: &[u16]) -> Vec<u8> {
        let mut strip = Vec::new();
        for group in pixels.as_chunks::<10>().0 {
            for pair in group.as_chunks::<2>().0 {
                let (a, b) = (pair[0], pair[1]);
                strip.extend_from_slice(&[a as u8, ((a >> 8) | (b << 4)) as u8, (b >> 4) as u8]);
            }
            strip.push(0);
        }
        strip
    }

    #[test]
    fn high_res_padded12_roundtrip_and_header_without_samples() {
        let (w, h) = (20usize, 8usize);
        let mut pixels: Vec<u16> = (0..w * h).map(|i| (i * 103 % 4096) as u16).collect();
        pixels[..6].copy_from_slice(&[0, 4095, 1, 4094, 256, 2048]);
        let bytes =
            orf_with_exif(w as u32, h as u32, 16, padded_samples(&pixels), ByteOrder::Little, Some(exif_cfa(ByteOrder::Little, &[1, 0, 2, 1])));
        let raw = crate::decode(&bytes).unwrap();
        assert_eq!(raw.data, RawData::U16(pixels));
        assert_eq!(raw.bits, 12);
        assert_eq!(raw.cfa.unwrap().name(), "GRBG");
        let header = decode(&bytes, Mode::Header).unwrap();
        assert_eq!(header.data.len(), 0, "tagged packed ORF must not allocate sensor samples to probe");
        assert_eq!(header.info(), crate::decode(&bytes).unwrap().info());
        let parsed = Tiff::parse(&bytes).unwrap();
        let offset = parsed.ifds[0].u64(t::STRIP_OFFSETS).unwrap() as usize;
        let mut short = bytes.clone();
        short.truncate(offset + 15);
        assert!(matches!(crate::decode(&short), Err(RawError::Corrupt(_) | RawError::Tiff(_))));
        assert!(matches!(crate::probe_info(&short), Err(RawError::Corrupt(_) | RawError::Tiff(_))));
        let count_at = parsed.ifds[0].get(t::STRIP_BYTE_COUNTS).unwrap().offset as usize;
        let mut outside = bytes.clone();
        outside[count_at..count_at + 4].copy_from_slice(&(bytes.len() as u32 + 1).to_le_bytes());
        assert!(matches!(crate::decode(&outside), Err(RawError::Corrupt(_))));
        let mut unknown = bytes.clone();
        unknown[offset + 15] = 1;
        assert!(matches!(crate::decode(&unknown), Err(RawError::Unsupported(_))));
        // Every truncation and mutations must produce a result, never a panic.
        for length in 0..bytes.len() {
            let _ = crate::decode(&bytes[..length]);
            let _ = crate::probe_info(&bytes[..length]);
        }
        for at in (0..bytes.len()).step_by(7) {
            let mut mutated = bytes.clone();
            mutated[at] ^= 0xff;
            let _ = crate::decode(&mutated);
            let _ = crate::probe_info(&mutated);
        }
    }

    #[test]
    fn compressed_orf_stays_unsupported_even_with_cfa_metadata() {
        let bytes = orf_with_exif(40, 24, 16, vec![0; 40], ByteOrder::Little, Some(exif_cfa(ByteOrder::Little, &[0, 1, 1, 2])));
        assert!(matches!(crate::decode(&bytes), Err(RawError::Unsupported(_))));
        assert!(matches!(crate::probe_info(&bytes), Err(RawError::Unsupported(_))));
    }

    #[test]
    fn unverified_big_endian_padded_packing_is_unsupported() {
        let bytes = orf_with_exif(20, 8, 16, padded_samples(&[1025; 160]), ByteOrder::Big, Some(exif_cfa(ByteOrder::Big, &[0, 1, 1, 2])));
        assert!(matches!(crate::decode(&bytes), Err(RawError::Unsupported(_))));
        assert!(matches!(crate::probe_info(&bytes), Err(RawError::Unsupported(_))));
    }

    #[test]
    fn overflowing_maker_note_crop_does_not_panic() {
        // A new-style Olympus note with one ImageProcessing IFD, wholly synthetic.
        let mut note = b"OLYMPUS\0II\x03\0".to_vec();
        note.extend_from_slice(&1u16.to_le_bytes());
        note.extend_from_slice(&IMAGE_PROCESSING.to_le_bytes());
        note.extend_from_slice(&4u16.to_le_bytes());
        note.extend_from_slice(&1u32.to_le_bytes());
        note.extend_from_slice(&30u32.to_le_bytes());
        note.extend_from_slice(&0u32.to_le_bytes());
        note.extend_from_slice(&4u16.to_le_bytes());
        for (i, tag) in CROP.iter().enumerate() {
            note.extend_from_slice(&tag.to_le_bytes());
            note.extend_from_slice(&16u16.to_le_bytes()); // LONG8: hostile but parsed by the TIFF reader
            note.extend_from_slice(&1u32.to_le_bytes());
            note.extend_from_slice(&(84u32 + i as u32 * 8).to_le_bytes());
        }
        note.extend_from_slice(&0u32.to_le_bytes());
        for value in [u64::MAX, 0, 2, 1] {
            note.extend_from_slice(&value.to_le_bytes());
        }
        let exif = IfdBuilder::new().with(t::MAKER_NOTE, Value::Undefined(note));
        let bytes = orf_with_exif(16, 16, 16, vec![1; 512], ByteOrder::Little, Some(exif));
        let tiff = Tiff::parse(&bytes).unwrap();
        let mn = maker_note(&bytes, &tiff).unwrap();
        assert_eq!(sub_ifd(&bytes, &mn, IMAGE_PROCESSING).unwrap().u64(CROP[0]), Some(u64::MAX));
        assert_eq!(crate::decode(&bytes).unwrap().active_area, Rect::new(0, 0, 16, 16));
    }

    #[test]
    fn word16_shifted_and_packed12() {
        let (w, h) = (16usize, 4usize);
        let px: Vec<u16> = (0..w * h).map(|i| ((i * 211) % 4096) as u16).collect();
        let words: Vec<u8> = px.iter().flat_map(|v| (v << 4).to_le_bytes()).collect();
        let bytes = orf(w as u32, h as u32, 16, words);
        assert_eq!(crate::probe(&bytes), Some(RawFormat::Orf));
        let r = crate::decode(&bytes).unwrap();
        assert_eq!((r.data.clone(), r.bits), (RawData::U16(px.clone()), 12));
        // 12-bit: MSB-first stream, every 32-bit word byte-swapped
        let mut be = Vec::new();
        for p in px.chunks(2) {
            let v = (p[0] as u32) << 12 | p[1] as u32;
            be.extend_from_slice(&[(v >> 16) as u8, (v >> 8) as u8, v as u8]);
        }
        let le: Vec<u8> = be.chunks(4).flat_map(|c| [c[3], c[2], c[1], c[0]]).collect();
        let bytes = orf(w as u32, h as u32, 12, le);
        assert_eq!(crate::decode(&bytes).unwrap().data, RawData::U16(px));
        let bytes = orf(w as u32, h as u32, 16, vec![0; 40]);
        assert!(matches!(crate::decode(&bytes), Err(RawError::Unsupported(_))));
    }
}
