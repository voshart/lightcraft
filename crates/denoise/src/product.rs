//! The denoised picture as cached data, and the blend the Amount slider controls.
//!
//! A product is camera RGB at the sensor's resolution (what the model gave, before the raw's own opcodes and
//! crop), kept as half floats in strips of rows, each strip byte-shuffled and deflated with its own checksum.
//! It is derived data: any problem reading it (missing, truncated, corrupt, made for another photo or model) is
//! just a cache miss, never a panic, and the picture can be made again.
//!
//! File: `LCDN`, version, header length, header JSON (`width`, `height`, `strip`, `key`), then for each strip its
//! compressed length and CRC-32, then the strips' bytes. `key` says what the product was made from (the photo's
//! content, the model, this code's version, the settings it depends on); a file with another key is stale.

use std::io::{Read, Write};
use std::path::Path;

use lightcraft_raster::Rgb32f;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 4] = b"LCDN";
const VERSION: u8 = 1;
const MAX_HEADER: usize = 4096;
const MAX_SIDE: usize = 1 << 17;
/// Most pixels a product may have (the denoiser's own limit, [`crate::run::MAX_PIXELS`]).
const MAX_PIXELS: usize = crate::run::MAX_PIXELS;
/// Rows per strip: about a megapixel of three planes at 24 MP widths.
const STRIP_ROWS: usize = 128;
/// Bumped when the meaning of a product changes (included in keys by the caller).
pub const ALGORITHM: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum ProductError {
    #[error("the denoised picture could not be read or written: {0}")]
    Io(String),
    #[error("not a usable denoised picture: {0}")]
    Format(String),
    /// A product made from something else (another photo state, model or version).
    #[error("the denoised picture is out of date")]
    Stale,
}

impl From<std::io::Error> for ProductError {
    fn from(e: std::io::Error) -> Self {
        ProductError::Io(e.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Header {
    pub width: usize,
    pub height: usize,
    pub strip: usize,
    pub key: String,
}

/// `f32` as IEEE half-float bits, rounded to nearest even; beyond the range it saturates, NaN becomes 0.
pub fn to_f16(v: f32) -> u16 {
    if !v.is_finite() {
        return if v.is_nan() {
            0
        } else if v > 0.0 {
            0x7bff
        } else {
            0xfbff
        };
    }
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp >= 31 {
        return sign | 0x7bff;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let man = man | 0x80_0000;
        let shift = (14 - exp) as u32;
        let (half, rem, halfway) = (man >> shift, man & ((1u32 << shift) - 1), 1u32 << (shift - 1));
        let mut h = half as u16;
        if rem > halfway || (rem == halfway && h & 1 == 1) {
            h += 1;
        }
        return sign | h;
    }
    let mut h = (((exp as u32) << 10) | (man >> 13)) as u16;
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    if h >= 0x7c00 {
        h = 0x7bff;
    }
    sign | h
}

/// Half-float bits as `f32`.
pub fn from_f16(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let man = f32::from(h & 0x3ff);
    match exp {
        0 => sign * man * 2f32.powi(-24),
        31 => sign * 65504.0,
        _ => sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    }
}

fn crc(bytes: &[u8]) -> u32 {
    // CRC-32 (IEEE), bitwise: strips are small and this runs once per strip
    let mut table = [0u32; 256];
    for (i, t) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *t = c;
    }
    !bytes.iter().fold(!0u32, |c, &b| table[((c ^ u32::from(b)) & 0xff) as usize] ^ (c >> 8))
}

/// One strip as bytes: for each plane (R, G, B) the low bytes of its half floats, then the high bytes.
fn encode_strip(img: &Rgb32f, y0: usize, y1: usize) -> Vec<u8> {
    let n = (y1 - y0) * img.width;
    let mut raw = vec![0u8; 6 * n];
    let rows = img.data.get(y0 * img.width..y1 * img.width).unwrap_or(&[]);
    for (i, px) in rows.iter().enumerate() {
        for (c, &v) in px.iter().enumerate() {
            let h = to_f16(v);
            if let Some(lo) = raw.get_mut(2 * c * n + i) {
                *lo = h as u8;
            }
            if let Some(hi) = raw.get_mut((2 * c + 1) * n + i) {
                *hi = (h >> 8) as u8;
            }
        }
    }
    miniz_oxide::deflate::compress_to_vec(&raw, 3)
}

/// Write `img` to `path` (through a temporary file next to it, so a reader never sees half a product).
pub fn write(path: &Path, img: &Rgb32f, key: &str) -> Result<(), ProductError> {
    if img.width == 0 || img.height == 0 || img.width > MAX_SIDE || img.height > MAX_SIDE || img.data.len() != img.width * img.height {
        return Err(ProductError::Format("the picture has no usable size".into()));
    }
    let strips: Vec<(usize, usize)> = (0..img.height).step_by(STRIP_ROWS).map(|y| (y, (y + STRIP_ROWS).min(img.height))).collect();
    let packed: Vec<Vec<u8>> = strips.par_iter().map(|&(a, b)| encode_strip(img, a, b)).collect();
    let header = serde_json::to_vec(&Header { width: img.width, height: img.height, strip: STRIP_ROWS, key: key.to_string() })
        .map_err(|e| ProductError::Format(e.to_string()))?;
    let tmp = path.with_extension("tmp");
    let result = (|| -> Result<(), ProductError> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        f.write_all(MAGIC)?;
        f.write_all(&[VERSION, 0, 0, 0])?;
        f.write_all(&(header.len() as u32).to_le_bytes())?;
        f.write_all(&header)?;
        for p in &packed {
            f.write_all(&(p.len() as u32).to_le_bytes())?;
            f.write_all(&crc(p).to_le_bytes())?;
        }
        for p in &packed {
            f.write_all(p)?;
        }
        f.flush()?;
        drop(f);
        std::fs::rename(&tmp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// The parsed start of a product file: header, each strip's (compressed length, CRC-32), and the file length the
/// start says the whole file has.
struct Start {
    header: Header,
    table: Vec<(usize, u32)>,
    expected_len: u64,
}

fn read_start(r: &mut impl Read) -> Result<Start, ProductError> {
    let mut fixed = [0u8; 12];
    r.read_exact(&mut fixed)?;
    if fixed.get(0..4) != Some(&MAGIC[..]) || fixed.get(4) != Some(&VERSION) {
        return Err(ProductError::Format("not a denoised picture of this version".into()));
    }
    let len = fixed.get(8..12).and_then(|b| <[u8; 4]>::try_from(b).ok()).map(u32::from_le_bytes).unwrap_or(u32::MAX) as usize;
    if len > MAX_HEADER {
        return Err(ProductError::Format("the header is too large".into()));
    }
    let mut hb = vec![0u8; len];
    r.read_exact(&mut hb)?;
    let h: Header = serde_json::from_slice(&hb).map_err(|e| ProductError::Format(e.to_string()))?;
    let pixels = h.width.saturating_mul(h.height);
    if h.width == 0 || h.height == 0 || h.width > MAX_SIDE || h.height > MAX_SIDE || pixels > MAX_PIXELS || h.strip != STRIP_ROWS {
        return Err(ProductError::Format("the size is not usable".into()));
    }
    let n = h.height.div_ceil(h.strip);
    let mut table = Vec::with_capacity(n);
    let mut total = 12 + len as u64 + 8 * n as u64;
    for _ in 0..n {
        let mut e = [0u8; 8];
        r.read_exact(&mut e)?;
        let l = u32::from_le_bytes([e[0], e[1], e[2], e[3]]) as usize;
        let c = u32::from_le_bytes([e[4], e[5], e[6], e[7]]);
        // a strip never compresses to much more than its raw size
        if l > 12 * h.strip * h.width + 4096 {
            return Err(ProductError::Format("a strip is larger than it can be".into()));
        }
        total += l as u64;
        table.push((l, c));
    }
    Ok(Start { header: h, table, expected_len: total })
}

/// The header of the product at `path`, when it is one.
pub fn read_header_at(path: &Path) -> Result<Header, ProductError> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    Ok(read_start(&mut f)?.header)
}

/// Whether `path` holds a complete-looking product made from `key` (the header and the file's length are checked;
/// strips are checked when read).
pub fn is_current(path: &Path, key: &str) -> bool {
    let check = || -> Result<bool, ProductError> {
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let s = read_start(&mut f)?;
        Ok(s.header.key == key && std::fs::metadata(path)?.len() == s.expected_len)
    };
    check().unwrap_or(false)
}

/// Read the product at `path`, made from `key`, with every `factor × factor` block of pixels averaged into one
/// (`factor` 1 = full size). The result is camera RGB.
pub fn read(path: &Path, key: &str, factor: usize) -> Result<Rgb32f, ProductError> {
    let factor = factor.max(1);
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let Start { header: h, table, expected_len } = read_start(&mut f)?;
    if h.key != key {
        return Err(ProductError::Stale);
    }
    if std::fs::metadata(path)?.len() != expected_len {
        return Err(ProductError::Format("the file is not the length its header says".into()));
    }
    let (w, ow, oh) = (h.width, h.width.div_ceil(factor), h.height.div_ceil(factor));
    let mut out = Rgb32f { width: ow, height: oh, data: vec![[0.0; 3]; ow * oh] };
    // sums of the block row being built, and how many input rows went into it
    let mut sum = vec![[0f32; 3]; ow];
    let mut rows_in = 0usize;
    let mut oy = 0usize;
    for (s, &(len, expect_crc)) in table.iter().enumerate() {
        let mut packed = vec![0u8; len];
        f.read_exact(&mut packed)?;
        if crc(&packed) != expect_crc {
            return Err(ProductError::Format(format!("strip {s} is damaged")));
        }
        let y0 = s * h.strip;
        let rows = (h.height - y0).min(h.strip);
        let n = rows * w;
        let raw =
            miniz_oxide::inflate::decompress_to_vec_with_limit(&packed, 6 * n).map_err(|e| ProductError::Format(format!("strip {s}: {e:?}")))?;
        if raw.len() != 6 * n {
            return Err(ProductError::Format(format!("strip {s} has the wrong size")));
        }
        for r in 0..rows {
            for x in 0..w {
                let i = r * w + x;
                let px = &mut sum[x / factor];
                for (c, v) in px.iter_mut().enumerate() {
                    let bits = u16::from(raw[2 * c * n + i]) | (u16::from(raw[(2 * c + 1) * n + i]) << 8);
                    *v += from_f16(bits);
                }
            }
            rows_in += 1;
            let last_row = y0 + r + 1 == h.height;
            if rows_in == factor || last_row {
                for (ox, px) in sum.iter_mut().enumerate() {
                    // the block's size at the right and bottom edges may be smaller
                    let bw = factor.min(w - ox * factor);
                    let k = 1.0 / (rows_in * bw) as f32;
                    if let Some(o) = out.data.get_mut(oy * ow + ox) {
                        *o = px.map(|v| v * k);
                    }
                    *px = [0.0; 3];
                }
                rows_in = 0;
                oy += 1;
            }
        }
    }
    Ok(out)
}

/// `base + (denoised − base) · amount` for every pixel (`amount` 0 to 1), in place in `base`. The two pictures must
/// be the same size; if they are not, `base` is left as it was and `false` is returned.
pub fn blend(base: &mut Rgb32f, denoised: &Rgb32f, amount: f32) -> bool {
    if base.width != denoised.width || base.height != denoised.height {
        return false;
    }
    let a = if amount.is_finite() { amount.clamp(0.0, 1.0) } else { 0.0 };
    base.data.par_iter_mut().zip(denoised.data.par_iter()).for_each(|(b, d)| {
        for c in 0..3 {
            b[c] += (d[c] - b[c]) * a;
        }
    });
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("lc-denoise-{name}-{}.lcdn", std::process::id()))
    }

    fn picture(w: usize, h: usize) -> Rgb32f {
        Rgb32f {
            width: w,
            height: h,
            data: (0..w * h).map(|i| [(i % w) as f32 / w as f32, (i / w) as f32 / h as f32, 0.001 + 0.5 * ((i % 7) as f32 / 7.0)]).collect(),
        }
    }

    #[test]
    fn half_floats_round_trip_closely() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 0.333_333, 1e-4, 6.1e-5, 3e-6, 123.456, 65000.0, -0.002] {
            let back = from_f16(to_f16(v));
            assert!((back - v).abs() <= v.abs() * 0.001 + 1e-7, "{v} -> {back}");
        }
        assert_eq!(from_f16(to_f16(1e10)), 65504.0);
        assert_eq!(to_f16(f32::NAN), 0);
        assert_eq!(from_f16(to_f16(f32::NEG_INFINITY)), -65504.0);
    }

    #[test]
    fn a_product_survives_a_round_trip_at_full_and_reduced_size() {
        let (w, h) = (301, 517);
        let img = picture(w, h);
        let p = temp("round");
        write(&p, &img, "k1").unwrap();
        assert!(is_current(&p, "k1") && !is_current(&p, "k2"));
        assert_eq!(read_header_at(&p).unwrap().width, w);
        let full = read(&p, "k1", 1).unwrap();
        assert_eq!((full.width, full.height), (w, h));
        for (a, b) in full.data.iter().zip(&img.data) {
            for c in 0..3 {
                assert!((a[c] - b[c]).abs() < 0.002, "{a:?} vs {b:?}");
            }
        }
        // averaged into blocks of 4: the sizes round up, and the mean of the picture is kept
        let small = read(&p, "k1", 4).unwrap();
        assert_eq!((small.width, small.height), (w.div_ceil(4), h.div_ceil(4)));
        let mean = |i: &Rgb32f| i.data.iter().map(|p| f64::from(p[1])).sum::<f64>() / i.data.len() as f64;
        assert!((mean(&small) - mean(&img)).abs() < 0.01);
        // a factor that does not divide the strip height is the same average
        let odd = read(&p, "k1", 5).unwrap();
        assert_eq!((odd.width, odd.height), (w.div_ceil(5), h.div_ceil(5)));
        assert!((mean(&odd) - mean(&img)).abs() < 0.01);
        assert!(matches!(read(&p, "other", 1), Err(ProductError::Stale)));
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn a_smooth_picture_compresses_well() {
        let img = Rgb32f {
            width: 512,
            height: 512,
            data: (0..512 * 512)
                .map(|i| {
                    let v = 0.2 + 0.0001 * (i % 512) as f32;
                    [v, v * 0.8, v * 0.6]
                })
                .collect(),
        };
        let p = temp("size");
        write(&p, &img, "k").unwrap();
        let bytes = std::fs::metadata(&p).unwrap().len() as usize;
        assert!(bytes < 512 * 512 * 6 / 2, "{bytes} bytes for a smooth picture");
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn damaged_files_are_errors_not_panics() {
        let img = picture(100, 300);
        let p = temp("damage");
        write(&p, &img, "k").unwrap();
        let good = std::fs::read(&p).unwrap();
        // truncated at every length
        for cut in (0..good.len()).step_by(37) {
            std::fs::write(&p, &good[..cut]).unwrap();
            assert!(read(&p, "k", 1).is_err(), "cut at {cut}");
            assert!(!is_current(&p, "k"));
        }
        // a flipped byte anywhere is caught by a checksum, the header parse, or the key
        for at in (0..good.len()).step_by(53) {
            let mut bad = good.clone();
            bad[at] ^= 0xff;
            std::fs::write(&p, &bad).unwrap();
            let _ = read(&p, "k", 1); // must not panic; most are errors
        }
        // a header that claims an absurd size
        let mut huge = b"LCDN\x01\0\0\0".to_vec();
        let hdr = br#"{"width":4000000000,"height":4000000000,"strip":128,"key":"k"}"#;
        huge.extend((hdr.len() as u32).to_le_bytes());
        huge.extend(hdr);
        std::fs::write(&p, &huge).unwrap();
        assert!(matches!(read(&p, "k", 1), Err(ProductError::Format(_))));
        std::fs::write(&p, b"").unwrap();
        assert!(read(&p, "k", 1).is_err());
        assert!(read(&temp("missing"), "k", 1).is_err());
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn writing_rejects_unusable_pictures() {
        let p = temp("bad");
        assert!(write(&p, &Rgb32f { width: 0, height: 0, data: vec![] }, "k").is_err());
        assert!(write(&p, &Rgb32f { width: 3, height: 3, data: vec![[0.0; 3]; 8] }, "k").is_err());
        assert!(!p.exists());
    }

    #[test]
    fn blend_mixes_by_amount_and_rejects_other_sizes() {
        let mut base = Rgb32f { width: 2, height: 1, data: vec![[0.0; 3], [1.0; 3]] };
        let dn = Rgb32f { width: 2, height: 1, data: vec![[1.0; 3], [0.0; 3]] };
        assert!(blend(&mut base, &dn, 0.25));
        assert_eq!(base.data, vec![[0.25; 3], [0.75; 3]]);
        assert!(blend(&mut base, &dn, f32::NAN));
        assert_eq!(base.data, vec![[0.25; 3], [0.75; 3]]);
        assert!(!blend(&mut base, &Rgb32f { width: 1, height: 1, data: vec![[0.0; 3]] }, 1.0));
        assert!(blend(&mut base, &dn, 1.0));
        assert_eq!(base.data, dn.data);
    }

    #[test]
    fn crc_matches_the_standard_check_value() {
        assert_eq!(crc(b"123456789"), 0xcbf4_3926);
    }
}
