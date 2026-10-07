//! Reading an `.onnx` file into a [`Net`] (feature `tract`, which brings the ONNX protobuf types).
//!
//! Only what the denoise U-Nets are made of is understood: `Conv` (3 × 3 "same" or 1 × 1, stride 1, no groups, no
//! dilation) optionally followed by `LeakyRelu`, `ConvTranspose` (2 × 2, stride 2) likewise, `MaxPool` (2 × 2, stride 2),
//! a two-input `Concat` on the channels that feeds a convolution, and a final `DepthToSpace` (2, `CRD`). Any other
//! operator or attribute is [`NetError::Unsupported`], which tells the caller to use the CPU runner instead.

use std::collections::HashMap;
use std::path::Path;

use tract_onnx::pb::{AttributeProto, NodeProto, TensorProto, tensor_shape_proto, type_proto};
use tract_onnx::prelude::Framework;

use crate::net::{Conv, ConvT2, MAX_WEIGHTS, Net, NetError, Op, TensorId};

fn unsupported<T>(why: impl Into<String>) -> Result<T, NetError> {
    Err(NetError::Unsupported(why.into()))
}

fn invalid<T>(why: impl Into<String>) -> Result<T, NetError> {
    Err(NetError::Invalid(why.into()))
}

/// Read the model at `path`.
pub fn read(path: &Path) -> Result<Net, NetError> {
    // tract's protobuf reader is a large library fed a stranger's file: a panic in it is an error here
    let proto = std::panic::catch_unwind(|| tract_onnx::onnx().proto_model_for_path(path))
        .map_err(|_| NetError::Invalid("the ONNX reader gave up on this file".into()))?
        .map_err(|e| NetError::Invalid(format!("{e:#}")))?;
    let Some(graph) = proto.graph else { return invalid("no graph") };

    let initialisers: HashMap<&str, &TensorProto> = graph.initializer.iter().map(|t| (t.name.as_str(), t)).collect();
    let inputs: Vec<_> = graph.input.iter().filter(|i| !initialisers.contains_key(i.name.as_str())).collect();
    let ([input], [output]) = (inputs.as_slice(), graph.output.as_slice()) else { return unsupported("not one input and one output") };
    let in_channels = match channel_dim(input) {
        Some(c) => c,
        None => return unsupported("the input's channel count is not written in the file"),
    };

    // how many nodes read each name (a graph output counts as a reader)
    let mut readers: HashMap<&str, usize> = HashMap::new();
    for n in &graph.node {
        for i in &n.input {
            *readers.entry(i.as_str()).or_default() += 1;
        }
    }
    *readers.entry(output.name.as_str()).or_default() += 1;

    let mut ids: HashMap<String, TensorId> = HashMap::new();
    ids.insert(input.name.clone(), 0);
    // a Concat is not a layer of its own: the convolution that reads it takes both sources
    let mut joins: HashMap<String, (TensorId, TensorId)> = HashMap::new();
    let mut ops: Vec<Op> = Vec::new();
    let mut weights_total = 0usize;

    let nodes = &graph.node;
    let mut i = 0;
    while i < nodes.len() {
        let Some(node) = nodes.get(i) else { break };
        i += 1;
        let this = ops.len() + 1;
        let lookup = |name: &str| -> Result<TensorId, NetError> {
            ids.get(name).copied().ok_or_else(|| NetError::Invalid(format!("`{}` reads `{name}`, which nothing makes", node.op_type)))
        };
        // a LeakyRelu straight after this layer, which nothing else reads, is part of it
        let mut fuse_leaky = |out: &str| -> Result<(Option<f32>, String), NetError> {
            if let Some(next) = nodes.get(i)
                && next.op_type == "LeakyRelu"
                && next.input.first().map(String::as_str) == Some(out)
                && readers.get(out).copied() == Some(1)
            {
                let alpha = attr(next, "alpha").map_or(0.01, |a| a.f);
                i += 1;
                let name = next.output.first().cloned().ok_or_else(|| NetError::Invalid("LeakyRelu without an output".into()))?;
                return Ok((Some(alpha), name));
            }
            Ok((None, out.to_string()))
        };
        match node.op_type.as_str() {
            "Concat" => {
                if attr(node, "axis").map(|a| a.i) != Some(1) {
                    return unsupported("Concat on another axis than the channels");
                }
                let [a, b] = node.input.as_slice() else { return unsupported("Concat of other than two tensors") };
                let out = node.output.first().ok_or_else(|| NetError::Invalid("Concat without an output".into()))?;
                if readers.get(out.as_str()).copied() != Some(1) {
                    return unsupported("a Concat that is read by more than one layer");
                }
                joins.insert(out.clone(), (lookup(a)?, lookup(b)?));
            }
            "Conv" => {
                let (w, b) = weights_of(node, &initialisers, &mut weights_total)?;
                let (cout, cin, kh, kw) = four(&w.0)?;
                let b = if b.is_empty() { vec![0.0; cout] } else { b };
                if kh != kw || !(kh == 1 || kh == 3) {
                    return unsupported(format!("a {kh} × {kw} convolution"));
                }
                if ints_or(node, "strides", &[1, 1]) != [1, 1]
                    || ints_or(node, "dilations", &[1, 1]) != [1, 1]
                    || attr(node, "group").is_some_and(|a| a.i != 1)
                {
                    return unsupported("a convolution with strides, dilation or groups");
                }
                let pad = (kh as i64 - 1) / 2;
                if ints_or(node, "pads", &[pad; 4]) != [pad; 4] || attr(node, "auto_pad").is_some_and(|a| !a.s.is_empty() && a.s != b"NOTSET") {
                    return unsupported("a convolution that is not padded to keep its size");
                }
                let src_name = node.input.first().ok_or_else(|| NetError::Invalid("Conv without an input".into()))?;
                let (src, src2) = match joins.remove(src_name) {
                    Some((a, b)) => (a, Some(b)),
                    None => (lookup(src_name)?, None),
                };
                let out = node.output.first().ok_or_else(|| NetError::Invalid("Conv without an output".into()))?;
                let (leaky, out_name) = fuse_leaky(out)?;
                ids.insert(out_name, this);
                ops.push(Op::Conv(Conv { src, src2, cin, cout, k: kh, weight: w.1, bias: b, leaky }));
            }
            "ConvTranspose" => {
                let (w, b) = weights_of(node, &initialisers, &mut weights_total)?;
                let (cin, cout, kh, kw) = four(&w.0)?;
                let b = if b.is_empty() { vec![0.0; cout] } else { b };
                if (kh, kw) != (2, 2)
                    || ints_or(node, "strides", &[1, 1]) != [2, 2]
                    || ints_or(node, "pads", &[0; 4]) != [0; 4]
                    || ints_or(node, "dilations", &[1, 1]) != [1, 1]
                    || attr(node, "group").is_some_and(|a| a.i != 1)
                    || attr(node, "output_padding").is_some_and(|a| a.ints.iter().any(|&v| v != 0))
                {
                    return unsupported("a transposed convolution other than 2 × 2 with stride 2");
                }
                let src = lookup(node.input.first().map_or("", String::as_str))?;
                let out = node.output.first().ok_or_else(|| NetError::Invalid("ConvTranspose without an output".into()))?;
                let (leaky, out_name) = fuse_leaky(out)?;
                ids.insert(out_name, this);
                ops.push(Op::ConvT2(ConvT2 { src, cin, cout, weight: w.1, bias: b, leaky }));
            }
            "MaxPool" => {
                if ints_or(node, "kernel_shape", &[]) != [2, 2]
                    || ints_or(node, "strides", &[1, 1]) != [2, 2]
                    || ints_or(node, "pads", &[0; 4]) != [0; 4]
                    || ints_or(node, "dilations", &[1, 1]) != [1, 1]
                    || attr(node, "ceil_mode").is_some_and(|a| a.i != 0)
                {
                    return unsupported("a max-pool other than 2 × 2 with stride 2");
                }
                let src = lookup(node.input.first().map_or("", String::as_str))?;
                let out = node.output.first().ok_or_else(|| NetError::Invalid("MaxPool without an output".into()))?;
                ids.insert(out.clone(), this);
                ops.push(Op::MaxPool2 { src });
            }
            "DepthToSpace" => {
                if attr(node, "blocksize").map(|a| a.i) != Some(2) || attr(node, "mode").map(|a| a.s.as_slice()) != Some(&b"CRD"[..]) {
                    return unsupported("depth-to-space other than block size 2 in CRD order");
                }
                let src = lookup(node.input.first().map_or("", String::as_str))?;
                let out = node.output.first().ok_or_else(|| NetError::Invalid("DepthToSpace without an output".into()))?;
                ids.insert(out.clone(), this);
                ops.push(Op::DepthToSpace2 { src });
            }
            other => return unsupported(format!("the `{other}` operator")),
        }
    }
    if !joins.is_empty() {
        return unsupported("a Concat whose result no convolution reads");
    }
    if ids.get(output.name.as_str()) != Some(&ops.len()) {
        return unsupported("the output is not the last layer");
    }
    Net::new(in_channels, ops)
}

fn attr<'a>(n: &'a NodeProto, name: &str) -> Option<&'a AttributeProto> {
    n.attribute.iter().find(|a| a.name == name)
}

/// The integer list attribute `name`, or `default` when the node does not have it.
fn ints_or(n: &NodeProto, name: &str, default: &[i64]) -> Vec<i64> {
    attr(n, name).map_or_else(|| default.to_vec(), |a| a.ints.clone())
}

fn four(dims: &[i64]) -> Result<(usize, usize, usize, usize), NetError> {
    match dims {
        [a, b, c, d] if dims.iter().all(|&v| v > 0 && v <= 1 << 20) => Ok((*a as usize, *b as usize, *c as usize, *d as usize)),
        _ => invalid("a weight tensor that is not four-dimensional"),
    }
}

type Weights = ((Vec<i64>, Vec<f32>), Vec<f32>);

/// The weight and bias initialisers of a convolution node.
fn weights_of(node: &NodeProto, initialisers: &HashMap<&str, &TensorProto>, total: &mut usize) -> Result<Weights, NetError> {
    let mut get = |idx: usize| -> Result<(Vec<i64>, Vec<f32>), NetError> {
        let name = node.input.get(idx).ok_or_else(|| NetError::Invalid(format!("`{}` without its weights", node.op_type)))?;
        let t = initialisers.get(name.as_str()).ok_or_else(|| NetError::Unsupported(format!("weights that are computed (`{name}`)")))?;
        let values = floats(t)?;
        *total = total.saturating_add(values.len());
        if *total > MAX_WEIGHTS {
            return invalid("the network has too many weights");
        }
        Ok((t.dims.clone(), values))
    };
    let w = get(1)?;
    let b = if node.input.len() > 2 { get(2)?.1 } else { Vec::new() };
    Ok((w, b))
}

fn floats(t: &TensorProto) -> Result<Vec<f32>, NetError> {
    if t.data_type != 1 {
        return unsupported("weights that are not 32-bit floats");
    }
    let want: usize = t
        .dims
        .iter()
        .try_fold(1usize, |n, &d| usize::try_from(d).ok().and_then(|d| n.checked_mul(d)))
        .ok_or_else(|| NetError::Invalid("a tensor with a negative or huge size".into()))?;
    if want > MAX_WEIGHTS {
        return invalid("a tensor with too many numbers");
    }
    let values: Vec<f32> =
        if !t.raw_data.is_empty() { t.raw_data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect() } else { t.float_data.clone() };
    if values.len() != want {
        return invalid(format!("tensor `{}` has {} numbers, not the {want} its shape says", t.name, values.len()));
    }
    Ok(values)
}

/// The channel count in an input's declared shape `[batch, channels, …]`.
fn channel_dim(v: &tract_onnx::pb::ValueInfoProto) -> Option<usize> {
    let type_proto::Value::TensorType(t) = v.r#type.as_ref()?.value.as_ref()?;
    let tensor_shape_proto::dimension::Value::DimValue(c) = t.shape.as_ref()?.dim.get(1)?.value.as_ref()? else { return None };
    usize::try_from(*c).ok().filter(|&c| c > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::unet_onnx;
    use tract_onnx::prelude::*;

    fn temp_model(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("lc-denoise-onnx-{name}-{}.onnx", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        p
    }

    fn input(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i * 7919) % 1013) as f32 / 1013.0).collect()
    }

    #[test]
    fn a_u_net_is_read_with_its_layers_fused_and_runs_like_tract() {
        let (tile, depth) = (32usize, 2usize);
        let p = temp_model("unet", &unet_onnx(tile as u64, 8, depth, 7));
        let net = read(&p).unwrap();
        // per level two convolutions; a pool per level but the last; going up a transposed convolution and two convolutions
        // per level (the join is part of the first); then the 1 × 1 convolution and depth-to-space
        assert_eq!(net.ops().len(), 2 * (depth + 1) + depth + depth * 3 + 2);
        assert!(matches!(net.ops().last(), Some(crate::net::Op::DepthToSpace2 { .. })));
        assert!(net.ops().iter().any(|o| matches!(o, crate::net::Op::Conv(c) if c.src2.is_some())), "the joins became two-source convolutions");
        assert!(
            net.ops().iter().all(|o| !matches!(o, crate::net::Op::Conv(c) if c.leaky.is_none() && c.k == 3)),
            "every 3 × 3 convolution has its activation"
        );
        assert_eq!((net.in_channels, net.out_channels(), net.depth()), (4, 3, 2));
        assert!(net.fits_tile(tile) && !net.fits_tile(30));

        // the reference interpreter and tract agree on the same network
        let x = input(4 * tile * tile);
        let ours = crate::reference::run(&net, tile, &x).unwrap();
        let model = tract_onnx::onnx()
            .model_for_path(&p)
            .unwrap()
            .with_input_fact(0, f32::fact([1, 4, tile, tile]).into())
            .unwrap()
            .into_optimized()
            .unwrap()
            .into_runnable()
            .unwrap();
        let tensor: Tensor = tract_ndarray::Array4::from_shape_vec((1, 4, tile, tile), x).unwrap().into();
        let theirs = model.run(tvec!(tensor.into())).unwrap();
        let theirs: Vec<f32> = theirs[0].to_plain_array_view::<f32>().unwrap().iter().copied().collect();
        assert_eq!(ours.len(), theirs.len());
        let scale = theirs.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = ours.iter().zip(&theirs).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst <= 1e-4 * scale, "worst difference {worst} against a scale of {scale}");
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn what_the_description_cannot_say_is_unsupported_and_garbage_is_an_error() {
        // not a model
        let p = temp_model("junk", b"this is not an onnx file");
        assert!(matches!(read(&p), Err(NetError::Invalid(_)) | Err(NetError::Unsupported(_))));
        let _ = std::fs::remove_file(p);
        // a missing file
        assert!(read(Path::new("/does/not/exist.onnx")).is_err());
    }
}
