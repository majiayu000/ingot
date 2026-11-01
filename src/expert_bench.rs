//! Prefill routed-expert GEMM timing on real weights; run explicitly.
use crate::metal::Gpu;
use objc2_metal::MTLCommandBuffer;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

// Production tile layout from qwen_prefill::resident_ffn_packed.
fn plan(tokens: usize, experts: usize, top: usize, skew: f64) -> (Vec<u32>, Vec<u32>) {
    let mut state = 0x9e3779b97f4a7c15;
    let mut groups = vec![Vec::new(); experts];
    for token in 0..tokens {
        let mut chosen = Vec::with_capacity(top);
        while chosen.len() < top {
            let e = ((lcg(&mut state).powf(skew)) * experts as f64) as usize % experts;
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        for (k, e) in chosen.into_iter().enumerate() {
            groups[e].push((token * top + k) as u32);
        }
    }
    let (mut pairs, mut tiles) = (Vec::new(), Vec::new());
    for (expert, assignments) in groups.iter().enumerate() {
        let first = pairs.len();
        pairs.extend_from_slice(assignments);
        for offset in (0..assignments.len()).step_by(32) {
            let nt = (assignments.len() - offset).min(32);
            tiles.extend([
                expert as u32,
                (first + offset) as u32,
                nt as u32,
                u32::from(nt >= 2),
            ]);
        }
    }
    (pairs, tiles)
}

#[test]
#[ignore = "requires QWEN_MODEL; prints GPU time and TFLOPS, dumps outputs for variant diffs"]
fn routed_expert_gemm() {
    let gpu = Gpu::new().unwrap();
    let path = std::env::var("QWEN_MODEL").unwrap();
    let layer: usize = std::env::var("BENCH_LAYER").map_or(1, |v| v.parse().unwrap());
    let tokens: usize = std::env::var("BENCH_TOKENS").map_or(4095, |v| v.parse().unwrap());
    let dump = std::env::var_os("BENCH_DUMP").filter(|d| !d.is_empty());
    let mut g = crate::gguf::Gguf::open(std::path::Path::new(&path)).unwrap();
    let (experts, top) = (512usize, 10usize);
    let mut load = |kind: &str| {
        let t = g
            .tensor(&format!("blk.{layer}.ffn_{kind}_exps.weight"))
            .unwrap()
            .clone();
        (gpu.load_weight(&mut g, &t).unwrap(), t)
    };
    let (gate, up, down) = (load("gate"), load("up"), load("down"));
    // Variants alternate within each round so GPU clock and load drift cancel.
    type Encode<'a> = &'a dyn Fn(&crate::metal::Command);
    let time = |encodes: &[Encode]| -> Vec<f64> {
        let mut samples = vec![Vec::new(); encodes.len()];
        for round in 0..16 {
            for (encode, samples) in encodes.iter().zip(&mut samples) {
                let cmd = gpu.command().unwrap();
                for _ in 0..4 {
                    encode(&cmd);
                }
                cmd.commit();
                cmd.waitUntilCompleted();
                assert!(cmd.error().is_none(), "{:?}", cmd.error());
                if round > 1 {
                    samples.push((cmd.GPUEndTime() - cmd.GPUStartTime()) / 4.0);
                }
            }
        }
        samples
            .into_iter()
            .map(|mut s| {
                s.sort_by(f64::total_cmp);
                s[s.len() / 2]
            })
            .collect()
    };
    let random = |n: usize| {
        let mut state = 7u64;
        let x: Vec<f32> = (0..n).map(|_| (lcg(&mut state) - 0.5) as f32).collect();
        gpu.floats(&x).unwrap()
    };
    let words = |v: &[u32]| {
        gpu.bytes(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let save = |y: &crate::metal::Buffer, name: String| -> Vec<f32> {
        let out = gpu.read_f32(y).unwrap();
        if let Some(dir) = &dump {
            let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(std::path::Path::new(dir).join(name), bytes).unwrap();
        }
        out
    };
    for skew in [1.0f64, 2.0] {
        let (pairs, tiles) = plan(tokens, experts, top, skew);
        let (pb, tb) = (words(&pairs), words(&tiles));
        let blocks = tiles.len() / 4;
        for (stage, (w, t)) in [("gate", &gate), ("down", &down)] {
            let packed = stage == "down";
            let (cols, rows) = (t.dims[0], t.dims[1]);
            let params = [
                cols as u32,
                rows as u32,
                t.stride as u32,
                t.kind,
                top as u32,
            ];
            let x = random(if packed { tokens * top } else { tokens } * cols);
            let y = gpu.output(rows * tokens * top).unwrap();
            let name = if packed {
                "quant_expert_packed"
            } else {
                "quant_expert_gather"
            };
            let s = time(&[&|cmd: &crate::metal::Command| {
                gpu.op(cmd, name, &[w, &x, &tb, &y, &pb], &params, rows, blocks)
                    .unwrap();
            }])[0];
            println!(
                "EXPERT_GEMM layer={layer} {stage} kind={} skew={skew} median_us={:.1} tflops={:.2}",
                t.kind,
                s * 1e6,
                2.0 * (tokens * top * rows * cols) as f64 / s / 1e12
            );
            save(&y, format!("{layer}-{stage}-{skew}.f32"));
        }

        let t = &gate.1;
        let (cols, rows) = (t.dims[0], t.dims[1]);
        let params = [
            cols as u32,
            rows as u32,
            t.stride as u32,
            t.kind,
            top as u32,
        ];
        let x = random(tokens * cols);
        let (gy, uy, fused) = (
            gpu.output(rows * tokens * top).unwrap(),
            gpu.output(rows * tokens * top).unwrap(),
            gpu.output(rows * tokens * top).unwrap(),
        );
        let act = std::cell::RefCell::new(None);
        let separate = |cmd: &crate::metal::Command| {
            for (w, y) in [(&gate.0, &gy), (&up.0, &uy)] {
                gpu.op(
                    cmd,
                    "quant_expert_gather",
                    &[w, &x, &tb, y, &pb],
                    &params,
                    rows,
                    blocks,
                )
                .unwrap();
            }
            *act.borrow_mut() = Some(gpu.activation(cmd, &gy, &uy, rows * tokens * top).unwrap());
        };
        let combined = |cmd: &crate::metal::Command| {
            gpu.op(
                cmd,
                "quant_expert_gate_up",
                &[&gate.0, &up.0, &x, &tb, &fused, &pb],
                &params,
                rows,
                blocks,
            )
            .unwrap();
        };
        let flops = 2.0 * 2.0 * (tokens * top * rows * cols) as f64;
        let times = time(&[&separate, &combined]);
        for (label, s) in ["separate", "fused"].into_iter().zip(times) {
            println!(
                "EXPERT_GEMM layer={layer} gate_up/{label} kind={} skew={skew} median_us={:.1} tflops={:.2}",
                t.kind,
                s * 1e6,
                flops / s / 1e12
            );
        }
        let act = act.borrow().clone().unwrap();
        save(&act, format!("{layer}-gate_up-{skew}.f32"));
        let (a, b) = (gpu.read_f32(&act).unwrap(), gpu.read_f32(&fused).unwrap());
        assert!(
            a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()),
            "fused gate/up differs from gather, gather and swiglu"
        );
    }
}

#[test]
#[ignore = "requires QWEN_MODEL; count-one decode projection bandwidth on real tensors"]
fn decode_matvec_bandwidth() {
    let gpu = Gpu::new().unwrap();
    let path = std::env::var("QWEN_MODEL").unwrap();
    let mut g = crate::gguf::Gguf::open(std::path::Path::new(&path)).unwrap();
    let time = |encode: &dyn Fn(&crate::metal::Command)| -> f64 {
        let mut samples = Vec::new();
        for round in 0..14 {
            let cmd = gpu.command().unwrap();
            for _ in 0..8 {
                encode(&cmd);
            }
            cmd.commit();
            cmd.waitUntilCompleted();
            assert!(cmd.error().is_none(), "{:?}", cmd.error());
            if round > 1 {
                samples.push((cmd.GPUEndTime() - cmd.GPUStartTime()) / 8.0);
            }
        }
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    };
    // Cycle each (tensor, format, shape) family across layers so reads stream from DRAM.
    let mut families: Vec<(String, Vec<crate::gguf::Tensor>)> = Vec::new();
    for layer in 0..48 {
        for suffix in [
            "attn_qkv",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_gate",
            "attn_output",
            "ssm_out",
            "ssm_alpha",
            "ffn_gate_shexp",
            "ffn_up_shexp",
            "ffn_down_shexp",
            "ffn_gate_inp",
            "indexer.q_proj",
            "hc_attn_down",
        ] {
            let Ok(t) = g.tensor(&format!("blk.{layer}.{suffix}.weight")).cloned() else {
                continue;
            };
            let key = format!("{suffix} kind={} {}x{}", t.kind, t.dims[0], t.dims[1]);
            match families.iter_mut().find(|(k, _)| *k == key) {
                Some((_, f)) => f.push(t),
                None => families.push((key, vec![t])),
            }
        }
    }
    for (key, tensors) in families {
        let family: Vec<_> = tensors
            .iter()
            .map(|t| gpu.load_weight(&mut g, t).unwrap())
            .collect();
        let t = &tensors[0];
        let (cols, rows) = (t.dims[0], t.dims[1]);
        let x = gpu.floats(&vec![0.01f32; cols]).unwrap();
        let reps = (256usize << 20).div_ceil(t.bytes * family.len()).max(1);
        let mut samples = Vec::new();
        for round in 0..8 {
            let cmd = gpu.command().unwrap();
            for _ in 0..reps {
                for w in &family {
                    gpu.matvec(&cmd, w, cols, rows, t.stride, t.kind, &x)
                        .unwrap();
                }
            }
            cmd.commit();
            cmd.waitUntilCompleted();
            assert!(cmd.error().is_none(), "{:?}", cmd.error());
            if round > 1 {
                samples
                    .push((cmd.GPUEndTime() - cmd.GPUStartTime()) / (reps * family.len()) as f64);
            }
        }
        samples.sort_by(f64::total_cmp);
        let s = samples[samples.len() / 2];
        println!(
            "STREAM {key} layers={} bytes/token={} us={:.1} GBps={:.0}",
            family.len(),
            t.bytes * family.len(),
            s * 1e6,
            t.bytes as f64 / s / 1e9
        );
    }
    // BF16 HC down+inject and up+collapse exactly as qwen_hc::hc_project encodes them.
    let (width, hc, lr) = (2560usize, 4usize, 320usize);
    let get = |g: &mut crate::gguf::Gguf, n: &str| {
        let t = g.tensor(n).unwrap().clone();
        (gpu.load_weight(g, &t).unwrap(), t)
    };
    let (down, dt) = get(&mut g, "blk.2.hc_attn_down.weight");
    let (inject, it) = get(&mut g, "blk.2.hc_attn_inject.weight");
    let (up, ut) = get(&mut g, "blk.2.hc_attn_up.weight");
    let norm = gpu.floats(&vec![0.01f32; width * hc]).unwrap();
    let (lo, gates, out) = (
        gpu.output(lr).unwrap(),
        gpu.output(hc).unwrap(),
        gpu.output(width).unwrap(),
    );
    let s = time(&|cmd| {
        gpu.op(
            cmd,
            "hc_down_inject",
            &[&down, &inject, &norm, &lo, &gates],
            &[
                (width * hc) as u32,
                lr as u32,
                dt.stride as u32,
                it.stride as u32,
                hc as u32,
                1,
                (1.0f32 / hc as f32).to_bits(),
            ],
            lr + hc,
            1,
        )
        .unwrap();
    });
    println!(
        "MATVEC hc_down_inject bytes={} us={:.1} GBps={:.0}",
        dt.bytes + it.bytes,
        s * 1e6,
        (dt.bytes + it.bytes) as f64 / s / 1e9
    );
    let s = time(&|cmd| {
        gpu.op(
            cmd,
            "hc_up_collapse",
            &[&up, &lo, &norm, &out],
            &[width as u32, hc as u32, lr as u32, ut.stride as u32, 1],
            width.div_ceil(4),
            1,
        )
        .unwrap();
    });
    println!(
        "MATVEC hc_up_collapse bytes={} us={:.1} GBps={:.0}",
        ut.bytes,
        s * 1e6,
        ut.bytes as f64 / s / 1e9
    );
}

#[test]
#[ignore = "requires QWEN_MODEL; Q4_K/Q5_K/Q6_K/IQ4_NL/IQ3_S/IQ4_XS decode projections against F64 and across token counts"]
fn vector_kernels_match_f64_and_counts() {
    let gpu = Gpu::new().unwrap();
    let path = std::env::var("QWEN_MODEL").unwrap();
    let mut g = crate::gguf::Gguf::open(std::path::Path::new(&path)).unwrap();
    let mut seen = Vec::new();
    for layer in 0..48 {
        for suffix in [
            "attn_gate",
            "ssm_out",
            "attn_output",
            "attn_q",
            "attn_qkv",
            "attn_k",
            "attn_v",
            "ffn_gate_shexp",
            "ffn_up_shexp",
            "ffn_down_shexp",
        ] {
            let name = format!("blk.{layer}.{suffix}.weight");
            let Ok(t) = g.tensor(&name).cloned() else {
                continue;
            };
            if !matches!(t.kind, 12 | 13 | 14 | 20 | 21 | 23)
                || seen.contains(&(t.kind, t.dims.clone()))
            {
                continue;
            }
            seen.push((t.kind, t.dims.clone()));
            let (cols, rows) = (t.dims[0], t.dims[1]);
            let w = gpu.load_weight(&mut g, &t).unwrap();
            let reference = gpu
                .dequant(
                    &gpu.read_rows(&mut g, &name, &(0..rows).collect::<Vec<_>>())
                        .unwrap(),
                )
                .unwrap();
            let mut state = 11u64;
            let input: Vec<f32> = (0..4 * cols)
                .map(|_| (lcg(&mut state) - 0.5) as f32)
                .collect();
            let run = |x: &[f32], count: usize| {
                let x = gpu.floats(x).unwrap();
                let cmd = gpu.command().unwrap();
                let y = gpu
                    .matmul(&cmd, &w, cols, rows, t.stride, t.kind, 0, &x, count)
                    .unwrap();
                gpu.complete(cmd).unwrap();
                gpu.read_f32(&y).unwrap()
            };
            let single: Vec<Vec<f32>> = (0..4)
                .map(|k| run(&input[k * cols..(k + 1) * cols], 1))
                .collect();
            for count in 2..=4 {
                let batch = run(&input[..count * cols], count);
                for k in 0..count {
                    assert!(
                        batch[k * rows..(k + 1) * rows]
                            .iter()
                            .zip(&single[k])
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{name}: count {count} token {k} differs from count one"
                    );
                }
            }
            let mut worst = 0f64;
            for (k, out) in single.iter().enumerate() {
                for r in 0..rows {
                    let e: f64 = (0..cols)
                        .map(|c| {
                            f64::from(reference[r * cols + c]) * f64::from(input[k * cols + c])
                        })
                        .sum();
                    let a = f64::from(out[r]);
                    assert!(
                        a.is_finite() && (a - e).abs() <= 1e-4 + 5e-5 * e.abs(),
                        "{name} token {k} row {r}: {a} != {e}"
                    );
                    worst = worst.max((a - e).abs());
                }
            }
            println!(
                "VECTOR {name} kind={} {cols}x{rows} counts1-4 bitwise, max_abs_err={worst:.3e}",
                t.kind
            );
        }
    }
    assert!(!seen.is_empty());
}

#[test]
#[ignore = "requires QWEN_MODEL; count-one routed expert bandwidth streaming all 48 layers"]
fn routed_decode_bandwidth() {
    let gpu = Gpu::new().unwrap();
    let path = std::env::var("QWEN_MODEL").unwrap();
    let mut g = crate::gguf::Gguf::open(std::path::Path::new(&path)).unwrap();
    let top = 10usize;
    let mut state = 5u64;
    let mut layers = Vec::new();
    for layer in 0..48 {
        let mut ids = Vec::new();
        while ids.len() < top {
            let e = (lcg(&mut state) * 512.0) as u32;
            if !ids.contains(&e) {
                ids.push(e);
            }
        }
        let ids = gpu
            .bytes(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .unwrap();
        let w: Vec<_> = ["gate", "up", "down"]
            .iter()
            .map(|k| {
                let t = g
                    .tensor(&format!("blk.{layer}.ffn_{k}_exps.weight"))
                    .unwrap()
                    .clone();
                (gpu.load_weight(&mut g, &t).unwrap(), t)
            })
            .collect();
        layers.push((ids, w));
    }
    let x = gpu.floats(&vec![0.01f32; 2560]).unwrap();
    let act = gpu.floats(&vec![0.01f32; 640 * top]).unwrap();
    let out_up = gpu.output(640 * top).unwrap();
    let out_g = gpu.output(640 * top).unwrap();
    let out_down = gpu.output(2560 * top).unwrap();
    let routed = |cmd: &crate::metal::Command,
                  w: &(crate::metal::Buffer, crate::gguf::Tensor),
                  x: &crate::metal::Buffer,
                  ids: &crate::metal::Buffer,
                  y: &crate::metal::Buffer,
                  per: bool| {
        let t = &w.1;
        gpu.op(
            cmd,
            "project_routed_batch",
            &[&w.0, x, ids, y],
            &[
                t.dims[0] as u32,
                t.dims[1] as u32,
                t.stride as u32,
                t.kind,
                top as u32,
                u32::from(per),
            ],
            t.dims[1],
            top,
        )
        .unwrap();
    };
    for (stage, kinds) in [
        ("gate_up", vec![22u32]),
        ("gate_up", vec![16, 29]),
        ("down", vec![42]),
    ] {
        let mut bytes = 0usize;
        let mut samples = Vec::new();
        for round in 0..10 {
            let cmd = gpu.command().unwrap();
            bytes = 0;
            for _ in 0..4 {
                for (ids, w) in &layers {
                    if stage == "down" {
                        routed(&cmd, &w[2], &act, ids, &out_down, true);
                        bytes += w[2].1.bytes / 512 * top;
                    } else if kinds.contains(&w[0].1.kind) {
                        if w[0].1.kind == 22 {
                            let t = &w[0].1;
                            gpu.op(
                                &cmd,
                                "routed_gate_up_iq2s",
                                &[&w[0].0, &w[1].0, &x, ids, &out_up],
                                &[
                                    t.dims[0] as u32,
                                    t.dims[1] as u32,
                                    t.stride as u32,
                                    t.kind,
                                    top as u32,
                                    0,
                                ],
                                t.dims[1],
                                top,
                            )
                            .unwrap();
                        } else {
                            routed(&cmd, &w[0], &x, ids, &out_g, false);
                            routed(&cmd, &w[1], &x, ids, &out_up, false);
                        }
                        bytes += (w[0].1.bytes + w[1].1.bytes) / 512 * top;
                    }
                }
            }
            cmd.commit();
            cmd.waitUntilCompleted();
            assert!(cmd.error().is_none(), "{:?}", cmd.error());
            if round > 1 {
                samples.push(cmd.GPUEndTime() - cmd.GPUStartTime());
            }
        }
        samples.sort_by(f64::total_cmp);
        let s = samples[samples.len() / 2];
        println!(
            "ROUTED {stage} kinds={kinds:?} bytes={bytes} ms={:.3} per_layer_us={:.1} GBps={:.0}",
            s * 1e3,
            s * 1e6
                / (bytes as f64
                    / (bytes as f64
                        / (4.0
                            * layers
                                .iter()
                                .filter(|(_, w)| stage == "down" || kinds.contains(&w[0].1.kind))
                                .count() as f64))),
            bytes as f64 / s / 1e9
        );
    }
}

#[test]
#[ignore = "Apple GPU; DeltaNet prefill recurrence time versus heads and tokens"]
fn delta_net_scaling() {
    let gpu = Gpu::new().unwrap();
    let (d, nk) = (128usize, 16usize);
    for (nv, tokens) in [
        (48usize, 4095usize),
        (12, 4095),
        (1, 4095),
        (48, 1024),
        (48, 256),
    ] {
        let channels = 2 * nk * d + nv * d;
        let mut state = 3u64;
        let mut rand = |n: usize, scale: f64| -> crate::metal::Buffer {
            gpu.floats(
                &(0..n)
                    .map(|_| ((lcg(&mut state) - 0.5) * scale) as f32)
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        };
        let q = rand(tokens * nk * d, 0.1);
        let k = rand(tokens * nk * d, 0.1);
        let conv = rand(tokens * channels, 1.0);
        let gates = gpu
            .floats(
                &(0..tokens * nv)
                    .flat_map(|_| [0.95f32, 0.5])
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let st = rand(nv * d * d, 0.01);
        let y = gpu.output(tokens * nv * d).unwrap();
        let snap = gpu.output(1).unwrap();
        let mut samples = Vec::new();
        for round in 0..5 {
            let cmd = gpu.command().unwrap();
            gpu.op(
                &cmd,
                "delta_net_batch",
                &[&q, &k, &conv, &gates, &st, &y, &snap],
                &[
                    d as u32,
                    nk as u32,
                    nv as u32,
                    tokens as u32,
                    channels as u32,
                    0,
                ],
                d / 4,
                nv,
            )
            .unwrap();
            cmd.commit();
            cmd.waitUntilCompleted();
            assert!(cmd.error().is_none(), "{:?}", cmd.error());
            if round > 0 {
                samples.push(cmd.GPUEndTime() - cmd.GPUStartTime());
            }
        }
        samples.sort_by(f64::total_cmp);
        let s = samples[samples.len() / 2];
        println!(
            "DELTA nv={nv} tokens={tokens} ms={:.3} us_per_token={:.2}",
            s * 1e3,
            s * 1e6 / tokens as f64
        );
    }
}

// Round to the nearest F16 value (ties to even), including subnormals.
fn to_half(v: f32) -> f64 {
    let v = f64::from(v);
    if v == 0.0 {
        return 0.0;
    }
    let quantum = 2f64.powi((v.abs().log2().floor() as i32).max(-14) - 10);
    (v / quantum).round_ties_even() * quantum
}

#[test]
#[ignore = "requires Apple Silicon; Q4_K/Q5_K/IQ2_XS/IQ3_XXS/IQ4_NL/IQ3_S kernels on random blocks against F64 and across token counts"]
fn mix_formats_match_f64_and_counts() {
    let gpu = Gpu::new().unwrap();
    let (cols, rows, experts, top) = (2560usize, 37usize, 3usize, 2usize);
    let mut state = 23u64;
    // Multiples of 1/32 stay exact in the wide GEMM's half operand staging.
    let input: Vec<f32> = (0..32 * cols)
        .map(|i| ((i * 17 % 31) as f32 - 15.) * 0.03125)
        .collect();
    let close = |a: f32, e: f64| {
        let a = f64::from(a);
        a.is_finite() && (a - e).abs() <= 1e-4 + 5e-5 * e.abs()
    };
    for (kind, block, values, halves) in [
        (12u32, 144usize, 256usize, &[0usize, 2][..]),
        (13, 176, 256, &[0, 2]),
        (17, 74, 256, &[0]),
        (18, 98, 256, &[0]),
        (20, 18, 32, &[0]),
        (21, 110, 256, &[0]),
    ] {
        let stride = cols / values * block;
        let mut data: Vec<u8> = (0..experts * rows * stride)
            .map(|_| (lcg(&mut state) * 256.0) as u8)
            .collect();
        // Positive normal F16 scales in [2^-8, 2^-5.4]: random bytes stay finite.
        for b in data.chunks_exact_mut(block) {
            for &h in halves {
                let bits = 0x1c00u16 + (lcg(&mut state) * 2560.0) as u16;
                b[h..h + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        let matrix = crate::gguf::Matrix {
            data: data.clone(),
            cols,
            rows: experts * rows,
            count: 1,
            kind,
            stride,
        };
        let reference = gpu.dequant(&matrix).unwrap();
        let expect = |row: usize, token: usize| -> f64 {
            (0..cols)
                .map(|c| f64::from(reference[row * cols + c]) * f64::from(input[token * cols + c]))
                .sum()
        };
        let w = gpu.bytes(&data).unwrap();
        let run = |count: usize| {
            let x = gpu.floats(&input[..count * cols]).unwrap();
            let cmd = gpu.command().unwrap();
            let y = gpu
                .matmul(&cmd, &w, cols, rows, stride, kind, 1, &x, count)
                .unwrap();
            gpu.complete(cmd).unwrap();
            gpu.read_f32(&y).unwrap()
        };
        let one: Vec<Vec<f32>> = (0..4)
            .map(|k| {
                let x = gpu.floats(&input[k * cols..(k + 1) * cols]).unwrap();
                let cmd = gpu.command().unwrap();
                let y = gpu
                    .matmul(&cmd, &w, cols, rows, stride, kind, 1, &x, 1)
                    .unwrap();
                gpu.complete(cmd).unwrap();
                gpu.read_f32(&y).unwrap()
            })
            .collect();
        for count in 2..=4 {
            let batch = run(count);
            for (k, single) in one.iter().enumerate().take(count) {
                assert!(
                    batch[k * rows..(k + 1) * rows]
                        .iter()
                        .zip(single)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "kind {kind}: count {count} token {k} differs from count one"
                );
            }
        }
        for (k, out) in one.iter().enumerate() {
            for (r, &a) in out.iter().enumerate() {
                let e = expect(rows + r, k);
                assert!(close(a, e), "kind {kind} token {k} row {r}: {a} != {e}");
            }
        }
        // The wide GEMM stages weights16 values as F16 and accumulates in F32.
        let wide = run(32);
        for k in 0..32 {
            for r in 0..rows {
                let e: f64 = (0..cols)
                    .map(|c| {
                        to_half(reference[(rows + r) * cols + c]) * f64::from(input[k * cols + c])
                    })
                    .sum();
                let a = wide[k * rows + r];
                assert!(
                    close(a, e),
                    "kind {kind} wide token {k} row {r}: {a} != {e}"
                );
            }
        }
        if matches!(kind, 17 | 18 | 20 | 21) {
            let routed = |tokens: usize, first: usize| {
                let ids: Vec<u32> = (0..tokens * top)
                    .map(|p| ((first * top + p) % experts) as u32)
                    .collect();
                let ids = gpu
                    .bytes(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
                    .unwrap();
                let x = gpu
                    .floats(&input[first * cols..(first + tokens) * cols])
                    .unwrap();
                let y = gpu.output(rows * tokens * top).unwrap();
                let cmd = gpu.command().unwrap();
                gpu.op(
                    &cmd,
                    "project_routed_batch",
                    &[&w, &x, &ids, &y],
                    &[cols as u32, rows as u32, stride as u32, kind, top as u32, 0],
                    rows,
                    tokens * top,
                )
                .unwrap();
                gpu.complete(cmd).unwrap();
                gpu.read_f32(&y).unwrap()
            };
            let batch = routed(4, 0);
            for token in 0..4 {
                let single = routed(1, token);
                assert!(
                    batch[token * top * rows..(token + 1) * top * rows]
                        .iter()
                        .zip(&single)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "kind {kind}: routed token {token} differs between counts"
                );
                for k in 0..top {
                    let expert = (token * top + k) % experts;
                    for r in 0..rows {
                        let e = expect(expert * rows + r, token);
                        let a = single[k * rows + r];
                        assert!(
                            close(a, e),
                            "kind {kind} routed token {token} expert {expert} row {r}: {a} != {e}"
                        );
                    }
                }
            }
        }
        println!("MIX kind={kind} counts1-4 bitwise; count1, F16-staged count32 and routed F64 ok");
    }
}
