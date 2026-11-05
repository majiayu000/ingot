//! GPU regression tests and explicitly ignored hardware probes.
use super::*;

#[cfg(test)]
mod model_operator_tests {
    use super::*;
    fn close(actual: &[f32], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert!((*a as f64 - b).abs() < 2e-5, "{a} != {b}");
        }
    }
    #[test]
    fn quantized_matmul_handles_token_tails_and_expert_offsets() {
        let gpu = Gpu::new().unwrap();
        // Distinct expert/row values catch row-group tails and expert offset errors.
        for rows in [1, 3, 4, 5, 7, 33] {
            let raw: Vec<_> = (0..2 * rows * 32).map(|i| (i / 32 + 1) as f32).collect();
            let weights = gpu.floats(&raw).unwrap();
            for count in [1, 2, 4, 5, 8] {
                let inputs: Vec<_> = (0..32 * count).map(|i| (i / 32 + 1) as f32).collect();
                let x = gpu.floats(&inputs).unwrap();
                let cmd = gpu.command().unwrap();
                let out = gpu
                    .matmul(&cmd, &weights, 32, rows, 128, 0, 1, &x, count)
                    .unwrap();
                let actual = gpu.finish(cmd, &out).unwrap();
                let expected: Vec<_> = (0..rows * count)
                    .map(|i| ((rows + i % rows + 1) * 32 * (i / rows + 1)) as f64)
                    .collect();
                close(&actual, &expected);
            }
        }
    }
    #[test]
    fn routed_batch_handles_mixed_ids_and_q4_rows() {
        let gpu = Gpu::new().unwrap();
        let rows = 7;
        let cols = 64;
        let experts = 5;
        let top = 3;
        // Independent Q4_0 fixture: signed nibble minus eight, F16 scale 0.5.
        let mut data = Vec::new();
        for row in 0..rows * experts {
            for block in 0..cols / 32 {
                data.extend(0x3800u16.to_le_bytes());
                for j in 0..16 {
                    data.push((((row + block + j + 3) % 16) << 4 | ((row + block + j) % 16)) as u8);
                }
            }
        }
        let w = gpu.bytes(&data).unwrap();
        for tokens in [1, 2, 3, 5, 8] {
            let pairs = tokens * top;
            let ids: Vec<_> = (0..pairs).map(|i| ((i * 7 + 1) % experts) as u32).collect();
            let ids_buffer = gpu
                .bytes(&ids.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<_>>())
                .unwrap();
            for per in [false, true] {
                let inputs: Vec<f32> = (0..cols * if per { pairs } else { tokens })
                    .map(|i| (i % 19) as f32 * 0.125 - 1.)
                    .collect();
                let x = gpu.floats(&inputs).unwrap();
                let y = gpu.output(rows * pairs).unwrap();
                let cmd = gpu.command().unwrap();
                gpu.op(
                    &cmd,
                    "project_routed_batch",
                    &[&w, &x, &ids_buffer, &y],
                    &[cols as u32, rows as u32, 36, 2, top as u32, per as u32],
                    rows,
                    pairs,
                )
                .unwrap();
                let actual = gpu.finish(cmd, &y).unwrap();
                let expected = (0..pairs * rows)
                    .map(|i| {
                        let pair = i / rows;
                        let row = ids[pair] as usize * rows + i % rows;
                        let input = if per { pair } else { pair / top };
                        (0..cols)
                            .map(|j| {
                                let nibble =
                                    (row + j / 32 + j % 16 + if j % 32 >= 16 { 3 } else { 0 }) % 16;
                                (nibble as f64 - 8.) * 0.5 * f64::from(inputs[input * cols + j])
                            })
                            .sum()
                    })
                    .collect::<Vec<f64>>();
                close(&actual, &expected);
            }
        }
    }
    #[test]
    fn qwen_routing_handles_cross_lane_ties_and_partial_simd_groups() {
        let gpu = Gpu::new().unwrap();
        for count in [1, 31, 33, 288, 512] {
            for ties in [false, true] {
                let logits: Vec<_> = (0..count)
                    .map(|i| {
                        if ties {
                            1000.
                        } else {
                            ((i * 17) % 23) as f32 - 11.
                        }
                    })
                    .collect();
                let top = count.min(10);
                let (expected_ids, expected_weights) = crate::route(&logits, top).unwrap();
                let x = gpu.floats(&logits).unwrap();
                let ids = gpu.output(top).unwrap();
                let weights = gpu.output(top).unwrap();
                let cmd = gpu.command().unwrap();
                gpu.op(
                    &cmd,
                    "route_gpu",
                    &[&x, &ids, &weights],
                    &[count as u32, top as u32],
                    1,
                    1,
                )
                .unwrap();
                let actual = gpu.finish(cmd, &weights).unwrap();
                assert_eq!(
                    gpu.read_u32(&ids, top).unwrap(),
                    expected_ids.iter().map(|&i| i as u32).collect::<Vec<_>>()
                );
                close(
                    &actual,
                    &expected_weights
                        .iter()
                        .map(|&v| v as f64)
                        .collect::<Vec<_>>(),
                );
            }
        }
    }
    #[test]
    fn gdn_updates_all_heads_and_preserves_recurrence() {
        let gpu = Gpu::new().unwrap();
        let q = gpu.floats(&[1., 0., 0., 1.]).unwrap();
        let k = q.clone();
        let v = gpu.floats(&[2., 3., 4., 5., 6., 7., 8., 9.]).unwrap();
        let zeros = gpu.floats(&[0.; 4]).unwrap();
        let a = gpu.floats(&[-1.; 4]).unwrap();
        let state = gpu.zeros(16).unwrap();
        for scale in [0.5, 0.625] {
            let cmd = gpu.command().unwrap();
            let out = gpu.output(8).unwrap();
            gpu.op(
                &cmd,
                "delta_net",
                &[&q, &k, &v, &zeros, &zeros, &zeros, &a, &state, &out],
                &[2, 2],
                2,
                4,
            )
            .unwrap();
            let actual = gpu.finish(cmd, &out).unwrap();
            close(
                &actual,
                &[2., 3., 4., 5., 6., 7., 8., 9.].map(|v| v * scale / 2f64.sqrt()),
            );
        }
    }
    #[test]
    fn causal_dilated_convolution_uses_per_channel_history() {
        let gpu = Gpu::new().unwrap();
        let weights = gpu.floats(&[1., 2., 3., 1., 2., 3.]).unwrap();
        let history = gpu.zeros(8).unwrap();
        for (t, input) in [[1., 2.], [3., 4.], [5., 6.]].iter().enumerate() {
            let x = gpu.floats(input).unwrap();
            let out = gpu.output(2).unwrap();
            let cmd = gpu.command().unwrap();
            gpu.op(
                &cmd,
                "convolution",
                &[&x, &weights, &history, &out],
                &[2, 3, 2, 0, 12],
                1,
                1,
            )
            .unwrap();
            let actual = gpu.finish(cmd, &out).unwrap();
            let raw = if t == 2 {
                [17f64, 22.]
            } else {
                [input[0] as f64 * 3., input[1] as f64 * 3.]
            };
            close(&actual, &raw.map(|x| x / (1. + (-x).exp())));
        }
    }
    #[test]
    fn qsa_budget_tail_and_ties() {
        let gpu = Gpu::new().unwrap();
        for (length, ties) in [(2055, false), (2055, true), (2052, false)] {
            let blocks = length / 4;
            let scores = gpu
                .floats(
                    &(0..blocks)
                        .map(|b| if ties { 0. } else { b as f32 })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let mask = gpu.output(length).unwrap();
            let cmd = gpu.command().unwrap();
            gpu.op(
                &cmd,
                "index_mask",
                &[&scores, &mask],
                &[length as u32, 4, 2048],
                blocks.div_ceil(32),
                1,
            )
            .unwrap();
            let actual = gpu.finish(cmd, &mask).unwrap();
            let expected: Vec<f64> = (0..length)
                .map(|i| {
                    if ties {
                        if (2048..2052).contains(&i) { 0. } else { 1. }
                    } else if length == 2055 {
                        if i < 4 { 0. } else { 1. }
                    } else if i == 3 {
                        0.
                    } else {
                        1.
                    }
                })
                .collect();
            close(&actual, &expected);
        }
    }
    #[test]
    fn gqa_attention_and_mask_match_double_precision() {
        let gpu = Gpu::new().unwrap();
        let q = gpu.floats(&[1., 0., 1., 0., 0., 1., 0., 1.]).unwrap();
        let k = gpu.floats(&[1., 0., 0., 1., 0., 1., 1., 0.]).unwrap();
        let v = gpu.floats(&[2., 3., 6., 7., 4., 5., 8., 9.]).unwrap();
        for mask in [[1., 1.], [1., 0.]] {
            let maskbuf = gpu.floats(&mask).unwrap();
            let score = gpu.output(8).unwrap();
            let prob = gpu.output(8).unwrap();
            let out = gpu.output(8).unwrap();
            let cmd = gpu.command().unwrap();
            gpu.op(
                &cmd,
                "attention_scores",
                &[&q, &k, &maskbuf, &score],
                &[2, 4, 2, 2],
                2,
                4,
            )
            .unwrap();
            gpu.op(&cmd, "softmax_rows", &[&score, &prob], &[2], 4, 1)
                .unwrap();
            gpu.op(
                &cmd,
                "attention_values",
                &[&prob, &v, &out],
                &[2, 4, 2, 2],
                2,
                4,
            )
            .unwrap();
            let actual = gpu.finish(cmd, &out).unwrap();
            let p = if mask[1] == 0. {
                1.
            } else {
                let e = (1. / 2f64.sqrt()).exp();
                e / (1. + e)
            };
            let base = [
                4. - 2. * p,
                5. - 2. * p,
                4. - 2. * p,
                5. - 2. * p,
                8. - 2. * p,
                9. - 2. * p,
                8. - 2. * p,
                9. - 2. * p,
            ];
            close(&actual, &base);
        }
    }
}

#[cfg(test)]
mod glm_operator_tests {
    use super::*;
    fn read(gpu: &Gpu, b: &Buffer) -> Vec<f32> {
        gpu.read_u32(b, b.length() / 4)
            .unwrap()
            .into_iter()
            .map(f32::from_bits)
            .collect()
    }
    fn close(a: &[f32], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert!((*a as f64 - b).abs() < 3e-5, "{a} != {b}");
        }
    }
    #[test]
    fn glm_routing_uses_bias_only_for_selection_and_scales_unbiased_scores() {
        let gpu = Gpu::new().unwrap();
        let cmd = gpu.command().unwrap();
        let logits = gpu.floats(&[2., -2., 0., 2.]).unwrap();
        let bias = gpu.floats(&[0., 1., 0., 0.]).unwrap();
        let ids = gpu.output(2).unwrap();
        let weights = gpu.output(2).unwrap();
        gpu.op(
            &cmd,
            "glm_route",
            &[&logits, &bias, &ids, &weights],
            &[4, 2, 2.5f32.to_bits()],
            1,
            1,
        )
        .unwrap();
        gpu.complete(cmd).unwrap();
        assert_eq!(gpu.read_u32(&ids, 2).unwrap(), [1, 0]);
        close(
            &read(&gpu, &weights),
            &[2.5 / (1.0 + 2.0f64.exp()), 2.5 / (1.0 + (-2.0f64).exp())],
        );
        for count in [1, 33, 288] {
            for ties in [false, true] {
                let logits: Vec<_> = (0..count)
                    .map(|i| if ties { 0. } else { (i % 7) as f32 - 3. })
                    .collect();
                let bias: Vec<_> = (0..count)
                    .map(|i| {
                        if ties {
                            0.
                        } else {
                            ((i * 11) % 13) as f32 * 0.125
                        }
                    })
                    .collect();
                let scores: Vec<_> = logits
                    .iter()
                    .zip(&bias)
                    .map(|(&l, &b)| 1. / (1. + (-f64::from(l)).exp()) + f64::from(b))
                    .collect();
                let top = count.min(8);
                let mut expected: Vec<_> = (0..count).collect();
                expected.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
                expected.truncate(top);
                let expected_weights: Vec<_> = expected
                    .iter()
                    .map(|&i| 1. / (1. + (-f64::from(logits[i])).exp()))
                    .collect();
                let total: f64 = expected_weights.iter().sum();
                let x = gpu.floats(&logits).unwrap();
                let b = gpu.floats(&bias).unwrap();
                let ids = gpu.output(top).unwrap();
                let weights = gpu.output(top).unwrap();
                let cmd = gpu.command().unwrap();
                gpu.op(
                    &cmd,
                    "glm_route",
                    &[&x, &b, &ids, &weights],
                    &[count as u32, top as u32, 2.5f32.to_bits()],
                    1,
                    1,
                )
                .unwrap();
                let actual = gpu.finish(cmd, &weights).unwrap();
                assert_eq!(
                    gpu.read_u32(&ids, top).unwrap(),
                    expected.iter().map(|&i| i as u32).collect::<Vec<_>>()
                );
                close(
                    &actual,
                    &expected_weights
                        .iter()
                        .map(|v| v / total * 2.5)
                        .collect::<Vec<_>>(),
                );
            }
        }
    }
    #[test]
    fn glm_kda_decays_key_coordinates_and_restores_recurrent_updates() {
        let gpu = Gpu::new().unwrap();
        let d = 4;
        let heads = 2;
        let qv = [0.5, 0.5, 0.5, 0.5, 1., 0., 0., 0.];
        let kv = [0.5, 0.5, 0.5, 0.5, 0., 1., 0., 0.];
        let vv = [1., 2., -3., 4., 5., -6., 7., 8.];
        let fv = [-2., -1., 0., 1., -1., 0., 1., 2.];
        let dtv = [0.1, -0.2, 0.3, -0.4, 0.2, 0.3, -0.1, 0.4];
        let av = [-1., -2.];
        let bv = [0.3, -0.4];
        let q = gpu.floats(&qv).unwrap();
        let k = gpu.floats(&kv).unwrap();
        let v = gpu.floats(&vv).unwrap();
        let f = gpu.floats(&fv).unwrap();
        let dt = gpu.floats(&dtv).unwrap();
        let a = gpu.floats(&av).unwrap();
        let beta = gpu.floats(&bv).unwrap();
        let state = gpu.zeros(d * d * heads).unwrap();
        let out = gpu.output(d * heads).unwrap();
        let mut expected = vec![0f64; d * d * heads];
        for _ in 0..3 {
            let cmd = gpu.command().unwrap();
            gpu.op(
                &cmd,
                "glm_kda",
                &[&q, &k, &v, &f, &beta, &dt, &a, &state, &out],
                &[d as u32, (-5f32).to_bits()],
                d,
                heads,
            )
            .unwrap();
            gpu.complete(cmd).unwrap();
            let mut y = vec![0f64; d * heads];
            for h in 0..heads {
                for j in 0..d {
                    let row = &mut expected[(h * d + j) * d..(h * d + j + 1) * d];
                    let mut prediction = 0.;
                    for i in 0..d {
                        let z = (fv[h * d + i] + dtv[h * d + i]) as f64;
                        row[i] *= (-5.0 / (1.0 + ((av[h] as f64) * z).exp())).exp();
                        prediction += row[i] * (kv[h * d + i] as f64);
                    }
                    let delta =
                        ((vv[h * d + j] as f64) - prediction) / (1.0 + (-(bv[h] as f64)).exp());
                    for i in 0..d {
                        row[i] += (kv[h * d + i] as f64) * delta;
                        y[h * d + j] += row[i] * (qv[h * d + i] as f64) / 2.;
                    }
                }
            }
            close(&read(&gpu, &out), &y);
            close(&read(&gpu, &state), &expected);
        }
    }
    #[test]
    fn glm_sinkhorn_and_residual_mix_preserve_source_destination_orientation() {
        let gpu = Gpu::new().unwrap();
        let mixes: Vec<_> = (0..24).map(|i| (i as f32 - 10.) / 7.).collect();
        let bases: Vec<_> = (0..24).map(|i| ((i * 7) % 13) as f32 / 11.).collect();
        let scales = [0.2, -0.3, 0.4];
        let m = gpu.floats(&mixes).unwrap();
        let b = gpu.floats(&bases).unwrap();
        let sc = gpu.floats(&scales).unwrap();
        let gates = gpu.output(24).unwrap();
        let cmd = gpu.command().unwrap();
        gpu.op(
            &cmd,
            "glm_hc_gates",
            &[&m, &b, &sc, &gates],
            &[4, 20, 1e-6f32.to_bits()],
            1,
            1,
        )
        .unwrap();
        gpu.complete(cmd).unwrap();
        let mut expected = vec![0f64; 24];
        for h in 0..4 {
            expected[h] = 1. / (1. + (-((mixes[h] * scales[0] + bases[h]) as f64)).exp()) + 1e-6;
            expected[4 + h] =
                2. / (1. + (-((mixes[4 + h] * scales[1] + bases[4 + h]) as f64)).exp());
        }
        for src in 0..4 {
            let values: Vec<_> = (0..4)
                .map(|dst| (mixes[8 + src * 4 + dst] * scales[2] + bases[8 + src * 4 + dst]) as f64)
                .collect();
            let sum: f64 = values.iter().map(|x| x.exp()).sum();
            for dst in 0..4 {
                expected[8 + src * 4 + dst] = values[dst].exp() / sum + 1e-6;
            }
        }
        for it in 0..20 {
            if it > 0 {
                for src in 0..4 {
                    let sum: f64 =
                        (0..4).map(|dst| expected[8 + src * 4 + dst]).sum::<f64>() + 1e-6;
                    for dst in 0..4 {
                        expected[8 + src * 4 + dst] /= sum;
                    }
                }
            }
            for dst in 0..4 {
                let sum: f64 = (0..4).map(|src| expected[8 + src * 4 + dst]).sum::<f64>() + 1e-6;
                for src in 0..4 {
                    expected[8 + src * 4 + dst] /= sum;
                }
            }
        }
        close(&read(&gpu, &gates), &expected);
        let residual = gpu.floats(&[1., 2., 3., 4., 5., 6., 7., 8.]).unwrap();
        let x = gpu.floats(&[10., 20.]).unwrap();
        let out = gpu.output(8).unwrap();
        let cmd = gpu.command().unwrap();
        gpu.op(
            &cmd,
            "glm_hc_post",
            &[&residual, &x, &gates, &out],
            &[2, 4],
            1,
            1,
        )
        .unwrap();
        gpu.complete(cmd).unwrap();
        let values: Vec<_> = (0..8)
            .map(|i| {
                let dst = i / 2;
                let j = i % 2;
                let mut sum = [10., 20.][j] * expected[4 + dst];
                for src in 0..4 {
                    sum += ((src * 2 + j + 1) as f64) * expected[8 + src * 4 + dst];
                }
                sum
            })
            .collect();
        close(&read(&gpu, &out), &values);
    }
    #[test]
    fn glm_mla_absorbs_keys_and_projects_values_for_each_head() {
        let gpu = Gpu::new().unwrap();
        let (heads, qk, latent, value, length) = (2, 3, 5, 4, 3);
        let query: Vec<f32> = (0..heads * qk).map(|i| (i as f32 - 2.) / 8.).collect();
        let kb: Vec<f32> = (0..heads * latent * qk)
            .map(|i| ((i * 7 % 19) as f32 - 9.) / 16.)
            .collect();
        let kv: Vec<f32> = (0..length * latent)
            .map(|i| ((i * 3 % 11) as f32 - 5.) / 4.)
            .collect();
        let vb: Vec<f32> = (0..heads * value * latent)
            .map(|i| ((i * 5 % 13) as f32 - 6.) / 16.)
            .collect();
        let q = gpu.floats(&query).unwrap();
        let k = gpu.floats(&kb).unwrap();
        let cache = gpu.floats(&kv).unwrap();
        let v = gpu.floats(&vb).unwrap();
        let ids = gpu
            .bytes(
                &[0u32, 1]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let mask = gpu.floats(&[1., 0., 1.]).unwrap();
        let absorbed = gpu.output(heads * latent).unwrap();
        let scores = gpu.output(heads * length).unwrap();
        let probs = gpu.output(heads * length).unwrap();
        let values = gpu.output(heads * latent).unwrap();
        let out = gpu.output(heads * value).unwrap();
        let cmd = gpu.command().unwrap();
        gpu.op(
            &cmd,
            "project_experts",
            &[&k, &q, &ids, &absorbed],
            &[qk as u32, latent as u32, (qk * 4) as u32, 0, qk as u32],
            latent,
            heads,
        )
        .unwrap();
        gpu.op(
            &cmd,
            "glm_mla_scores",
            &[&absorbed, &cache, &mask, &scores],
            &[latent as u32, length as u32, qk as u32],
            length,
            heads,
        )
        .unwrap();
        gpu.op(
            &cmd,
            "softmax_rows",
            &[&scores, &probs],
            &[length as u32],
            heads,
            1,
        )
        .unwrap();
        gpu.op(
            &cmd,
            "attention_values",
            &[&probs, &cache, &values],
            &[latent as u32, heads as u32, 1, length as u32],
            latent,
            heads,
        )
        .unwrap();
        gpu.op(
            &cmd,
            "project_experts",
            &[&v, &values, &ids, &out],
            &[
                latent as u32,
                value as u32,
                (latent * 4) as u32,
                0,
                latent as u32,
            ],
            value,
            heads,
        )
        .unwrap();
        gpu.complete(cmd).unwrap();
        let mut expected_absorbed = vec![0f64; heads * latent];
        let mut expected_probs = vec![0f64; heads * length];
        let mut expected_values = vec![0f64; heads * latent];
        let mut expected_out = vec![0f64; heads * value];
        for h in 0..heads {
            for j in 0..latent {
                for i in 0..qk {
                    expected_absorbed[h * latent + j] +=
                        kb[(h * latent + j) * qk + i] as f64 * query[h * qk + i] as f64;
                }
            }
            for t in [0, 2] {
                let score = (0..latent)
                    .map(|j| expected_absorbed[h * latent + j] * kv[t * latent + j] as f64)
                    .sum::<f64>()
                    / (qk as f64).sqrt();
                expected_probs[h * length + t] = score.exp();
            }
            let sum = expected_probs[h * length..(h + 1) * length]
                .iter()
                .sum::<f64>();
            for t in 0..length {
                expected_probs[h * length + t] /= sum;
            }
            for j in 0..latent {
                for t in 0..length {
                    expected_values[h * latent + j] +=
                        expected_probs[h * length + t] * kv[t * latent + j] as f64;
                }
            }
            for j in 0..value {
                for i in 0..latent {
                    expected_out[h * value + j] +=
                        vb[(h * value + j) * latent + i] as f64 * expected_values[h * latent + i];
                }
            }
        }
        close(&read(&gpu, &absorbed), &expected_absorbed);
        close(&read(&gpu, &probs), &expected_probs);
        close(&read(&gpu, &values), &expected_values);
        close(&read(&gpu, &out), &expected_out);
    }
    #[test]
    fn glm_pooling_sparse_mask_and_clamped_swiglu() {
        let gpu = Gpu::new().unwrap();
        let keys = gpu.floats(&[1., 2., 3., 4., 5., 6., 7., 8.]).unwrap();
        let gate = gpu.floats(&[0.; 8]).unwrap();
        let ape = gpu.floats(&[0., 1., 1., 0., 0., -1., -1., 0.]).unwrap();
        let pooled = gpu.output(2).unwrap();
        let cmd = gpu.command().unwrap();
        gpu.op(
            &cmd,
            "glm_pool",
            &[&keys, &gate, &ape, &pooled],
            &[2, 4, 0],
            1,
            1,
        )
        .unwrap();
        gpu.complete(cmd).unwrap();
        let mut expected = vec![0f64; 2];
        let ape = [0f64, 1., 1., 0., 0., -1., -1., 0.];
        for j in 0..2 {
            let mut sum = 0.;
            for t in 0..4 {
                let a = ape[2 * t + j].exp();
                sum += a;
                expected[j] += a * ((2 * t + j + 1) as f64);
            }
            expected[j] /= sum;
        }
        close(&read(&gpu, &pooled), &expected);
        let score = gpu.floats(&[1., 3., 2.]).unwrap();
        let mask = gpu.output(14).unwrap();
        let cmd = gpu.command().unwrap();
        gpu.op(&cmd, "glm_index_mask", &[&score, &mask], &[14, 4, 8], 1, 1)
            .unwrap();
        gpu.complete(cmd).unwrap();
        assert_eq!(
            read(&gpu, &mask),
            [0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
        );
        for (length, ties) in [(2055, false), (2055, true), (2052, false)] {
            let complete = length / 4;
            let score = gpu
                .floats(
                    &(0..complete)
                        .map(|b| if ties { 0. } else { b as f32 })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let mask = gpu.output(length).unwrap();
            let cmd = gpu.command().unwrap();
            gpu.op(
                &cmd,
                "glm_index_mask",
                &[&score, &mask],
                &[length as u32, 4, 2048],
                length.div_ceil(32),
                1,
            )
            .unwrap();
            gpu.complete(cmd).unwrap();
            for (t, actual) in read(&gpu, &mask).into_iter().enumerate() {
                let visible = t >= complete * 4
                    || if ties {
                        t / 4 < 512
                    } else {
                        t / 4 >= complete - 512
                    };
                assert_eq!(actual, if visible { 1. } else { 0. });
            }
        }
        for length in [1usize, 3, 4, 7, 8, 9, 11] {
            let complete = length / 4;
            let scores: Vec<_> = (0..complete.max(1)).map(|i| i as f32).collect();
            let score = gpu.floats(&scores).unwrap();
            let mask = gpu.output(length).unwrap();
            let cmd = gpu.command().unwrap();
            gpu.op(
                &cmd,
                "glm_index_mask",
                &[&score, &mask],
                &[length as u32, 4, 4],
                complete.max(1).div_ceil(32),
                1,
            )
            .unwrap();
            let actual = gpu.finish(cmd, &mask).unwrap();
            for (i, v) in actual.into_iter().enumerate() {
                assert_eq!(
                    v,
                    if i >= complete * 4 || complete <= 1 || i / 4 == complete - 1 {
                        1.0
                    } else {
                        0.0
                    }
                );
            }
        }
        let gate = gpu.floats(&[20., -20., 2.]).unwrap();
        let up = gpu.floats(&[30., -30., 3.]).unwrap();
        let out = gpu.output(3).unwrap();
        let cmd = gpu.command().unwrap();
        gpu.op(
            &cmd,
            "glm_swiglu",
            &[&gate, &up, &out],
            &[3, 10f32.to_bits()],
            1,
            1,
        )
        .unwrap();
        let actual = gpu.finish(cmd, &out).unwrap();
        close(
            &actual,
            &[
                100. / (1. + (-10f64).exp()),
                200. / (1. + 20f64.exp()),
                6. / (1. + (-2f64).exp()),
            ],
        );
    }
}

#[cfg(test)]
mod real_matrix_probe {
    use super::*;
    #[test]
    #[ignore = "QWEN_DRAFT real Q8/BF16 fidelity shapes; fixed F32 pipelines, no timing"]
    fn qwen_fidelity_wide_f32() {
        let gpu = Gpu::new().unwrap();
        let path = std::env::var("QWEN_DRAFT").unwrap();
        let mut g = crate::gguf::Gguf::open(std::path::Path::new(&path)).unwrap();
        for (suffix, kind, cols, rows, counts) in [
            ("nextn.eh_proj.weight", 8, 5120, 2560, [60usize, 512]),
            ("hc_attn_down.weight", 8, 10240, 320, [128, 129]),
            ("hc_attn_up.weight", 8, 320, 10240, [128, 129]),
            ("attn_k.weight", 8, 2560, 512, [128, 129]),
            ("indexer.k_proj.weight", 30, 2560, 128, [128, 129]),
        ] {
            let name = format!("blk.48.{suffix}");
            let t = g.tensor(&name).unwrap();
            assert_eq!((t.kind, t.dims.as_slice()), (kind, [cols, rows].as_slice()));
            let matrix = g
                .rows(&gpu.reads, &name, &(0..rows).collect::<Vec<_>>())
                .unwrap();
            // Existing GPU read_rows decode supplies F32 weights; only the dot is F64.
            let decoded = gpu.dequant(&matrix).unwrap();
            let w = gpu.bytes(&matrix.data).unwrap();
            let pipeline = &gpu.projections[&("quant_gemm_wide_f32", kind)];
            let limit = gpu.device.maxThreadgroupMemoryLength();
            let used = pipeline.staticThreadgroupMemoryLength();
            println!(
                "F32_WIDE_LIMIT kind={kind} device_tg_bytes={limit} pipeline_tg_bytes={used} max_threads={} simd_width={}",
                pipeline.maxTotalThreadsPerThreadgroup(),
                pipeline.threadExecutionWidth()
            );
            assert!(used >= 12288 && used <= limit);
            assert!(pipeline.maxTotalThreadsPerThreadgroup() >= 128);
            assert_eq!(pipeline.threadExecutionWidth(), 32);
            for count in counts {
                let input = (0..count * cols)
                    .map(|i| ((i * 17 % 31) as f32 - 15.) * 0.03125)
                    .collect::<Vec<_>>();
                let x = gpu.floats(&input).unwrap();
                let n = rows * count;
                const GUARD: u32 = 0x7fc1_5620;
                let bytes = vec![GUARD; n + 32]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>();
                let old_storage = gpu.bytes(&bytes).unwrap();
                let new_storage = gpu.bytes(&bytes).unwrap();
                let old_out = old_storage.view_f32(16, n).unwrap();
                let new_out = new_storage.view_f32(16, n).unwrap();
                let cmd = gpu.command().unwrap();
                for (kernel, tile, nt, out) in [
                    ("quant_gemm", 16, 16, &old_out),
                    ("quant_gemm_wide_f32", 64, 32, &new_out),
                ] {
                    gpu.encode(
                        &cmd,
                        &gpu.projections[&(kernel, kind)],
                        &[&w, &x, out],
                        &[
                            cols as u32,
                            rows as u32,
                            matrix.stride as u32,
                            kind,
                            0,
                            count as u32,
                        ],
                        MTLSize {
                            width: rows,
                            height: count.div_ceil(nt),
                            depth: 1,
                        },
                        tile,
                    )
                    .unwrap();
                }
                let selected = gpu
                    .matmul(&cmd, &w, cols, rows, matrix.stride, kind, 0, &x, count)
                    .unwrap();
                gpu.complete(cmd).unwrap();
                // Read only live views as F32; NaN guard padding is inspected as raw bits.
                let old = gpu.read_f32(&old_out).unwrap();
                let actual = gpu.read_f32(&new_out).unwrap();
                assert_eq!(
                    gpu.read_u32(&selected, n).unwrap(),
                    gpu.read_u32(&new_out, n).unwrap()
                );
                for storage in [&old_storage, &new_storage] {
                    let bits = gpu.read_u32(storage, n + 32).unwrap();
                    assert!(
                        bits[..16]
                            .iter()
                            .chain(&bits[n + 16..])
                            .all(|&v| v == GUARD)
                    );
                }
                let (mut max_abs, mut error, mut energy, mut bit_diff) = (0f64, 0f64, 0f64, 0usize);
                for (&a, &b) in actual.iter().zip(&old) {
                    let delta = f64::from(a) - f64::from(b);
                    max_abs = max_abs.max(delta.abs());
                    error += delta * delta;
                    energy += f64::from(b).powi(2);
                    bit_diff += usize::from(a.to_bits() != b.to_bits());
                    assert!(delta.abs() <= 1e-4 + 5e-5 * f64::from(b).abs());
                }
                let mut f64_max = 0f64;
                for token in [0, count / 2, count - 1] {
                    for row in [0, rows / 2, rows - 1] {
                        let expected = decoded[row * cols..(row + 1) * cols]
                            .iter()
                            .zip(&input[token * cols..(token + 1) * cols])
                            .map(|(&a, &b)| f64::from(a) * f64::from(b))
                            .sum::<f64>();
                        for values in [&old, &actual] {
                            let delta = (f64::from(values[token * rows + row]) - expected).abs();
                            f64_max = f64_max.max(delta);
                            assert!(delta <= 1e-4 + 5e-5 * expected.abs());
                        }
                    }
                }
                println!(
                    "F32_WIDE_NUMERIC tensor={name} kind={kind} cols={cols} rows={rows} count={count} finite=full guard=pass selector_fixed_exact=true old_f32_bitdiff={bit_diff} max_abs={max_abs:.9e} relative_l2={:.9e} gpu_dequant_f64_boundary_samples=9 f64_max_abs={f64_max:.9e}",
                    (error / energy.max(1e-30)).sqrt()
                );
            }
        }
    }

    #[test]
    #[ignore = "requires real GGUFs; checks grouped expert tile boundaries against F64"]
    fn real_tiled_experts_f64() {
        let gpu = Gpu::new().unwrap();
        let mut checked = std::collections::BTreeSet::new();
        for variable in ["QWEN_MODEL", "GLM_MODEL"] {
            let mut g =
                crate::gguf::Gguf::open(std::path::Path::new(&std::env::var(variable).unwrap()))
                    .unwrap();
            let mut names = g.tensors.keys().cloned().collect::<Vec<_>>();
            names.sort();
            for name in names {
                let t = g.tensor(&name).unwrap().clone();
                if checked.contains(&t.kind) || t.bytes / t.stride < 130 || t.dims[0] < 128 {
                    continue;
                }
                let matrix = g
                    .rows(&gpu.reads, &name, &(0..130).collect::<Vec<_>>())
                    .unwrap();
                let decoded = gpu.dequant(&matrix).unwrap();
                let w = gpu.bytes(&matrix.data).unwrap();
                // 65 output rows exercise both the full row tile and a one-row tail.
                // A matrix-mode tail can contain fewer than eight tokens because its
                // parent expert group, not this final tile, selected the matrix path.
                let tiles: [u32; 24] = [
                    1, 3, 7, 0, 0, 10, 16, 1, 1, 26, 9, 1, 0, 35, 1, 1, 0, 36, 32, 1, 1, 68, 19, 1,
                ];
                let metadata = gpu
                    .bytes(
                        &tiles
                            .iter()
                            .flat_map(|i| i.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let input = (0..87 * matrix.cols)
                    .map(|i| ((i * 17 % 31) as f32 - 15.) * 0.03125)
                    .collect::<Vec<_>>();
                let x = gpu.floats(&input).unwrap();
                let out = gpu.zeros(87 * 65).unwrap();
                let cmd = gpu.command().unwrap();
                gpu.op(
                    &cmd,
                    "quant_expert_gemm",
                    &[&w, &x, &metadata, &out],
                    &[matrix.cols as u32, 65, matrix.stride as u32, matrix.kind],
                    65,
                    6,
                )
                .unwrap();
                let actual = gpu.finish(cmd, &out).unwrap();
                assert!(actual[..3 * 65].iter().all(|&v| v == 0.));
                let mut maximum = 0f64;
                for tile in tiles.chunks_exact(4) {
                    for token in tile[1] as usize..(tile[1] + tile[2]) as usize {
                        for row in 0..65 {
                            let wr = (tile[0] as usize * 65 + row) * matrix.cols;
                            let expected = decoded[wr..wr + matrix.cols]
                                .iter()
                                .zip(&input[token * matrix.cols..(token + 1) * matrix.cols])
                                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                                .sum::<f64>();
                            let error = (f64::from(actual[token * 65 + row]) - expected).abs();
                            maximum = maximum.max(error);
                            assert!(
                                error <= 1e-4 + 5e-5 * expected.abs(),
                                "format {} token {token} row {row}: {} != {expected}",
                                matrix.kind,
                                actual[token * 65 + row]
                            );
                        }
                    }
                }
                println!("EXPERT_TILE_F64 format={} max_abs={maximum}", matrix.kind);
                checked.insert(matrix.kind);
            }
        }
        assert_eq!(
            checked,
            std::collections::BTreeSet::from([0, 8, 10, 14, 16, 20, 21, 22, 23, 29, 30, 42])
        );
    }

    #[test]
    #[ignore = "requires real GGUFs; independent dequantized F64 matrix sums"]
    fn real_quantized_gemm_f64() {
        let gpu = Gpu::new().unwrap();
        let mut checked = std::collections::BTreeSet::new();
        for variable in ["QWEN_MODEL", "GLM_MODEL"] {
            let mut g =
                crate::gguf::Gguf::open(std::path::Path::new(&std::env::var(variable).unwrap()))
                    .unwrap();
            let mut names = g.tensors.keys().cloned().collect::<Vec<_>>();
            names.sort();
            for name in names {
                let t = g.tensor(&name).unwrap().clone();
                if checked.contains(&t.kind) || t.bytes / t.stride < 33 || t.dims[0] < 128 {
                    continue;
                }
                let rows = (0..33).collect::<Vec<_>>();
                let matrix = g.rows(&gpu.reads, &name, &rows).unwrap();
                let decoded = gpu.dequant(&matrix).unwrap();
                let w = gpu.bytes(&matrix.data).unwrap();
                let mut maximum = 0f64;
                for count in [1, 2, 3, 4, 7, 8, 17, 32, 128] {
                    let input = (0..count * matrix.cols)
                        .map(|i| ((i * 17 % 31) as f32 - 15.) * 0.03125)
                        .collect::<Vec<_>>();
                    let x = gpu.floats(&input).unwrap();
                    let cmd = gpu.command().unwrap();
                    let y = gpu
                        .matmul(
                            &cmd,
                            &w,
                            matrix.cols,
                            33,
                            matrix.stride,
                            matrix.kind,
                            0,
                            &x,
                            count,
                        )
                        .unwrap();
                    let actual = gpu.finish(cmd, &y).unwrap();
                    if count >= 32 {
                        let old = gpu.output(33 * count).unwrap();
                        let cmd = gpu.command().unwrap();
                        gpu.encode(
                            &cmd,
                            &gpu.projections[&("quant_gemm", matrix.kind)],
                            &[&w, &x, &old],
                            &[
                                matrix.cols as u32,
                                33,
                                matrix.stride as u32,
                                matrix.kind,
                                0,
                                count as u32,
                            ],
                            MTLSize {
                                width: 33,
                                height: count.div_ceil(16),
                                depth: 1,
                            },
                            16,
                        )
                        .unwrap();
                        assert_eq!(
                            gpu.finish(cmd, &old).unwrap(),
                            actual,
                            "wide tile must keep F32 accumulation order: format {} count {}",
                            matrix.kind,
                            count
                        );
                    }

                    for token in 0..count {
                        for row in 0..33 {
                            let expected = decoded[row * matrix.cols..(row + 1) * matrix.cols]
                                .iter()
                                .zip(&input[token * matrix.cols..(token + 1) * matrix.cols])
                                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                                .sum::<f64>();
                            let err = (actual[token * 33 + row] as f64 - expected).abs();
                            maximum = maximum.max(err);
                            assert!(
                                err <= 1e-4 + 5e-5 * expected.abs(),
                                "format {} token {token} row {row}: {} != {expected}",
                                t.kind,
                                actual[token * 33 + row]
                            );
                        }
                    }
                }
                println!("GEMM_F64 format={} tensor={name} max_abs={maximum}", t.kind);
                checked.insert(t.kind);
            }
        }
        assert_eq!(
            checked,
            std::collections::BTreeSet::from([0, 8, 10, 14, 16, 20, 21, 22, 23, 29, 30, 42])
        );
    }
}

#[cfg(test)]
mod sparse_attention_tests {
    use super::*;
    #[test]
    fn compact_attention_keeps_original_reduction_lanes() {
        let gpu = Gpu::new().unwrap();
        let (dim, heads, kv, count) = (32, 4, 2, 3);
        for length in [3, 4, 31, 32, 33, 2047, 2051, 2052, 8192] {
            for sparse in [false, true] {
                let position = length - count;
                let data = |n: usize, salt: usize| {
                    (0..n)
                        .map(|i| ((i * 17 + salt) % 113) as f32 / 113. - 0.5)
                        .collect::<Vec<_>>()
                };
                let query = data(count * heads * dim, 1);
                let q = gpu.floats(&query).unwrap();
                let k = gpu.floats(&data(length * kv * dim, 3)).unwrap();
                let v = gpu.floats(&data(length * kv * dim, 7)).unwrap();
                let masks = (0..count * length)
                    .map(|i| {
                        let t = i / length;
                        let j = i % length;
                        (j <= position + t && (!sparse || j % 5 < 2 || j == position + t)) as u8
                            as f32
                    })
                    .collect::<Vec<_>>();
                let mask = gpu.floats(&masks).unwrap();
                let ids = gpu.output(length * count).unwrap();
                let offsets = gpu.output(33 * count).unwrap();
                let score = gpu.output(count * heads * length).unwrap();
                let prob = gpu.output(count * heads * length).unwrap();
                let out = gpu.output(count * heads * dim).unwrap();
                let cmd = gpu.command().unwrap();
                gpu.op(
                    &cmd,
                    "compact_mask",
                    &[&mask, &ids, &offsets],
                    &[length as u32, position as u32, length as u32],
                    count,
                    1,
                )
                .unwrap();
                let p = [dim as u32, heads as u32, kv as u32, length as u32];
                gpu.op(
                    &cmd,
                    "attention_scores_compact",
                    &[&q, &k, &ids, &offsets, &score],
                    &p,
                    length,
                    heads * count,
                )
                .unwrap();
                gpu.op(
                    &cmd,
                    "softmax_compact",
                    &[&score, &offsets, &prob],
                    &[length as u32, heads as u32],
                    heads * count,
                    1,
                )
                .unwrap();
                gpu.op(
                    &cmd,
                    "attention_values_compact",
                    &[&prob, &v, &ids, &offsets, &out],
                    &p,
                    dim,
                    heads * count,
                )
                .unwrap();
                let actual = gpu.finish(cmd, &out).unwrap();
                let offsets = gpu.read_u32(&offsets, 33 * count).unwrap();
                let ids = gpu.read_u32(&ids, length * count).unwrap();
                for t in 0..count {
                    let mut expected_ids = Vec::new();
                    for lane in 0..32 {
                        assert_eq!(offsets[t * 33 + lane] as usize, expected_ids.len());
                        expected_ids.extend(
                            (lane..=position + t)
                                .step_by(32)
                                .filter(|&j| masks[t * length + j] > 0.)
                                .map(|j| j as u32),
                        );
                    }
                    assert_eq!(offsets[t * 33 + 32] as usize, expected_ids.len());
                    assert_eq!(
                        &ids[t * length..t * length + expected_ids.len()],
                        expected_ids
                    );
                    let q = gpu
                        .floats(&query[t * heads * dim..(t + 1) * heads * dim])
                        .unwrap();
                    let n = position + t + 1;
                    let mask = gpu.floats(&masks[t * length..t * length + n]).unwrap();
                    let s = gpu.output(heads * n).unwrap();
                    let probs = gpu.output(heads * n).unwrap();
                    let y = gpu.output(heads * dim).unwrap();
                    let cmd = gpu.command().unwrap();
                    let p = [dim as u32, heads as u32, kv as u32, n as u32];
                    gpu.op(&cmd, "attention_scores", &[&q, &k, &mask, &s], &p, n, heads)
                        .unwrap();
                    gpu.op(&cmd, "softmax_rows", &[&s, &probs], &[n as u32], heads, 1)
                        .unwrap();
                    gpu.op(&cmd, "attention_values", &[&probs, &v, &y], &p, dim, heads)
                        .unwrap();
                    let expected = gpu.finish(cmd, &y).unwrap();
                    assert_eq!(
                        &actual[t * heads * dim..(t + 1) * heads * dim],
                        expected,
                        "length={length} sparse={sparse} token={t}"
                    );
                }
                if !sparse {
                    let score = gpu.output(count * heads * length).unwrap();
                    let prob = gpu.output(count * heads * length).unwrap();
                    let out = gpu.output(count * heads * dim).unwrap();
                    let cmd = gpu.command().unwrap();
                    let p = [
                        dim as u32,
                        heads as u32,
                        kv as u32,
                        length as u32,
                        position as u32,
                    ];
                    gpu.op(
                        &cmd,
                        "attention_scores_dense_batch",
                        &[&q, &k, &score],
                        &p,
                        length,
                        heads * count,
                    )
                    .unwrap();
                    gpu.op(
                        &cmd,
                        "softmax_dense_batch",
                        &[&score, &prob],
                        &[length as u32, heads as u32, position as u32],
                        heads * count,
                        1,
                    )
                    .unwrap();
                    gpu.op(
                        &cmd,
                        "attention_values_dense_batch",
                        &[&prob, &v, &out],
                        &p,
                        dim,
                        heads * count,
                    )
                    .unwrap();
                    assert_eq!(gpu.finish(cmd, &out).unwrap(), actual);
                }
            }
        }
    }
}

#[cfg(test)]
mod workspace_tests {
    use super::*;
    #[test]
    fn workspace_pressure_keeps_unneeded_idle_buffers_reusable() {
        let gpu = Gpu::new().unwrap();
        let page = 16 * 1024;
        gpu.configure_workspace(5 * page);
        let small = gpu.output(4).unwrap();
        let large = gpu.output(3 * page / 4).unwrap();
        let live = gpu.output(5).unwrap();
        drop((small, large));
        let before = gpu.workspace_statistics().0;
        let medium = gpu.output(2 * page / 4).unwrap();
        let small_again = gpu.output(4).unwrap();
        assert_eq!(
            gpu.workspace_statistics().0,
            before + 1,
            "making room for a larger shape must preserve unrelated reusable storage"
        );
        assert!(gpu.workspace_statistics().2 <= 5 * page);
        drop((live, medium, small_again));
    }
    #[test]
    fn workspace_keeps_encoded_and_submitted_buffers_alive() {
        let gpu = Gpu::new().unwrap();
        gpu.configure_workspace(8 * 16 * 1024);
        let input = gpu.floats(&[3., 7., -5., 9.]).unwrap();
        let result = gpu.output(4).unwrap();
        let first_address = result.contents();
        let producer = gpu.command().unwrap();
        gpu.op(&producer, "copy_range", &[&input, &result], &[4, 0], 4, 1)
            .unwrap();
        let final_result = gpu.output(4).unwrap();
        let consumer = gpu.command().unwrap();
        gpu.op(
            &consumer,
            "copy_range",
            &[&result, &final_result],
            &[4, 0],
            4,
            1,
        )
        .unwrap();
        drop(result);
        let other = gpu.output(4).unwrap();
        assert_ne!(
            other.contents(),
            first_address,
            "encoded work still owns the allocation"
        );
        gpu.complete(producer).unwrap();
        let another = gpu.output(4).unwrap();
        assert_ne!(
            another.contents(),
            first_address,
            "the next command also owns the allocation"
        );
        gpu.submit(&consumer);
        drop(consumer); // An error-return path must drain submitted work before pooling.
        assert_eq!(gpu.read_f32(&final_result).unwrap(), [3., 7., -5., 9.]);
        let recycled = gpu.output(4).unwrap();
        assert_eq!(recycled.contents(), first_address);
        assert_eq!(gpu.workspace_statistics().1, 1);
        drop((other, another, recycled, final_result));
        let before = gpu.workspace_statistics().0;
        for _ in 0..100 {
            drop(gpu.output(4).unwrap());
        }
        assert_eq!(gpu.workspace_statistics().0, before);
    }
    #[test]
    fn workspace_is_bounded_and_evicts_unused_sizes() {
        let gpu = Gpu::new().unwrap();
        gpu.configure_workspace(2 * 16 * 1024);
        let first = gpu.output(4).unwrap();
        let second = gpu.output(5).unwrap();
        assert!(
            gpu.output(6).is_err(),
            "the live set must fit the reservation"
        );
        drop(first);
        let third = gpu.output(6).unwrap();
        assert_eq!(gpu.workspace_statistics().2, 2 * 16 * 1024);
        drop((second, third));
        let out = gpu.output(7).unwrap();
        let cmd = gpu.command().unwrap();
        assert!(
            gpu.op(&cmd, "invalid_operation", &[&out], &[], 1, 1)
                .is_err()
        );
        drop((out, cmd));
        let before = gpu.workspace_statistics().0;
        drop(gpu.output(7).unwrap());
        assert_eq!(gpu.workspace_statistics().0, before);
        let zero = gpu.zeros(4).unwrap();
        assert_eq!(gpu.read_f32(&zero).unwrap(), [0.; 4]);
        assert_eq!(
            gpu.workspace_statistics().0,
            before,
            "persistent state bypasses workspace"
        );
    }
}

#[cfg(test)]
mod greedy_tests {
    use super::*;
    #[test]
    fn gpu_greedy_matches_finite_host_selection_and_global_ties() {
        let gpu = Gpu::new().unwrap();
        for vocabulary in [1, 31, 1024, 1025, 248320] {
            let mut values: Vec<f32> = (0..vocabulary * 3)
                .map(|i| ((i * 193 + 71) % 997) as f32 - 500.)
                .collect();
            values[..vocabulary].fill(-f32::MAX);
            values[vocabulary..2 * vocabulary].fill(-0.);
            values[2 * vocabulary - 1] = 0.;
            values[3 * vocabulary - 1] = 1000.;
            values[2 * vocabulary] = 1000.;
            let input = gpu.floats(&values).unwrap();
            let expected: Vec<_> = values
                .chunks_exact(vocabulary)
                .map(|x| crate::engine::greedy(x).unwrap())
                .collect();
            let actual = gpu
                .finish_greedy(gpu.command().unwrap(), &input, 3, vocabulary, None)
                .unwrap();
            assert_eq!(actual, expected, "vocabulary={vocabulary}");
            let mapping: Vec<u32> = (0..vocabulary as u32).rev().map(|id| id * 2 + 7).collect();
            let map = gpu
                .bytes(
                    &mapping
                        .iter()
                        .flat_map(|id| id.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let expected: Vec<_> = values
                .chunks_exact(vocabulary)
                .map(|x| {
                    x.iter()
                        .enumerate()
                        .max_by(|(a, x), (b, y)| {
                            x.partial_cmp(y)
                                .unwrap()
                                .then_with(|| mapping[*b].cmp(&mapping[*a]))
                        })
                        .map(|(id, _)| mapping[id])
                        .unwrap()
                })
                .collect();
            assert_eq!(
                gpu.finish_greedy(gpu.command().unwrap(), &input, 3, vocabulary, Some(&map))
                    .unwrap(),
                expected
            );
        }
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut values = vec![1.; 248320];
            values[1024] = bad;
            let input = gpu.floats(&values).unwrap();
            assert!(
                gpu.finish_greedy(gpu.command().unwrap(), &input, 1, values.len(), None)
                    .is_err()
            );
        }
    }
}

#[cfg(test)]
mod compact_empty_test {
    use super::*;
    #[test]
    fn empty_compact_attention_is_finite_zero() {
        let gpu = Gpu::new().unwrap();
        let length = 33;
        let mask = gpu.zeros(length).unwrap();
        let ids = gpu.output(length).unwrap();
        let offsets = gpu.output(33).unwrap();
        let q = gpu.floats(&[1.; 32]).unwrap();
        let k = gpu.floats(&vec![2.; 32 * length]).unwrap();
        let v = gpu.floats(&vec![3.; 32 * length]).unwrap();
        let score = gpu.output(length).unwrap();
        let prob = gpu.output(length).unwrap();
        let out = gpu.output(32).unwrap();
        let cmd = gpu.command().unwrap();
        gpu.op(
            &cmd,
            "compact_mask",
            &[&mask, &ids, &offsets],
            &[length as u32, length as u32 - 1, length as u32],
            1,
            1,
        )
        .unwrap();
        gpu.op(
            &cmd,
            "attention_scores_compact",
            &[&q, &k, &ids, &offsets, &score],
            &[32, 1, 1, length as u32],
            length,
            1,
        )
        .unwrap();
        gpu.op(
            &cmd,
            "softmax_compact",
            &[&score, &offsets, &prob],
            &[length as u32, 1],
            1,
            1,
        )
        .unwrap();
        gpu.op(
            &cmd,
            "attention_values_compact",
            &[&prob, &v, &ids, &offsets, &out],
            &[32, 1, 1, length as u32],
            32,
            1,
        )
        .unwrap();
        assert_eq!(gpu.finish(cmd, &out).unwrap(), [0.; 32]);
        assert_eq!(gpu.read_u32(&offsets, 33).unwrap(), [0; 33]);
    }
}

#[cfg(test)]
mod sparse_rank_tests {
    use super::*;
    #[test]
    #[ignore = "requires Apple Silicon"]
    fn sorted_sparse_mask_matches_quadratic_rank_and_cpu_ties() {
        let gpu = Gpu::new().unwrap();
        for count in [1usize, 2, 3, 4] {
            for position in [
                0usize, 2046, 2047, 2051, 4093, 4094, 4096, 8190, 8192, 16380,
            ] {
                let length = position + count;
                let blocks = length / 4;
                let stride = blocks.max(1);
                let values: Vec<f32> = (0..count * stride)
                    .map(|i| {
                        if i % 11 == 0 {
                            0.0
                        } else {
                            ((i * 37 + 19) % 97) as f32
                        }
                    })
                    .collect();
                let scores = gpu.floats(&values).unwrap();
                let original = gpu.zeros(count * length).unwrap();
                let sorted = gpu.zeros(count * length).unwrap();
                let params = [
                    length as u32,
                    4,
                    2048,
                    position as u32,
                    count as u32,
                    stride as u32,
                ];
                let cmd = gpu.command().unwrap();
                gpu.op(
                    &cmd,
                    "index_mask_batch",
                    &[&scores, &original],
                    &params,
                    (stride * count).div_ceil(32),
                    1,
                )
                .unwrap();
                gpu.op(
                    &cmd,
                    "index_mask_sorted",
                    &[&scores, &sorted],
                    &params,
                    count,
                    1,
                )
                .unwrap();
                gpu.complete(cmd).unwrap();
                let actual = gpu.read_f32(&sorted).unwrap();
                assert_eq!(
                    actual,
                    gpu.read_f32(&original).unwrap(),
                    "position={position}"
                );
                for t in 0..count {
                    let len = position + t + 1;
                    let mut ids: Vec<_> = (0..len / 4).collect();
                    ids.sort_by(|&a, &b| {
                        values[t * stride + b]
                            .total_cmp(&values[t * stride + a])
                            .then(a.cmp(&b))
                    });
                    let mut expected = vec![0f32; length];
                    expected[len / 4 * 4..len].fill(1.);
                    for (rank, b) in ids.into_iter().enumerate() {
                        for j in 0..4 {
                            expected[b * 4 + j] = f32::from(rank * 4 + j + len % 4 < 2051);
                        }
                    }
                    assert_eq!(
                        &actual[t * length..(t + 1) * length],
                        &expected,
                        "CPU position={position} row={t}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod online_attention_experiment {
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore = "Apple GPU experiment; no model loaded"]
    fn online_attention_shapes_and_timing() {
        let gpu = Gpu::new().unwrap();
        let mut seed = 19u64;
        let mut values = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((seed >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * scale
                })
                .collect()
        };
        for (position, count, compact, scale) in [
            (0usize, 63usize, false, 2f32),
            (0, 65, false, 2.),
            (0, 127, true, 2.),
            (4087, 8, true, 2.),
            (4087, 8, false, 16.),
            (0, 1023, false, 2.),
            (0, 4095, true, 2.),
            (0, 1024, false, 2.),
            (55, 9, false, 2.),
            (2046, 1, false, 2.),
            (2044, 4, false, 16.),
            (4096, 128, true, 2.),
            (4096, 1024, true, 2.),
            (8190, 1, true, 2.),
            (8188, 4, true, 16.),
        ] {
            let length = position + count;
            let capacity = if compact { 2051.min(length) } else { length };
            let parts = capacity.div_ceil(64);
            let qh = values(count * 24 * 256, scale);
            let kh = values(length * 2 * 256, 2.);
            let vh = values(length * 2 * 256, 2.);
            let mut ih = vec![0u32; count * capacity];
            let mut oh = vec![0u32; count * 33];
            let mut selected = Vec::new();
            for t in 0..count {
                let n = position + t + 1;
                let ids: Vec<usize> = (0..n.min(capacity))
                    .map(|j| j * n / n.min(capacity))
                    .collect();
                let mut at = 0;
                for lane in 0..32 {
                    oh[t * 33 + lane] = at as u32;
                    for &id in &ids {
                        if id % 32 == lane {
                            ih[t * capacity + at] = id as u32;
                            at += 1;
                        }
                    }
                }
                oh[t * 33 + 32] = at as u32;
                selected.push(ids);
            }
            let mut mask = vec![0.; count * length];
            for (t, ids) in selected.iter().enumerate() {
                for &id in ids {
                    mask[t * length + id] = 1.;
                }
            }
            let mask = gpu.floats(&mask).unwrap();
            let q = gpu.floats(&qh).unwrap();
            let mut kp = kh.clone();
            kp.extend(vec![f32::NAN; 64 * 2 * 256]);
            let k = gpu.floats(&kp).unwrap();
            let mut vp = vh.clone();
            vp.extend(vec![f32::NAN; 64 * 2 * 256]);
            let v = gpu.floats(&vp).unwrap();
            let indices = gpu
                .bytes(&ih.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
                .unwrap();
            let offsets = gpu
                .bytes(&oh.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
                .unwrap();
            let score = gpu.output(count * 24 * capacity).unwrap();
            let prob = gpu.output(count * 24 * capacity).unwrap();
            let partial = gpu.output(count * 24 * parts * 258).unwrap();
            let old = gpu.output(count * 24 * 256).unwrap();
            let new = gpu.output(count * 24 * 256).unwrap();
            let p = [
                256,
                24,
                2,
                capacity as u32,
                position as u32,
                u32::from(compact),
                parts as u32,
            ];
            let mut samples = [Vec::new(), Vec::new()];
            for iteration in 0..3 {
                let order = if iteration % 2 == 0 { [0, 1] } else { [1, 0] };
                for arm in order {
                    let start = Instant::now();
                    let cmd = gpu.command().unwrap();
                    if arm == 0 {
                        if compact {
                            gpu.op(
                                &cmd,
                                "attention_scores_compact",
                                &[&q, &k, &indices, &offsets, &score],
                                &p[..4],
                                capacity,
                                24 * count,
                            )
                            .unwrap();
                            gpu.op(
                                &cmd,
                                "softmax_compact",
                                &[&score, &offsets, &prob],
                                &[capacity as u32, 24],
                                24 * count,
                                1,
                            )
                            .unwrap();
                            gpu.op(
                                &cmd,
                                "attention_values_compact",
                                &[&prob, &v, &indices, &offsets, &old],
                                &p[..4],
                                256,
                                24 * count,
                            )
                            .unwrap();
                        } else {
                            gpu.op(
                                &cmd,
                                "attention_scores_dense_batch",
                                &[&q, &k, &score],
                                &p[..5],
                                capacity,
                                24 * count,
                            )
                            .unwrap();
                            gpu.op(
                                &cmd,
                                "softmax_dense_batch",
                                &[&score, &prob],
                                &[capacity as u32, 24, position as u32],
                                24 * count,
                                1,
                            )
                            .unwrap();
                            gpu.op(
                                &cmd,
                                "attention_values_dense_batch",
                                &[&prob, &v, &old],
                                &p[..5],
                                256,
                                24 * count,
                            )
                            .unwrap();
                        }
                    } else if count >= 8 {
                        let full = (length / 64 * 64).saturating_sub(position).min(count);
                        for (begin, rows, tail) in [(0, full, false), (full, count - full, true)] {
                            if rows == 0 {
                                continue;
                            }
                            let query = q.view_f32(begin * 24 * 256, rows * 24 * 256).unwrap();
                            let result = new.view_f32(begin * 24 * 256, rows * 24 * 256).unwrap();
                            let mask = mask.view_f32(begin * length, rows * length).unwrap();
                            gpu.op(
                                &cmd,
                                if tail {
                                    "attention_prefill_tail"
                                } else {
                                    "attention_prefill_tiles"
                                },
                                &[&query, &k, &v, &mask, &result],
                                &[
                                    256,
                                    24,
                                    2,
                                    length as u32,
                                    (position + begin) as u32,
                                    rows as u32,
                                    u32::from(compact),
                                ],
                                rows.div_ceil(8) * 4,
                                24,
                            )
                            .unwrap();
                        }
                    } else {
                        gpu.op(
                            &cmd,
                            "attention_online_parts",
                            &[&q, &k, &v, &indices, &offsets, &partial],
                            &p,
                            parts,
                            24 * count,
                        )
                        .unwrap();
                        gpu.op(
                            &cmd,
                            "attention_online_merge",
                            &[&partial, &new],
                            &[parts as u32],
                            24 * count,
                            1,
                        )
                        .unwrap();
                    }
                    gpu.submit(&cmd);
                    cmd.waitUntilCompleted();
                    let gpu_s = cmd.GPUEndTime() - cmd.GPUStartTime();
                    gpu.wait(cmd).unwrap();
                    if iteration > 0 {
                        samples[arm].push([start.elapsed().as_secs_f64(), gpu_s]);
                    }
                }
            }
            let a = gpu.read_f32(&old).unwrap();
            let b = gpu.read_f32(&new).unwrap();
            assert!(a.iter().chain(&b).all(|x| x.is_finite()));
            let error = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
                .sum::<f64>();
            let energy = a.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
            let relative = (error / energy.max(1e-30)).sqrt();
            assert!(relative < 2e-5, "online relative L2 {relative}");
            let mut cpu_errors = [0f64; 2];
            let mut cpu_energy = 0.;
            for row in [0, (count * 24) / 2, count * 24 - 1] {
                let t = row / 24;
                let head = row % 24;
                let kv = head / 12;
                let ids = &selected[t];
                let scores = ids
                    .iter()
                    .map(|&id| {
                        (0..256)
                            .map(|j| {
                                f64::from(qh[row * 256 + j])
                                    * f64::from(kh[(id * 2 + kv) * 256 + j])
                            })
                            .sum::<f64>()
                            / 16.
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
                let sum = weights.iter().sum::<f64>();
                for j in 0..256 {
                    let reference = ids
                        .iter()
                        .zip(&weights)
                        .map(|(&id, &w)| w * f64::from(vh[(id * 2 + kv) * 256 + j]))
                        .sum::<f64>()
                        / sum;
                    cpu_energy += reference * reference;
                    cpu_errors[0] += (f64::from(a[row * 256 + j]) - reference).powi(2);
                    cpu_errors[1] += (f64::from(b[row * 256 + j]) - reference).powi(2);
                }
            }
            let cpu_relative = cpu_errors.map(|x| (x / cpu_energy.max(1e-30)).sqrt());
            assert!(cpu_relative.iter().all(|x| *x < 2e-5));
            println!(
                "ONLINE {{\"position\":{position},\"count\":{count},\"compact\":{compact},\"relative_l2\":{relative},\"cpu_relative_l2\":{cpu_relative:?},\"old_wall_gpu_s\":{:?},\"online_wall_gpu_s\":{:?}}}",
                samples[0], samples[1]
            );
        }
    }
}

#[cfg(test)]
mod ordered_mask_test {
    use super::*;
    #[test]
    fn ordered_compaction_preserves_causal_positions() {
        let gpu = Gpu::new().unwrap();
        for position in [0usize, 1, 29, 31, 2046, 2047, 2050, 8190] {
            let count = 3;
            let length = position + count;
            let mut mask = vec![0f32; length * count];
            for t in 0..count {
                for j in 0..=position + t {
                    if t == 0 || (t == 1 && j % 7 != 0) {
                        mask[t * length + j] = 1.;
                    }
                }
            }
            let input = gpu.floats(&mask).unwrap();
            let ids = gpu.output(length * count).unwrap();
            let offsets = gpu.output(33 * count).unwrap();
            let cmd = gpu.command().unwrap();
            gpu.op(
                &cmd,
                "compact_mask_ordered",
                &[&input, &ids, &offsets],
                &[length as u32, position as u32, length as u32],
                count,
                1,
            )
            .unwrap();
            gpu.complete(cmd).unwrap();
            let ids = gpu.read_u32(&ids, length * count).unwrap();
            let offsets = gpu.read_u32(&offsets, 33 * count).unwrap();
            for t in 0..count {
                let expected = (0..=position + t)
                    .filter(|&j| mask[t * length + j] > 0.)
                    .map(|j| j as u32)
                    .collect::<Vec<_>>();
                assert_eq!(offsets[t * 33 + 32] as usize, expected.len());
                assert_eq!(
                    &ids[t * length..t * length + expected.len()],
                    expected.as_slice()
                );
            }
        }
    }
}

#[cfg(test)]
mod mapped_weight_tests {
    use super::*;
    #[test]
    fn mapped_weights_keep_tensor_offsets_and_inflight_lifetime() {
        // The tensor begins inside a page, crosses two page boundaries, and ends
        // in the final partial file page. Its values must not include the header.
        let path = std::env::temp_dir().join(format!("ingot-mapped-{}.gguf", std::process::id()));
        let values: Vec<f32> = (0..8192).map(|i| (i - 4096) as f32 * 0.125).collect();
        let mut data = Vec::new();
        let text = |out: &mut Vec<u8>, s: &str| {
            out.extend((s.len() as u64).to_le_bytes());
            out.extend(s.as_bytes());
        };
        data.extend(0x4655_4747u32.to_le_bytes());
        data.extend(3u32.to_le_bytes());
        data.extend(1u64.to_le_bytes());
        data.extend(1u64.to_le_bytes());
        text(&mut data, "general.architecture");
        data.extend(8u32.to_le_bytes());
        text(&mut data, "fixture");
        text(&mut data, "weight");
        data.extend(2u32.to_le_bytes());
        data.extend(4096u64.to_le_bytes());
        data.extend(2u64.to_le_bytes());
        data.extend(0u32.to_le_bytes());
        data.extend(0u64.to_le_bytes());
        data.resize(data.len().div_ceil(32) * 32, 0);
        for v in &values {
            data.extend(v.to_le_bytes());
        }
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.write_all(&data).unwrap();
        drop(file);
        let mut g = crate::gguf::Gguf::open(&path).unwrap();
        let t = g.tensor("weight").unwrap().clone();
        let gpu = Gpu::new().unwrap();
        let weight = gpu.load_weight(&mut g, &t).unwrap();
        assert_eq!(
            g.bytes_read, 0,
            "mapped weights must not issue pread copies"
        );
        assert_eq!(weight.length(), values.len() * 4);
        assert!(weight.0.offset > 0);
        assert!(weight.0.raw.as_ref().unwrap().length() > weight.length());
        assert_eq!(gpu.read_f32(&weight).unwrap(), values);
        let alive = Rc::downgrade(&weight.0);
        let command = gpu.command().unwrap();
        let out = gpu.output(values.len()).unwrap();
        gpu.op(
            &command,
            "copy_range",
            &[&weight, &out],
            &[values.len() as u32, 0],
            values.len(),
            1,
        )
        .unwrap();
        drop(weight);
        drop(g);
        assert!(
            alive.upgrade().is_some(),
            "encoded command owns mapped storage"
        );
        gpu.submit(&command);
        drop(command); // Early return after submit must also drain before unmapping.
        assert!(alive.upgrade().is_none());
        assert_eq!(gpu.read_f32(&out).unwrap(), values);
        let mut g = crate::gguf::Gguf::open(&path).unwrap();
        let weight = gpu.load_weight(&mut g, &t).unwrap();
        let alive = Rc::downgrade(&weight.0);
        let abandoned = gpu.command().unwrap();
        gpu.op(
            &abandoned,
            "copy_range",
            &[&weight, &out],
            &[values.len() as u32, 0],
            values.len(),
            1,
        )
        .unwrap();
        drop(weight);
        drop(g);
        assert!(alive.upgrade().is_some());
        drop(abandoned);
        assert!(
            alive.upgrade().is_none(),
            "unsubmitted command releases mappings"
        );
        std::fs::remove_file(path).unwrap();
    }
}

#[cfg(test)]
mod glm_compact_tests {
    use super::*;
    #[test]
    #[ignore = "explicit Apple GPU operator validation"]
    fn glm_sparse_selection_and_mla_keep_full_path_results() {
        let gpu = Gpu::new().unwrap();
        let (dim, heads, qk, pool, topk) = (512usize, 4usize, 128usize, 4usize, 2048usize);
        for length in [
            1usize, 4, 31, 2047, 2048, 2051, 2112, 8192, 8195, 16384, 16387,
        ] {
            let blocks = length / pool;
            // Deliberate ties and negative scores exercise stable block-ID ordering.
            let scores: Vec<_> = (0..blocks.max(1))
                .map(|i| ((i * 29 % 61) as f32 - 30.) / 8.)
                .collect();
            let ranking = gpu.floats(&scores).unwrap();
            let mask = gpu.output(length).unwrap();
            let sorted = gpu.output(length).unwrap();
            let cmd = gpu.command().unwrap();
            let params = [length as u32, pool as u32, topk as u32];
            gpu.op(
                &cmd,
                "glm_index_mask",
                &[&ranking, &mask],
                &params,
                blocks.max(1).div_ceil(32),
                1,
            )
            .unwrap();
            gpu.op(
                &cmd,
                "glm_index_mask_sorted",
                &[&ranking, &sorted],
                &params,
                1,
                1,
            )
            .unwrap();
            gpu.complete(cmd).unwrap();
            let mh = gpu.read_f32(&mask).unwrap();
            assert_eq!(
                mh,
                gpu.read_f32(&sorted).unwrap(),
                "pool selection length={length}"
            );
            let mut selected: Vec<_> = (0..blocks).collect();
            selected.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b)));
            selected.truncate(topk / pool);
            for (t, &visible) in mh.iter().enumerate() {
                assert_eq!(
                    visible > 0.,
                    t >= blocks * pool || selected.contains(&(t / pool))
                );
            }
            let capacity = length.min(topk + length % pool);
            let ids = gpu.output(capacity).unwrap();
            let offsets = gpu.output(33).unwrap();
            let data = |n: usize, salt: usize| {
                (0..n)
                    .map(|i| ((i * 17 + salt) % 127) as f32 / 127. - 0.5)
                    .collect::<Vec<_>>()
            };
            let qh = data(dim * heads, 3);
            let kh = data(dim * length, 5);
            let q = gpu.floats(&qh).unwrap();
            let latent = gpu.floats(&kh).unwrap();
            let dense_scores = gpu.output(length * heads).unwrap();
            let dense_prob = gpu.output(length * heads).unwrap();
            let dense = gpu.output(dim * heads).unwrap();
            let compact_scores = gpu.output(capacity * heads).unwrap();
            let compact_prob = gpu.output(capacity * heads).unwrap();
            let compact = gpu.output(dim * heads).unwrap();
            let cmd = gpu.command().unwrap();
            gpu.op(
                &cmd,
                "compact_mask",
                &[&sorted, &ids, &offsets],
                &[length as u32, (length - 1) as u32, capacity as u32],
                1,
                1,
            )
            .unwrap();
            gpu.op(
                &cmd,
                "glm_mla_scores",
                &[&q, &latent, &mask, &dense_scores],
                &[dim as u32, length as u32, qk as u32],
                length,
                heads,
            )
            .unwrap();
            gpu.op(
                &cmd,
                "softmax_rows",
                &[&dense_scores, &dense_prob],
                &[length as u32],
                heads,
                1,
            )
            .unwrap();
            gpu.op(
                &cmd,
                "attention_values",
                &[&dense_prob, &latent, &dense],
                &[dim as u32, heads as u32, 1, length as u32],
                dim,
                heads,
            )
            .unwrap();
            gpu.op(
                &cmd,
                "glm_mla_scores_compact",
                &[&q, &latent, &ids, &offsets, &compact_scores],
                &[dim as u32, capacity as u32, qk as u32],
                capacity,
                heads,
            )
            .unwrap();
            gpu.op(
                &cmd,
                "softmax_compact",
                &[&compact_scores, &offsets, &compact_prob],
                &[capacity as u32, heads as u32],
                heads,
                1,
            )
            .unwrap();
            gpu.op(
                &cmd,
                "attention_values_compact",
                &[&compact_prob, &latent, &ids, &offsets, &compact],
                &[dim as u32, heads as u32, 1, capacity as u32],
                dim,
                heads,
            )
            .unwrap();
            gpu.complete(cmd).unwrap();
            let expected = gpu.read_f32(&dense).unwrap();
            let actual = gpu.read_f32(&compact).unwrap();
            assert_eq!(expected, actual, "MLA length={length}");
            let mut max_error = 0f64;
            for h in 0..heads {
                let visible: Vec<_> = (0..length).filter(|&t| mh[t] > 0.).collect();
                let logits: Vec<_> = visible
                    .iter()
                    .map(|&t| {
                        (0..dim)
                            .map(|j| f64::from(qh[h * dim + j]) * f64::from(kh[t * dim + j]))
                            .sum::<f64>()
                            / (qk as f64).sqrt()
                    })
                    .collect();
                let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let probs: Vec<_> = logits.iter().map(|x| (x - max).exp()).collect();
                let sum = probs.iter().sum::<f64>();
                for j in 0..dim {
                    let reference = visible
                        .iter()
                        .zip(&probs)
                        .map(|(&t, &p)| p * f64::from(kh[t * dim + j]))
                        .sum::<f64>()
                        / sum;
                    max_error = max_error.max((f64::from(actual[h * dim + j]) - reference).abs());
                }
            }
            assert!(
                max_error < 1e-5,
                "MLA F64 error={max_error} length={length}"
            );
            println!(
                "PASS GLM compact length={length} selected={capacity} exact mask/MLA and F64 maxabs={max_error}"
            );
        }
    }
}

#[cfg(test)]
mod glm_online_tests {
    use super::*;

    #[test]
    #[ignore = "explicit Apple GPU operator validation"]
    fn glm_online_mla_matches_f64_and_selected_keys() {
        let gpu = Gpu::new().unwrap();
        let width = 512usize;
        let mut seed = 0x2d59_67eau32;
        let mut random = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((seed >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0
        };
        for (length, heads, scale, empty) in [
            (1usize, 64usize, 1f32, false),
            (63, 4, 1., false),
            (64, 64, 8., false),
            (65, 4, 1., false),
            (2047, 4, 1., false),
            (2048, 64, 1., false),
            (2051, 4, 1., false),
            (8192, 64, 1., false),
            (8195, 4, 8., false),
            (65, 4, 1., true),
        ] {
            let query: Vec<f32> = (0..heads * width).map(|_| random() * scale).collect();
            let cache: Vec<f32> = (0..length * width).map(|_| random()).collect();
            let mut ranking: Vec<usize> = (0..length / 4).collect();
            ranking.sort_by_key(|&i| (std::cmp::Reverse(i * 29 % 71), i));
            let mut selected = vec![0f32; length];
            if !empty {
                for &pool in ranking.iter().take(512) {
                    selected[pool * 4..pool * 4 + 4].fill(1.);
                }
                selected[(length / 4) * 4..].fill(1.);
            }
            let selected_count = selected.iter().filter(|&&x| x > 0.).count();
            let capacity = selected_count.max(1);
            let parts = capacity.div_ceil(64);
            let q = gpu.floats(&query).unwrap();
            let latent = gpu.floats(&cache).unwrap();
            let mask = gpu.floats(&selected).unwrap();
            let ordered = gpu.output(capacity).unwrap();
            let ordered_offsets = gpu.output(33).unwrap();
            let old_ids = gpu.output(capacity).unwrap();
            let old_offsets = gpu.output(33).unwrap();
            let partial = gpu.output(heads * parts * 514).unwrap();
            let output = gpu.output(heads * width).unwrap();
            let old_scores = gpu.output(heads * capacity).unwrap();
            let old_prob = gpu.output(heads * capacity).unwrap();
            let old_output = gpu.output(heads * width).unwrap();
            let cmd = gpu.command().unwrap();
            for (kernel, ids, offsets) in [
                ("compact_mask_ordered", &ordered, &ordered_offsets),
                ("compact_mask", &old_ids, &old_offsets),
            ] {
                gpu.op(
                    &cmd,
                    kernel,
                    &[&mask, ids, offsets],
                    &[length as u32, (length - 1) as u32, capacity as u32],
                    1,
                    1,
                )
                .unwrap();
            }
            gpu.op(
                &cmd,
                "glm_mla_online_parts",
                &[&q, &latent, &ordered, &ordered_offsets, &partial],
                &[length as u32, parts as u32, 1],
                parts,
                heads,
            )
            .unwrap();
            gpu.op(
                &cmd,
                "glm_mla_online_merge",
                &[&partial, &output],
                &[parts as u32],
                heads,
                1,
            )
            .unwrap();
            if !empty {
                gpu.op(
                    &cmd,
                    "glm_mla_scores_compact",
                    &[&q, &latent, &old_ids, &old_offsets, &old_scores],
                    &[width as u32, capacity as u32, 256],
                    capacity,
                    heads,
                )
                .unwrap();
                gpu.op(
                    &cmd,
                    "softmax_compact",
                    &[&old_scores, &old_offsets, &old_prob],
                    &[capacity as u32, heads as u32],
                    heads,
                    1,
                )
                .unwrap();
                gpu.op(
                    &cmd,
                    "attention_values_compact",
                    &[&old_prob, &latent, &old_ids, &old_offsets, &old_output],
                    &[width as u32, heads as u32, 1, capacity as u32],
                    width,
                    heads,
                )
                .unwrap();
            }
            gpu.complete(cmd).unwrap();
            let actual = gpu.read_f32(&output).unwrap();
            assert!(actual.iter().all(|x| x.is_finite()));
            let ordered_ids = gpu.read_u32(&ordered, selected_count).unwrap();
            let expected_ids: Vec<u32> = selected
                .iter()
                .enumerate()
                .filter_map(|(i, &x)| (x > 0.).then_some(i as u32))
                .collect();
            assert_eq!(ordered_ids, expected_ids);
            let mut max_abs = 0f64;
            let mut squared_error = 0f64;
            let mut squared_reference = 0f64;
            for head in 0..heads {
                let scores: Vec<f64> = expected_ids
                    .iter()
                    .map(|&pos| {
                        query[head * width..(head + 1) * width]
                            .iter()
                            .zip(&cache[pos as usize * width..(pos as usize + 1) * width])
                            .map(|(&a, &b)| f64::from(a) * f64::from(b))
                            .sum::<f64>()
                            / 16.0
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|&x| (x - maximum).exp()).collect();
                let denominator: f64 = weights.iter().sum();
                for col in 0..width {
                    let expected = if empty {
                        0.
                    } else {
                        expected_ids
                            .iter()
                            .zip(&weights)
                            .map(|(&pos, &weight)| {
                                f64::from(cache[pos as usize * width + col]) * weight
                            })
                            .sum::<f64>()
                            / denominator
                    };
                    let error = f64::from(actual[head * width + col]) - expected;
                    max_abs = max_abs.max(error.abs());
                    squared_error += error * error;
                    squared_reference += expected * expected;
                }
            }
            let relative = (squared_error / squared_reference.max(1e-30)).sqrt();
            assert!(
                max_abs < 2e-4 && relative < 3e-5,
                "{length}: {max_abs} {relative}"
            );
            if !empty {
                let previous = gpu.read_f32(&old_output).unwrap();
                assert!(previous.iter().all(|x| x.is_finite()));
                let old_difference = actual
                    .iter()
                    .zip(previous)
                    .map(|(&a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                println!(
                    "GLM_ONLINE length={length} heads={heads} selected={selected_count} scale={scale} max_abs_f64={max_abs} relative_f64={relative} max_abs_old={old_difference}"
                );
            }
            if selected_count == length {
                let dense_partial = gpu.output(heads * parts * 514).unwrap();
                let dense = gpu.output(heads * width).unwrap();
                let cmd = gpu.command().unwrap();
                gpu.op(
                    &cmd,
                    "glm_mla_online_parts",
                    &[&q, &latent, &ordered, &ordered_offsets, &dense_partial],
                    &[length as u32, parts as u32, 0],
                    parts,
                    heads,
                )
                .unwrap();
                gpu.op(
                    &cmd,
                    "glm_mla_online_merge",
                    &[&dense_partial, &dense],
                    &[parts as u32],
                    heads,
                    1,
                )
                .unwrap();
                gpu.complete(cmd).unwrap();
                assert_eq!(actual, gpu.read_f32(&dense).unwrap());
            }
        }
    }
}

// Small development screen; not task-quality acceptance.

#[cfg(test)]
mod routed_rows_probe {
    use super::*;
    #[test]
    #[ignore = "requires QWEN_MODEL; real expert rows, mixed routes and F64 tails"]
    fn real_routed_rows() {
        let gpu = Gpu::new().unwrap();
        let mut g =
            crate::gguf::Gguf::open(std::path::Path::new(&std::env::var("QWEN_MODEL").unwrap()))
                .unwrap();
        for name in [
            "blk.0.ffn_gate_exps.weight",
            "blk.1.ffn_gate_exps.weight",
            "blk.0.ffn_down_exps.weight",
            "blk.8.ffn_gate_exps.weight",
        ] {
            let t = g.tensor(name).unwrap().clone();
            for rows in [65, t.dims[1]] {
                let experts = if rows == 65 { 2 } else { 32 };
                let matrix = g
                    .rows(&gpu.reads, name, &(0..rows * experts).collect::<Vec<_>>())
                    .unwrap();
                let w = gpu.bytes(&matrix.data).unwrap();
                let decoded = if rows == 65 {
                    gpu.dequant(&matrix).unwrap()
                } else {
                    Vec::new()
                };
                for count in [1, 4] {
                    for per_pair in [false, true] {
                        let top = 10;
                        let pairs = top * count;
                        let ids = (0..pairs)
                            .map(|i| ((i * 7 + i / top) % experts) as u32)
                            .collect::<Vec<_>>();
                        let routes = gpu
                            .bytes(&ids.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<_>>())
                            .unwrap();
                        let input = (0..matrix.cols * if per_pair { pairs } else { count })
                            .map(|i| ((i * 17 % 31) as f32 - 15.) * 0.03125)
                            .collect::<Vec<_>>();
                        let x = gpu.floats(&input).unwrap();
                        let out = gpu.zeros(rows * pairs).unwrap();
                        let params = [
                            matrix.cols as u32,
                            rows as u32,
                            matrix.stride as u32,
                            matrix.kind,
                            top as u32,
                            per_pair as u32,
                        ];
                        let mut samples = Vec::new();
                        for round in 0..if rows == 65 { 1 } else { 20 } {
                            let start = std::time::Instant::now();
                            let cmd = gpu.command().unwrap();
                            for _ in 0..if rows == 65 { 1 } else { 8 } {
                                gpu.op(
                                    &cmd,
                                    "project_routed_batch",
                                    &[&w, &x, &routes, &out],
                                    &params,
                                    rows,
                                    pairs,
                                )
                                .unwrap();
                            }
                            cmd.commit();
                            cmd.waitUntilCompleted();
                            assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
                            if round >= 4 {
                                samples.push((
                                    start.elapsed().as_secs_f64(),
                                    cmd.GPUEndTime() - cmd.GPUStartTime(),
                                ));
                            }
                        }
                        let actual = gpu.read_f32(&out).unwrap();
                        if rows == 65 {
                            for pair in 0..pairs {
                                for row in 0..rows {
                                    let off = (ids[pair] as usize * rows + row) * matrix.cols;
                                    let input_row = if per_pair { pair } else { pair / top };
                                    let expected = decoded[off..off + matrix.cols]
                                        .iter()
                                        .zip(
                                            &input[input_row * matrix.cols
                                                ..(input_row + 1) * matrix.cols],
                                        )
                                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                                        .sum::<f64>();
                                    assert!(
                                        (f64::from(actual[pair * rows + row]) - expected).abs()
                                            <= 1e-4 + 5e-5 * expected.abs(),
                                        "{name} count{count} row{row} pair{pair}"
                                    );
                                }
                            }
                        } else {
                            println!(
                                "ROUTED_BENCH name={name} kind={} cols={} count={count} per_pair={per_pair} samples={samples:?}",
                                matrix.kind, matrix.cols
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod dense_mps_probe {
    use super::*;
    #[test]
    #[ignore = "real GGUF dense MPS decode plus GEMM F64 comparison"]
    fn real_dense_mps_f64() {
        objc2::rc::autoreleasepool(|_| {
            let gpu = Gpu::new().unwrap();
            let mut g = crate::gguf::Gguf::open(std::path::Path::new(
                &std::env::var("QWEN_MODEL").unwrap(),
            ))
            .unwrap();
            let mut shapes = std::collections::BTreeMap::new();
            for (name, t) in &g.tensors {
                if t.dims.len() == 2
                    && ((t.kind == 21 && matches!(t.dims[0], 2560 | 6144) && t.dims[1] >= 2560)
                        || (t.kind == 30
                            && ((t.dims[0] == 10240 && t.dims[1] == 320)
                                || (t.dims[0] == 320 && t.dims[1] == 10240))))
                {
                    shapes
                        .entry((t.kind, t.dims.clone()))
                        .or_insert(name.clone());
                }
            }
            for name in shapes.values() {
                let t = g.tensor(name).unwrap().clone();
                for rows in [65, t.dims[1]] {
                    let m = g
                        .rows(&gpu.reads, name, &(0..rows).collect::<Vec<_>>())
                        .unwrap();
                    let decoded = gpu.dequant(&m).unwrap();
                    let w = gpu.bytes(&m.data).unwrap();
                    for count in [4095, 4096] {
                        let input = (0..m.cols * count)
                            .map(|i| ((i * 17 % 31) as f32 - 15.) * 0.03125)
                            .collect::<Vec<_>>();
                        let x = gpu.floats(&input).unwrap();
                        let cmd = gpu.command().unwrap();
                        let out = gpu
                            .dense_mps(&cmd, &w, m.cols, rows, m.stride, m.kind, &x, count)
                            .unwrap();
                        // Poison the result before submission to test beta=0 with reused storage.
                        unsafe {
                            std::slice::from_raw_parts_mut(
                                out.contents().as_ptr().cast::<f32>(),
                                out.length() / 4,
                            )
                            .fill(f32::NAN);
                        }
                        let actual = gpu.finish(cmd, &out).unwrap();
                        assert!(
                            actual.iter().all(|x| x.is_finite()),
                            "nonfinite MPS {name}/{rows}/{count}"
                        );
                        let mut maximum = 0f64;
                        for token in [0, 1, count / 2, count - 2, count - 1] {
                            for row in [0, 1, rows / 2, rows - 2, rows - 1] {
                                let expected = decoded[row * m.cols..(row + 1) * m.cols]
                                    .iter()
                                    .zip(&input[token * m.cols..(token + 1) * m.cols])
                                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                                    .sum::<f64>();
                                let error =
                                    (f64::from(actual[token * rows + row]) - expected).abs();
                                maximum = maximum.max(error);
                                assert!(
                                    error <= 1e-4 + 5e-5 * expected.abs(),
                                    "{name} rows{rows} count{count} token{token} row{row}: error{error}"
                                );
                            }
                        }
                        println!(
                            "DENSE_MPS_F64 name={name} rows={rows} count={count} max_abs={maximum}"
                        );
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod reduction_probe {
    use super::*;
    #[test]
    #[ignore = "F64 normalization, exact routing/ties, interleaved GPU timing"]
    fn real_reduction_shapes() {
        let gpu = Gpu::new().unwrap();
        for count in [1usize, 4, 1024] {
            let values = (0..count * 512)
                .map(|i| ((i * 17 % 31) as f32 - 15.) * 0.25)
                .collect::<Vec<_>>();
            let x = gpu.floats(&values).unwrap();
            let ids = gpu.output(count * 10).unwrap();
            let weights = gpu.output(count * 10).unwrap();
            let mut expected_ids = Vec::new();
            let mut expected_weights = Vec::new();
            for row in values.chunks(512) {
                let mut order = (0..512).collect::<Vec<_>>();
                order
                    .sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then_with(|| a.cmp(&b)));
                order.truncate(10);
                let sum: f64 = order
                    .iter()
                    .map(|&i| f64::from(row[i] - row[order[0]]).exp())
                    .sum();
                expected_weights.extend(
                    order
                        .iter()
                        .map(|&i| f64::from(row[i] - row[order[0]]).exp() / sum),
                );
                expected_ids.extend(order.into_iter().map(|i| i as u32));
            }
            for (block, order) in [[false, true], [true, false]].iter().enumerate() {
                for &candidate in order {
                    let mut samples = Vec::new();
                    for rep in 0..20 {
                        let cmd = gpu.command().unwrap();
                        for _ in 0..8 {
                            gpu.encode(
                                &cmd,
                                &gpu.operations[if candidate {
                                    "route_top10"
                                } else {
                                    "route_gpu_batch"
                                }],
                                &[&x, &ids, &weights],
                                &[512, 10],
                                MTLSize {
                                    width: if candidate { count * 4 } else { count },
                                    height: 1,
                                    depth: 1,
                                },
                                if candidate { 4 } else { 1 },
                            )
                            .unwrap();
                        }
                        gpu.submit(&cmd);
                        cmd.waitUntilCompleted();
                        let elapsed = cmd.GPUEndTime() - cmd.GPUStartTime();
                        gpu.wait(cmd).unwrap();
                        if rep >= 4 {
                            samples.push(elapsed);
                        }
                    }
                    assert_eq!(gpu.read_u32(&ids, count * 10).unwrap(), expected_ids);
                    for (&a, &b) in gpu
                        .read_f32(&weights)
                        .unwrap()
                        .iter()
                        .zip(&expected_weights)
                    {
                        assert!((f64::from(a) - b).abs() < 1e-6);
                    }
                    println!(
                        "REDUCTION name=route count={count} block={block} candidate={candidate} samples={samples:?}"
                    );
                }
            }
        }
        for n in [2560usize, 10240] {
            for count in [4usize, 16, 4096] {
                let values = (0..n * count)
                    .map(|i| ((i * 17 % 31) as f32 - 15.) * 0.25)
                    .collect::<Vec<_>>();
                let weights = (0..n)
                    .map(|i| 1. + (i % 7) as f32 * 0.015625)
                    .collect::<Vec<_>>();
                let x = gpu.floats(&values).unwrap();
                let w = gpu.floats(&weights).unwrap();
                let y = gpu.output(n * count).unwrap();
                for (block, order) in [[false, true], [true, false]].iter().enumerate() {
                    for &candidate in order {
                        let mut samples = Vec::new();
                        for rep in 0..20 {
                            let cmd = gpu.command().unwrap();
                            for _ in 0..8 {
                                gpu.encode(
                                    &cmd,
                                    &gpu.operations[if candidate { "rms_wide" } else { "rms" }],
                                    &[&x, &w, &y],
                                    &[n as u32, n as u32, 0, n as u32, 1e-6f32.to_bits(), 0],
                                    MTLSize {
                                        width: if candidate { count * 4 } else { count },
                                        height: 1,
                                        depth: 1,
                                    },
                                    if candidate { 4 } else { 1 },
                                )
                                .unwrap();
                            }
                            gpu.submit(&cmd);
                            cmd.waitUntilCompleted();
                            let elapsed = cmd.GPUEndTime() - cmd.GPUStartTime();
                            gpu.wait(cmd).unwrap();
                            if rep >= 4 {
                                samples.push(elapsed);
                            }
                        }
                        let actual = gpu.read_f32(&y).unwrap();
                        for row in [0, count - 1] {
                            let sum: f64 = values[row * n..(row + 1) * n]
                                .iter()
                                .map(|&x| f64::from(x).powi(2))
                                .sum();
                            let inv = (sum / n as f64 + 1e-6f64).sqrt().recip();
                            for j in 0..n {
                                let expected =
                                    f64::from(values[row * n + j]) * inv * f64::from(weights[j]);
                                assert!((f64::from(actual[row * n + j]) - expected).abs() < 2e-5);
                            }
                        }
                        println!(
                            "REDUCTION name=rms{n} count={count} block={block} candidate={candidate} samples={samples:?}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod residency_probe {
    use super::*;
    #[test]
    #[ignore = "real mapped model weights and Metal residency accounting"]
    fn mapped_weights_are_resident_and_owned() {
        let gpu = Gpu::new().unwrap();
        let mut g =
            crate::gguf::Gguf::open(std::path::Path::new(&std::env::var("QWEN_MODEL").unwrap()))
                .unwrap();
        let buffers: Vec<_> = ["blk.0.attn_qkv.weight", "blk.0.ssm_out.weight"]
            .iter()
            .map(|name| {
                let t = g.tensor(name).unwrap().clone();
                gpu.load_weight(&mut g, &t).unwrap()
            })
            .collect();
        let cmd = gpu.command().unwrap();
        gpu.complete(cmd).unwrap();
        assert_eq!(gpu.residency.allocationCount(), 2);
        let bytes: usize = buffers
            .iter()
            .map(|b| b.0.raw.as_ref().unwrap().length())
            .sum();
        assert!(gpu.residency.allocatedSize() >= bytes as u64);
        println!(
            "RESIDENCY allocations={} committed_bytes={}",
            gpu.residency.allocationCount(),
            gpu.residency.allocatedSize()
        );
        drop(buffers);
        assert_eq!(gpu.residency.allocationCount(), 0);
    }
}

#[cfg(test)]
mod indexed_expert_probe {
    use super::*;
    #[test]
    #[ignore = "real expert matrices, shuffled route IDs and packed tile tails"]
    fn shuffled_routes_match_f64() {
        let gpu = Gpu::new().unwrap();
        let mut g =
            crate::gguf::Gguf::open(std::path::Path::new(&std::env::var("QWEN_MODEL").unwrap()))
                .unwrap();
        for name in [
            "blk.0.ffn_gate_exps.weight",
            "blk.1.ffn_gate_exps.weight",
            "blk.8.ffn_gate_exps.weight",
            "blk.0.ffn_down_exps.weight",
        ] {
            let t = g.tensor(name).unwrap().clone();
            let rows = 65;
            let m = g
                .rows(&gpu.reads, name, &(0..rows * 2).collect::<Vec<_>>())
                .unwrap();
            let w = gpu.bytes(&m.data).unwrap();
            let dw = gpu.dequant(&m).unwrap();
            for count in [1usize, 2, 15, 16, 17, 31, 32, 33] {
                let pairs: Vec<u32> = (0..count * 2).rev().map(|v| v as u32).collect();
                let pb = gpu
                    .bytes(
                        &pairs
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let mut tiles = Vec::<u32>::new();
                for expert in 0..2 {
                    for start in (0..count).step_by(32) {
                        let nt = (count - start).min(32);
                        tiles.extend([
                            expert as u32,
                            (expert * count + start) as u32,
                            nt as u32,
                            u32::from(nt >= 2),
                        ]);
                    }
                }
                let tb = gpu
                    .bytes(
                        &tiles
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let top = 10usize;
                for packed_input in [false, true] {
                    let inputs = if packed_input {
                        count * 2
                    } else {
                        (count * 2).div_ceil(top)
                    };
                    let input: Vec<f32> = (0..inputs * m.cols)
                        .map(|i| ((i * 17 % 31) as f32 - 15.) * 0.03125)
                        .collect();
                    let x = gpu.floats(&input).unwrap();
                    let y = gpu.output(rows * count * 2).unwrap();
                    let cmd = gpu.command().unwrap();
                    gpu.op(
                        &cmd,
                        if packed_input {
                            "quant_expert_packed"
                        } else {
                            "quant_expert_gather"
                        },
                        &[&w, &x, &tb, &y, &pb],
                        &[
                            m.cols as u32,
                            rows as u32,
                            m.stride as u32,
                            t.kind,
                            top as u32,
                        ],
                        rows,
                        tiles.len() / 4,
                    )
                    .unwrap();
                    gpu.complete(cmd).unwrap();
                    let actual = gpu.read_f32(&y).unwrap();
                    for (packed, &pair) in pairs[..count * 2].iter().enumerate() {
                        let expert = packed / count;
                        let src = if packed_input {
                            packed
                        } else {
                            pair as usize / top
                        };
                        let dst = packed;
                        for row in 0..rows {
                            let expected: f64 = (0..m.cols)
                                .map(|k| {
                                    f64::from(dw[(expert * rows + row) * m.cols + k])
                                        * f64::from(input[src * m.cols + k])
                                })
                                .sum();
                            let a = f64::from(actual[dst * rows + row]);
                            assert!(
                                a.is_finite()
                                    && (a - expected).abs() <= 1e-4 + 5e-5 * expected.abs(),
                                "{name} {count} {packed_input} {packed} {row}: {a} != {expected}"
                            );
                        }
                    }
                }
            }
            println!(
                "INDEXED_F64 {name} passed 65-row tails and token counts1,2,15,16,17,31,32,33"
            );
        }
    }

    #[test]
    #[ignore = "real gate/up experts: fused SwiGLU tiles equal gather, gather and swiglu bitwise"]
    fn fused_gate_up_matches_separate() {
        let gpu = Gpu::new().unwrap();
        let mut g =
            crate::gguf::Gguf::open(std::path::Path::new(&std::env::var("QWEN_MODEL").unwrap()))
                .unwrap();
        let words = |v: &[u32]| {
            gpu.bytes(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
                .unwrap()
        };
        let (rows, top) = (65usize, 10usize);
        for layer in [0, 1, 8] {
            let load = |g: &mut crate::gguf::Gguf, kind: &str| {
                let name = format!("blk.{layer}.ffn_{kind}_exps.weight");
                g.rows(&gpu.reads, &name, &(0..rows * 2).collect::<Vec<_>>())
                    .unwrap()
            };
            let (gm, um) = (load(&mut g, "gate"), load(&mut g, "up"));
            let (gw, uw) = (gpu.bytes(&gm.data).unwrap(), gpu.bytes(&um.data).unwrap());
            let params = [
                gm.cols as u32,
                rows as u32,
                gm.stride as u32,
                gm.kind,
                top as u32,
            ];
            for count in [1usize, 2, 15, 16, 17, 31, 32, 33] {
                let pairs: Vec<u32> = (0..count * 2).rev().map(|v| v as u32).collect();
                let mut tiles = Vec::<u32>::new();
                for expert in 0..2 {
                    for start in (0..count).step_by(32) {
                        let nt = (count - start).min(32);
                        tiles.extend([
                            expert as u32,
                            (expert * count + start) as u32,
                            nt as u32,
                            u32::from(nt >= 2),
                        ]);
                    }
                }
                let (pb, tb) = (words(&pairs), words(&tiles));
                let input: Vec<f32> = (0..(count * 2).div_ceil(top) * gm.cols)
                    .map(|i| ((i * 17 % 31) as f32 - 15.) * 0.03125)
                    .collect();
                let x = gpu.floats(&input).unwrap();
                let n = rows * count * 2;
                let (gy, uy, fused) = (
                    gpu.output(n).unwrap(),
                    gpu.output(n).unwrap(),
                    gpu.output(n).unwrap(),
                );
                let cmd = gpu.command().unwrap();
                for (w, y) in [(&gw, &gy), (&uw, &uy)] {
                    gpu.op(
                        &cmd,
                        "quant_expert_gather",
                        &[w, &x, &tb, y, &pb],
                        &params,
                        rows,
                        tiles.len() / 4,
                    )
                    .unwrap();
                }
                let act = gpu.activation(&cmd, &gy, &uy, n).unwrap();
                gpu.op(
                    &cmd,
                    "quant_expert_gate_up",
                    &[&gw, &uw, &x, &tb, &fused, &pb],
                    &params,
                    rows,
                    tiles.len() / 4,
                )
                .unwrap();
                gpu.complete(cmd).unwrap();
                let (a, b) = (gpu.read_f32(&act).unwrap(), gpu.read_f32(&fused).unwrap());
                assert!(
                    a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "layer {layer} count {count}: fused gate/up differs"
                );
            }
            println!(
                "FUSED_GATE_UP layer={layer} kind={} bitwise for 65-row tails and counts1-33",
                gm.kind
            );
        }
    }
}

#[cfg(test)]
mod half_tile_probe {
    use super::*;
    #[test]
    #[ignore = "mixed-precision real weight tile error and finite outputs"]
    fn real_operand_error() {
        let gpu = Gpu::new().unwrap();
        let mut g =
            crate::gguf::Gguf::open(std::path::Path::new(&std::env::var("QWEN_MODEL").unwrap()))
                .unwrap();
        for name in [
            "blk.0.ffn_gate_exps.weight",
            "blk.1.ffn_gate_exps.weight",
            "blk.8.ffn_gate_exps.weight",
            "blk.0.ffn_down_exps.weight",
            "blk.0.attn_qkv.weight",
        ] {
            let rows = 65;
            let m = g
                .rows(&gpu.reads, name, &(0..rows).collect::<Vec<_>>())
                .unwrap();
            let w = gpu.bytes(&m.data).unwrap();
            let dw = gpu.dequant(&m).unwrap();
            for count in [2usize, 15, 16, 17, 31, 32] {
                let input: Vec<f32> = (0..count * m.cols)
                    .map(|i| ((i * 17 % 101) as f32 - 50.) * 0.013271)
                    .collect();
                let x = gpu.floats(&input).unwrap();
                let y = gpu.output(rows * count).unwrap();
                let tiles = gpu
                    .bytes(
                        &[0u32, 0, count as u32, 1]
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let pairs: Vec<u32> = (0..count as u32).rev().collect();
                let pb = gpu
                    .bytes(
                        &pairs
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                for mode in [0, 1, 2] {
                    let cmd = gpu.command().unwrap();
                    if mode == 0 {
                        gpu.op(
                            &cmd,
                            "quant_expert_gemm",
                            &[&w, &x, &tiles, &y],
                            &[m.cols as u32, rows as u32, m.stride as u32, m.kind],
                            rows,
                            1,
                        )
                        .unwrap();
                    } else {
                        gpu.op(
                            &cmd,
                            if mode == 1 {
                                "quant_expert_gather"
                            } else {
                                "quant_expert_packed"
                            },
                            &[&w, &x, &tiles, &y, &pb],
                            &[m.cols as u32, rows as u32, m.stride as u32, m.kind, 1],
                            rows,
                            1,
                        )
                        .unwrap();
                    }
                    gpu.complete(cmd).unwrap();
                    let actual = gpu.read_f32(&y).unwrap();
                    let mut error = 0.;
                    let mut norm = 0.;
                    let mut max_abs = 0f64;
                    for (token, &pair) in pairs[..count].iter().enumerate() {
                        let src = if mode == 1 { pair as usize } else { token };
                        let dst = token;
                        for row in 0..rows {
                            let expected: f64 = (0..m.cols)
                                .map(|k| {
                                    f64::from(dw[row * m.cols + k])
                                        * f64::from(input[src * m.cols + k])
                                })
                                .sum();
                            let a = f64::from(actual[dst * rows + row]);
                            assert!(a.is_finite());
                            let d = (a - expected).abs();
                            max_abs = max_abs.max(d);
                            error += d * d;
                            norm += expected * expected;
                        }
                    }
                    let relative = (error / norm.max(1e-30)).sqrt();
                    assert!(relative < 0.001, "{name} {count}: relativeL2={relative}");
                    println!(
                        "HALF_ERROR {name} count={count} mode={mode} relative_l2={relative} max_abs={max_abs}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod pointwise_group_probe {
    use super::*;
    #[test]
    #[ignore = "pointwise launch geometry, bit-identical output and GPU intervals"]
    fn real_sizes() {
        let gpu = Gpu::new().unwrap();
        for n in [17usize, 257, 25600, 6553600] {
            let values = (0..n)
                .map(|i| ((i * 17 % 31) as f32 - 15.) * 0.125)
                .collect::<Vec<_>>();
            let a = gpu.floats(&values).unwrap();
            let b = gpu.floats(&values).unwrap();
            let out = gpu.output(n).unwrap();
            for op in ["swiglu", "unary"] {
                let pipeline = if op == "swiglu" {
                    &gpu.swiglu
                } else {
                    &gpu.operations[op]
                };
                let buffers = if op == "swiglu" {
                    vec![&a, &b, &out]
                } else {
                    vec![&a, &out]
                };
                let params = if op == "swiglu" {
                    vec![n as u32]
                } else {
                    vec![n as u32, 1, 0.25f32.to_bits()]
                };
                let mut reference = Vec::new();
                for (block, order) in [[32usize, 128, 256], [256, 128, 32]]
                    .into_iter()
                    .enumerate()
                {
                    for threads in order {
                        let mut samples = Vec::new();
                        for rep in 0..16 {
                            let cmd = gpu.command().unwrap();
                            for _ in 0..4 {
                                gpu.encode(
                                    &cmd,
                                    pipeline,
                                    &buffers,
                                    &params,
                                    MTLSize {
                                        width: if threads == 32 { n.div_ceil(32) } else { n },
                                        height: 1,
                                        depth: 1,
                                    },
                                    if threads == 32 { 1 } else { threads },
                                )
                                .unwrap();
                            }
                            gpu.submit(&cmd);
                            cmd.waitUntilCompleted();
                            let elapsed = cmd.GPUEndTime() - cmd.GPUStartTime();
                            gpu.wait(cmd).unwrap();
                            if rep >= 4 {
                                samples.push(elapsed);
                            }
                        }
                        let actual = gpu.read_f32(&out).unwrap();
                        if reference.is_empty() {
                            reference = actual;
                        } else {
                            assert!(
                                reference
                                    .iter()
                                    .zip(&actual)
                                    .all(|(a, b)| a.to_bits() == b.to_bits())
                            );
                        }
                        println!(
                            "POINTWISE op={op} n={n} block={block} threads={threads} samples={samples:?}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod dense_block_probe {
    use super::*;
    #[test]
    #[ignore = "real Q8 IQ3 BF16 block values bit-identical to scalar decode"]
    fn block_values() {
        let gpu = Gpu::new().unwrap();
        let mut g =
            crate::gguf::Gguf::open(std::path::Path::new(&std::env::var("QWEN_MODEL").unwrap()))
                .unwrap();
        for kind in [8u32, 21, 30] {
            let name = g
                .tensors
                .iter()
                .find(|(_, t)| t.kind == kind && t.dims[0] >= 128 && t.bytes / t.stride >= 65)
                .unwrap()
                .0
                .clone();
            let m = g
                .rows(&gpu.reads, &name, &(0..65).collect::<Vec<_>>())
                .unwrap();
            let expected = gpu.dequant(&m).unwrap();
            let w = gpu.bytes(&m.data).unwrap();
            let out = gpu.output(expected.len()).unwrap();
            let cmd = gpu.command().unwrap();
            gpu.encode(
                &cmd,
                &gpu.projections[&("decode_blocks16", kind)],
                &[&w, &out],
                &[m.cols as u32, 65, m.stride as u32],
                MTLSize {
                    width: (expected.len() / 16).div_ceil(32),
                    height: 1,
                    depth: 1,
                },
                1,
            )
            .unwrap();
            let actual = gpu.finish(cmd, &out).unwrap();
            assert!(
                expected
                    .iter()
                    .zip(actual)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{name}/{kind}"
            );
            println!("DENSE_BLOCK_EXACT {name}/{kind}");
        }
    }
    #[test]
    #[ignore = "real full Q8 head and projection vectors, F64 tails and GPU intervals"]
    fn q8_vectors() {
        let gpu = Gpu::new().unwrap();
        let mut g =
            crate::gguf::Gguf::open(std::path::Path::new(&std::env::var("QWEN_DRAFT").unwrap()))
                .unwrap();
        for name in [
            "output.weight",
            "blk.48.nextn.eh_proj.weight",
            "blk.48.hc_attn_down.weight",
        ] {
            let t = g.tensor(name).unwrap().clone();
            assert_eq!(t.kind, 8);
            for rows in [65, t.dims[1]] {
                let m = g
                    .rows(&gpu.reads, name, &(0..rows).collect::<Vec<_>>())
                    .unwrap();
                let w = gpu.bytes(&m.data).unwrap();
                let decoded = if rows == 65 {
                    gpu.dequant(&m).unwrap()
                } else {
                    Vec::new()
                };
                for count in [1usize, 2, 3, 4] {
                    let data = (0..m.cols * count)
                        .map(|i| ((i * 17 % 31) as f32 - 15.) * 0.03125)
                        .collect::<Vec<_>>();
                    let x = gpu.floats(&data).unwrap();
                    let out = gpu.output(rows * count).unwrap();
                    let params = [
                        m.cols as u32,
                        rows as u32,
                        m.stride as u32,
                        8,
                        0,
                        count as u32,
                    ];
                    for (block, order) in [[0usize, 1, 2, 4], [4, 2, 1, 0]].into_iter().enumerate()
                    {
                        for nr in order {
                            let mut samples = Vec::new();
                            for rep in 0..if rows == 65 { 1 } else { 12 } {
                                let cmd = gpu.command().unwrap();
                                for _ in 0..if rows == 65 { 1 } else { 4 } {
                                    if nr == 0 {
                                        gpu.encode(
                                            &cmd,
                                            &gpu.projections[&("quant_matmul", 8)],
                                            &[&w, &x, &out],
                                            &params,
                                            MTLSize {
                                                width: rows,
                                                height: 1,
                                                depth: 1,
                                            },
                                            4,
                                        )
                                        .unwrap();
                                    } else {
                                        let kernel = match nr {
                                            1 => "q8_dense1",
                                            2 => "q8_dense2",
                                            _ => "q8_dense4",
                                        };
                                        gpu.encode(
                                            &cmd,
                                            &gpu.operations[kernel],
                                            &[&w, &x, &out],
                                            &params,
                                            MTLSize {
                                                width: rows.div_ceil(4 * nr) * 4,
                                                height: count,
                                                depth: 1,
                                            },
                                            4,
                                        )
                                        .unwrap();
                                    }
                                }
                                gpu.submit(&cmd);
                                cmd.waitUntilCompleted();
                                let seconds = cmd.GPUEndTime() - cmd.GPUStartTime();
                                gpu.wait(cmd).unwrap();
                                if rep >= 4 {
                                    samples.push(seconds);
                                }
                            }
                            if rows == 65 {
                                let actual = gpu.read_f32(&out).unwrap();
                                for token in 0..count {
                                    for row in 0..rows {
                                        let expected: f64 = (0..m.cols)
                                            .map(|k| {
                                                f64::from(decoded[row * m.cols + k])
                                                    * f64::from(data[token * m.cols + k])
                                            })
                                            .sum();
                                        assert!(
                                            (f64::from(actual[token * rows + row]) - expected)
                                                .abs()
                                                <= 1e-4 + 5e-5 * expected.abs(),
                                            "{name}/{count}/{nr}/{row}"
                                        );
                                    }
                                }
                            } else {
                                println!(
                                    "Q8_VECTOR name={name} count={count} block={block} nr={nr} samples={samples:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod buffer_view_tests {
    use super::*;
    #[test]
    fn views_bind_offsets_and_retain_the_workspace_lease() {
        let gpu = Gpu::new().unwrap();
        gpu.configure_workspace(1 << 20);
        let parent = gpu.output(64).unwrap();
        let input = gpu
            .floats(&(0..64).map(|i| i as f32).collect::<Vec<_>>())
            .unwrap();
        let cmd = gpu.command().unwrap();
        gpu.op(&cmd, "copy_range", &[&input, &parent], &[64, 0], 64, 1)
            .unwrap();
        gpu.complete(cmd).unwrap();
        let view = parent.view_f32(16, 32).unwrap().view_f32(8, 16).unwrap();
        assert_eq!(
            gpu.read_f32(&view).unwrap(),
            (24..40).map(|i| i as f32).collect::<Vec<_>>()
        );
        assert!(view.view_f32(16, 1).is_err());
        let slices = [
            parent.view_f32(0, 16).unwrap(),
            parent.view_f32(16, 16).unwrap(),
        ];
        let combined = Buffer::contiguous_f32(&slices, 16).unwrap();
        assert_eq!(combined.length(), 128);
        assert!(Buffer::contiguous_f32(&[slices[1].clone(), slices[0].clone()], 16).is_none());
        let cmd = gpu.command().unwrap();
        gpu.op(&cmd, "copy_range", &[&input, &view], &[16, 0], 16, 1)
            .unwrap();
        gpu.complete(cmd).unwrap();
        let actual = gpu.read_f32(&parent).unwrap();
        for (i, &v) in actual.iter().enumerate() {
            assert_eq!(
                v,
                if (24..40).contains(&i) {
                    (i - 24) as f32
                } else {
                    i as f32
                }
            );
        }
        drop(slices);
        drop(combined);
        let weak = Rc::downgrade(&parent.0);
        let out = gpu.output(16).unwrap();
        let cmd = gpu.command().unwrap();
        gpu.op(&cmd, "copy_range", &[&view, &out], &[16, 0], 16, 1)
            .unwrap();
        drop(parent);
        drop(view);
        assert!(weak.upgrade().is_some());
        assert!(
            gpu.workspace
                .borrow()
                .free
                .get(&256)
                .is_none_or(|v| v.is_empty())
        );
        gpu.submit(&cmd);
        gpu.wait(cmd).unwrap();
        assert!(weak.upgrade().is_none());
        assert_eq!(
            gpu.read_f32(&out).unwrap(),
            (0..16).map(|i| i as f32).collect::<Vec<_>>()
        );
        assert_eq!(gpu.workspace.borrow().free.get(&256).unwrap().len(), 1);
    }
}
#[cfg(test)]
mod mps_crossover_probe {
    use super::*;
    #[test]
    #[ignore = "actual large Qwen shapes,1024/2048/4095/4096 GPU and API wall crossover"]
    fn real_large_counts() {
        let gpu = Gpu::new().unwrap();
        gpu.configure_workspace(4usize << 30);
        let mut g =
            crate::gguf::Gguf::open(std::path::Path::new(&std::env::var("QWEN_MODEL").unwrap()))
                .unwrap();
        for (kind, cols, rows) in [
            (30, 10240, 320),
            (30, 320, 10240),
            (21, 2560, 6144),
            (21, 6144, 2560),
            (21, 2560, 640),
            (23, 2560, 640),
            (23, 2560, 12288),
        ] {
            let name = g
                .tensors
                .iter()
                .find(|(_, t)| t.kind == kind && t.dims == [cols, rows])
                .unwrap()
                .0
                .clone();
            let tensor = g.tensor(&name).unwrap().clone();
            let w = gpu.load_weight(&mut g, &tensor).unwrap();
            for count in [1024usize, 4095] {
                let x = gpu
                    .floats(
                        &(0..cols * count)
                            .map(|i| ((i * 17 % 31) as f32 - 15.) * 0.03125)
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                for (block, order) in [[false, true], [true, false]].into_iter().enumerate() {
                    for mps in order {
                        let mut samples = Vec::new();
                        for rep in 0..8 {
                            let start = std::time::Instant::now();
                            let cmd = gpu.command().unwrap();
                            let y = if mps {
                                gpu.dense_mps(&cmd, &w, cols, rows, tensor.stride, kind, &x, count)
                                    .unwrap()
                            } else {
                                gpu.matmul(&cmd, &w, cols, rows, tensor.stride, kind, 0, &x, count)
                                    .unwrap()
                            };
                            gpu.submit(&cmd);
                            cmd.waitUntilCompleted();
                            let seconds = cmd.GPUEndTime() - cmd.GPUStartTime();
                            gpu.wait(cmd).unwrap();
                            if rep >= 2 {
                                samples.push((start.elapsed().as_secs_f64(), seconds));
                            }
                            drop(y);
                        }
                        println!(
                            "MPS_CROSS name={name} kind={kind} cols={cols} rows={rows} count={count} block={block} mps={mps} samples={samples:?}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod bf16_vector_probe {
    use super::*;
    #[test]
    #[ignore = "BF16 real shapes, F64 tail validation and ABBA kernel timing"]
    fn real_bf16_shapes() {
        let gpu = Gpu::new().unwrap();
        let mut g =
            crate::gguf::Gguf::open(std::path::Path::new(&std::env::var("QWEN_MODEL").unwrap()))
                .unwrap();
        let mut names = Vec::new();
        let mut shapes = std::collections::BTreeSet::new();
        for (name, t) in &g.tensors {
            if t.kind == 30
                && t.dims.len() == 2
                && matches!(t.dims[0], 320 | 2560 | 10240)
                && shapes.insert(t.dims.clone())
            {
                names.push(name.clone());
            }
        }
        for name in names {
            let t = g.tensor(&name).unwrap().clone();
            for (shape_i, rows) in [t.dims[1].min(65), t.dims[1]].into_iter().enumerate() {
                let m = g
                    .rows(&gpu.reads, &name, &(0..rows).collect::<Vec<_>>())
                    .unwrap();
                let w = gpu.bytes(&m.data).unwrap();
                let check = shape_i == 0;
                let decoded = if check {
                    gpu.dequant(&m).unwrap()
                } else {
                    Vec::new()
                };
                for count in [1usize, 2, 3, 4] {
                    let data = (0..m.cols * count)
                        .map(|i| ((i * 17 % 31) as f32 - 15.) * 0.03125)
                        .collect::<Vec<_>>();
                    let x = gpu.floats(&data).unwrap();
                    let out = gpu.output(rows * count).unwrap();
                    let params = [
                        m.cols as u32,
                        rows as u32,
                        m.stride as u32,
                        30,
                        0,
                        count as u32,
                    ];
                    for (block, order) in [[0usize, 1, 2, 4, 128], [128, 4, 2, 1, 0]]
                        .into_iter()
                        .enumerate()
                    {
                        for nr in order {
                            let mut samples = Vec::new();
                            for rep in 0..if check { 1 } else { 8 } {
                                let cmd = gpu.command().unwrap();
                                for _ in 0..if check { 1 } else { 4 } {
                                    if nr == 0 {
                                        gpu.encode(
                                            &cmd,
                                            &gpu.projections[&("quant_matmul", 30)],
                                            &[&w, &x, &out],
                                            &params,
                                            MTLSize {
                                                width: rows,
                                                height: 1,
                                                depth: 1,
                                            },
                                            4,
                                        )
                                        .unwrap();
                                    } else if nr == 128 {
                                        gpu.encode(
                                            &cmd,
                                            &gpu.operations["bf16_hc_split"],
                                            &[&w, &x, &out],
                                            &[m.cols as u32, rows as u32, m.stride as u32, 0],
                                            MTLSize {
                                                width: rows * 4,
                                                height: count,
                                                depth: 1,
                                            },
                                            4,
                                        )
                                        .unwrap();
                                    } else {
                                        let kernel = match nr {
                                            1 => "bf16_dense1",
                                            2 => "bf16_dense2",
                                            _ => "bf16_dense4",
                                        };
                                        gpu.encode(
                                            &cmd,
                                            &gpu.operations[kernel],
                                            &[&w, &x, &out],
                                            &params,
                                            MTLSize {
                                                width: rows.div_ceil(4 * nr) * 4,
                                                height: count,
                                                depth: 1,
                                            },
                                            4,
                                        )
                                        .unwrap();
                                    }
                                }
                                gpu.submit(&cmd);
                                cmd.waitUntilCompleted();
                                let seconds = cmd.GPUEndTime() - cmd.GPUStartTime();
                                gpu.wait(cmd).unwrap();
                                if rep >= 2 {
                                    samples.push(seconds);
                                }
                            }
                            if check {
                                let actual = gpu.read_f32(&out).unwrap();
                                for token in 0..count {
                                    for row in 0..rows {
                                        let expected: f64 = (0..m.cols)
                                            .map(|k| {
                                                f64::from(decoded[row * m.cols + k])
                                                    * f64::from(data[token * m.cols + k])
                                            })
                                            .sum();
                                        assert!(
                                            (f64::from(actual[token * rows + row]) - expected)
                                                .abs()
                                                <= 1e-4 + 5e-5 * expected.abs(),
                                            "{name}/{count}/{nr}/{row}"
                                        );
                                    }
                                }
                            } else {
                                println!(
                                    "BF16_VECTOR name={name} cols={} rows={rows} count={count} block={block} nr={nr} samples={samples:?}",
                                    m.cols
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod parallel_convolution_tests {
    use super::*;
    #[test]
    fn windows_preserve_outputs_history_and_rollback_positions() {
        let gpu = Gpu::new().unwrap();
        for (channels, taps, dilation, count) in [
            (129usize, 4usize, 1usize, 1usize),
            (129, 4, 1, 2),
            (129, 4, 1, 4),
            (129, 4, 4, 3),
            (129, 4, 4, 17),
            (129, 4, 16, 17),
            (129, 4, 16, 64),
            (10240, 4, 1, 1024),
            (10240, 4, 1, 4095),
        ] {
            let hist = (taps - 1) * dilation;
            let checkpoints = count.min(3);
            let input = (0..channels * count)
                .map(|i| ((i * 13 % 31) as f32 - 15.) * 0.03125)
                .collect::<Vec<_>>();
            let initial = (0..channels * hist)
                .map(|i| ((i * 17 % 29) as f32 - 14.) * 0.03125)
                .collect::<Vec<_>>();
            let weights = (0..channels * taps)
                .map(|i| ((i * 7 % 23) as f32 - 11.) * 0.03125)
                .collect::<Vec<_>>();
            let x = gpu.floats(&input).unwrap();
            let w = gpu.floats(&weights).unwrap();
            let y = [
                gpu.output(channels * count).unwrap(),
                gpu.output(channels * count).unwrap(),
            ];
            let histories = [gpu.floats(&initial).unwrap(), gpu.floats(&initial).unwrap()];
            let snapshots = [
                gpu.output(checkpoints * channels * hist).unwrap(),
                gpu.output(checkpoints * channels * hist).unwrap(),
            ];
            let params = [
                channels as u32,
                taps as u32,
                dilation as u32,
                0,
                (4 * taps) as u32,
                count as u32,
                checkpoints as u32,
            ];
            for arm in [0usize, 1] {
                let cmd = gpu.command().unwrap();
                gpu.op(
                    &cmd,
                    if arm == 0 {
                        "convolution_batch"
                    } else {
                        "convolution_parallel"
                    },
                    &[&x, &w, &histories[arm], &y[arm], &snapshots[arm]],
                    &params,
                    if arm == 0 {
                        channels.div_ceil(32)
                    } else {
                        (channels * count).div_ceil(32)
                    },
                    1,
                )
                .unwrap();
                if arm == 1 {
                    gpu.op(
                        &cmd,
                        "convolution_history",
                        &[&x, &histories[arm]],
                        &[channels as u32, taps as u32, dilation as u32, count as u32],
                        channels.div_ceil(32),
                        1,
                    )
                    .unwrap();
                }
                gpu.complete(cmd).unwrap();
            }
            assert_eq!(
                gpu.read_f32(&y[0]).unwrap(),
                gpu.read_f32(&y[1]).unwrap(),
                "output c{channels} d{dilation} n{count}"
            );
            assert_eq!(
                gpu.read_f32(&histories[0]).unwrap(),
                gpu.read_f32(&histories[1]).unwrap(),
                "history c{channels} d{dilation} n{count}"
            );
            assert_eq!(
                gpu.read_f32(&snapshots[0]).unwrap(),
                gpu.read_f32(&snapshots[1]).unwrap(),
                "snapshots c{channels} d{dilation} n{count}"
            );
            if count >= 1024 {
                for (block, order) in [[0usize, 1], [1, 0]].into_iter().enumerate() {
                    for arm in order {
                        let mut samples = Vec::new();
                        for rep in 0..8 {
                            let cmd = gpu.command().unwrap();
                            gpu.op(
                                &cmd,
                                if arm == 0 {
                                    "convolution_batch"
                                } else {
                                    "convolution_parallel"
                                },
                                &[&x, &w, &histories[arm], &y[arm], &snapshots[arm]],
                                &params,
                                if arm == 0 {
                                    channels.div_ceil(32)
                                } else {
                                    (channels * count).div_ceil(32)
                                },
                                1,
                            )
                            .unwrap();
                            if arm == 1 {
                                gpu.op(
                                    &cmd,
                                    "convolution_history",
                                    &[&x, &histories[arm]],
                                    &[channels as u32, taps as u32, dilation as u32, count as u32],
                                    channels.div_ceil(32),
                                    1,
                                )
                                .unwrap();
                            }
                            gpu.submit(&cmd);
                            cmd.waitUntilCompleted();
                            let elapsed = cmd.GPUEndTime() - cmd.GPUStartTime();
                            gpu.wait(cmd).unwrap();
                            if rep >= 2 {
                                samples.push(elapsed);
                            }
                        }
                        println!(
                            "CONVOLUTION count={count} block={block} parallel={arm} samples={samples:?}"
                        );
                    }
                }
            }
        }
    }
}
#[cfg(test)]
mod hc_fusion_probe {
    use super::*;
    #[test]
    #[ignore = "real HC chain, exact intermediate/output bits and interleaved GPU timing"]
    fn real_hc_chains() {
        let gpu = Gpu::new().unwrap();
        for (model, prefixes) in [
            (
                "QWEN_MODEL",
                vec!["blk.0.hc_attn", "blk.0.hc_ffn", "output_hc"],
            ),
            (
                "QWEN_DRAFT",
                vec!["blk.48.hc_attn", "blk.48.hc_ffn", "blk.48.nextn.hc_head"],
            ),
        ] {
            let mut g =
                crate::gguf::Gguf::open(std::path::Path::new(&std::env::var(model).unwrap()))
                    .unwrap();
            let eps = g
                .float("qwen4exp.attention.layer_norm_rms_epsilon")
                .unwrap();
            for prefix in prefixes {
                let tensor = |suffix: &str| {
                    g.tensor(&format!("{prefix}_{suffix}.weight"))
                        .unwrap()
                        .clone()
                };
                let nt = tensor("norm");
                let dt = tensor("down");
                let ut = tensor("up");
                let it = g.tensor(&format!("{prefix}_inject.weight")).ok().cloned();
                println!(
                    "HC_SHAPE model={model} prefix={prefix} down={} up={} inject={:?}",
                    dt.kind,
                    ut.kind,
                    it.as_ref().map(|t| t.kind)
                );
                if dt.kind != 30 || ut.kind != 30 || it.as_ref().is_some_and(|t| t.kind != 30) {
                    continue;
                }
                assert_eq!(dt.dims, [10240, 320]);
                assert_eq!(ut.dims, [320, 10240]);
                let nw = gpu.load_weight(&mut g, &nt).unwrap();
                let dw = gpu.load_weight(&mut g, &dt).unwrap();
                let uw = gpu.load_weight(&mut g, &ut).unwrap();
                let iw = it.as_ref().map(|t| gpu.load_weight(&mut g, t).unwrap());
                for count in 1usize..=4 {
                    for pattern in 0..2 {
                        let input = (0..10240 * count)
                            .map(|i| {
                                if pattern == 0 {
                                    ((i * 17 % 31) as f32 - 15.) * 0.03125
                                } else {
                                    0.
                                }
                            })
                            .collect::<Vec<_>>();
                        let x = gpu.floats(&input).unwrap();
                        let mut reference: Option<(Vec<f32>, Vec<f32>, Vec<f32>)> = None;
                        for (block, order) in [[false, true], [true, false]].iter().enumerate() {
                            for &fused in order {
                                let mut times = Vec::new();
                                let mut last = None;
                                for rep in 0..if pattern == 0 { 6 } else { 1 } {
                                    let cmd = gpu.command().unwrap();
                                    for _ in 0..4 {
                                        let norm = gpu.output(10240 * count).unwrap();
                                        gpu.op(
                                            &cmd,
                                            "rms",
                                            &[&x, &nw, &norm],
                                            &[2560, 10240, 0, 2560, eps.to_bits(), 0],
                                            4 * count,
                                            1,
                                        )
                                        .unwrap();
                                        let (lo, mut inj) = if fused {
                                            let lo = gpu.output(320 * count).unwrap();
                                            let inj = gpu.output(4 * count).unwrap();
                                            gpu.op(
                                                &cmd,
                                                "hc_down_inject",
                                                &[
                                                    &dw,
                                                    iw.as_ref().unwrap_or(&dw),
                                                    &norm,
                                                    &lo,
                                                    &inj,
                                                ],
                                                &[
                                                    10240,
                                                    320,
                                                    dt.stride as u32,
                                                    it.as_ref().map_or(dt.stride, |t| t.stride)
                                                        as u32,
                                                    if it.is_some() { 4 } else { 0 },
                                                    count as u32,
                                                    0.25f32.to_bits(),
                                                ],
                                                if count <= 2 { 320 } else { 80 }
                                                    + if it.is_some() { 4 } else { 0 },
                                                count,
                                            )
                                            .unwrap();
                                            (lo, inj)
                                        } else {
                                            let raw = gpu
                                                .matmul(
                                                    &cmd, &dw, 10240, 320, dt.stride, 30, 0, &norm,
                                                    count,
                                                )
                                                .unwrap();
                                            let lo = gpu.output(320 * count).unwrap();
                                            gpu.op(
                                                &cmd,
                                                "unary",
                                                &[&raw, &lo],
                                                &[(320 * count) as u32, 1, 0.25f32.to_bits()],
                                                (320 * count).div_ceil(32),
                                                1,
                                            )
                                            .unwrap();
                                            (lo, gpu.output(4 * count).unwrap())
                                        };
                                        let out = gpu.output(2560 * count).unwrap();
                                        if fused && count < 4 {
                                            gpu.op(
                                                &cmd,
                                                "hc_up_collapse",
                                                &[&uw, &lo, &norm, &out],
                                                &[2560, 4, 320, ut.stride as u32, count as u32],
                                                2560usize.div_ceil(4),
                                                count,
                                            )
                                            .unwrap();
                                        } else if fused {
                                            let gates = gpu
                                                .matmul(
                                                    &cmd, &uw, 320, 10240, ut.stride, ut.kind, 0,
                                                    &lo, count,
                                                )
                                                .unwrap();
                                            gpu.op(
                                                &cmd,
                                                "hc_collapse_sigmoid",
                                                &[&norm, &gates, &out],
                                                &[2560, 4, count as u32],
                                                (2560 * count).div_ceil(32),
                                                1,
                                            )
                                            .unwrap();
                                        } else {
                                            let gates = gpu
                                                .matmul(
                                                    &cmd, &uw, 320, 10240, ut.stride, ut.kind, 0,
                                                    &lo, count,
                                                )
                                                .unwrap();
                                            let activated = gpu.output(10240 * count).unwrap();
                                            gpu.op(
                                                &cmd,
                                                "unary",
                                                &[&gates, &activated],
                                                &[(10240 * count) as u32, 0, 1f32.to_bits()],
                                                (10240 * count).div_ceil(32),
                                                1,
                                            )
                                            .unwrap();
                                            gpu.op(
                                                &cmd,
                                                "hc_collapse_batch",
                                                &[&norm, &activated, &out],
                                                &[2560, 4, count as u32],
                                                (2560 * count).div_ceil(32),
                                                1,
                                            )
                                            .unwrap();
                                        }
                                        if !fused {
                                            if let (Some(w), Some(t)) = (&iw, &it) {
                                                inj = gpu
                                                    .matmul(
                                                        &cmd, w, 10240, 4, t.stride, t.kind, 0,
                                                        &norm, count,
                                                    )
                                                    .unwrap();
                                            }
                                        }
                                        last = Some((lo, inj, out));
                                    }
                                    gpu.submit(&cmd);
                                    cmd.waitUntilCompleted();
                                    let seconds = cmd.GPUEndTime() - cmd.GPUStartTime();
                                    gpu.wait(cmd).unwrap();
                                    if rep >= 2 {
                                        times.push(seconds / 4.);
                                    }
                                }
                                let (lo, inj, out) = last.unwrap();
                                let actual = (
                                    gpu.read_f32(&lo).unwrap(),
                                    if it.is_some() {
                                        gpu.read_f32(&inj).unwrap()
                                    } else {
                                        vec![]
                                    },
                                    gpu.read_f32(&out).unwrap(),
                                );
                                if let Some(r) = &reference {
                                    for (label, a, b) in [
                                        ("lo", &actual.0, &r.0),
                                        ("inject", &actual.1, &r.1),
                                        ("out", &actual.2, &r.2),
                                    ] {
                                        let mismatches = a
                                            .iter()
                                            .zip(b)
                                            .filter(|(a, b)| a.to_bits() != b.to_bits())
                                            .count();
                                        assert_eq!(
                                            mismatches, 0,
                                            "{model}/{prefix}/{count}/{pattern}/{fused}/{label}"
                                        );
                                    }
                                } else {
                                    reference = Some(actual);
                                }
                                println!(
                                    "HC_FUSION model={model} prefix={prefix} count={count} pattern={pattern} block={block} fused={fused} samples={times:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod routed_verify_probe {
    use super::*;
    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }
    // GPU seconds per encoding of `body`, eight encodings per command buffer.
    fn gpu_time(gpu: &Gpu, body: &dyn Fn(&Command)) -> f64 {
        let mut samples = Vec::new();
        for round in 0..24 {
            let cmd = gpu.command().unwrap();
            for _ in 0..8 {
                body(&cmd);
            }
            cmd.commit();
            cmd.waitUntilCompleted();
            assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
            if round >= 4 {
                samples.push((cmd.GPUEndTime() - cmd.GPUStartTime()) / 8.0);
            }
        }
        median(samples)
    }
    fn run(gpu: &Gpu, body: &dyn Fn(&Command)) {
        let cmd = gpu.command().unwrap();
        body(&cmd);
        gpu.complete(cmd).unwrap();
    }
    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }
    #[test]
    #[ignore = "requires QWEN_MODEL; routed verify candidates on resident real experts"]
    fn real_routed_verify() {
        let gpu = Gpu::new().unwrap();
        let mut g =
            crate::gguf::Gguf::open(std::path::Path::new(&std::env::var("QWEN_MODEL").unwrap()))
                .unwrap();
        let layer = std::env::var("ROUTED_LAYER").unwrap_or_else(|_| "0".into());
        let tensor = |kind: &str| {
            g.tensor(&format!("blk.{layer}.ffn_{kind}_exps.weight"))
                .unwrap()
                .clone()
        };
        let (gt, ut, dt) = (tensor("gate"), tensor("up"), tensor("down"));
        let down_name = format!("blk.{layer}.ffn_down_exps.weight");
        assert_eq!((gt.kind, ut.kind, dt.kind), (22, 22, 42));
        let gw = gpu.load_weight(&mut g, &gt).unwrap();
        let uw = gpu.load_weight(&mut g, &ut).unwrap();
        let dw = gpu.load_weight(&mut g, &dt).unwrap();
        let (width, ff, experts, top) = (gt.dims[0], gt.dims[1], gt.dims[2], 10usize);
        for count in [1usize, 2, 3, 4] {
            let pairs = top * count;
            // Distinct experts per token, deterministic and spread over all experts.
            let mut ids = Vec::new();
            let mut state = 0x9e37_79b9u32.wrapping_add(count as u32);
            for _ in 0..count {
                let mut token = Vec::new();
                while token.len() < top {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let id = (state >> 8) % experts as u32;
                    if !token.contains(&id) {
                        token.push(id);
                    }
                }
                ids.extend(token);
            }
            let routes = gpu
                .bytes(&ids.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<_>>())
                .unwrap();
            let x_input = (0..width * count)
                .map(|i| (i as f32 * 0.618_034).sin() * 0.37)
                .collect::<Vec<_>>();
            let x = gpu.floats(&x_input).unwrap();
            let down_input = (0..ff * pairs)
                .map(|i| (i as f32 * 0.414_214).cos() * 0.29)
                .collect::<Vec<_>>();
            let dx = gpu.floats(&down_input).unwrap();
            let up_params = [width as u32, ff as u32, gt.stride as u32, 22, top as u32, 0];
            let down_params = [ff as u32, width as u32, dt.stride as u32, 42, top as u32, 1];
            let (bg, bu, cg) = (
                gpu.zeros(ff * pairs).unwrap(),
                gpu.zeros(ff * pairs).unwrap(),
                gpu.zeros(ff * pairs).unwrap(),
            );
            let (fused, fused_tg) = (
                gpu.zeros(ff * pairs).unwrap(),
                gpu.zeros(ff * pairs).unwrap(),
            );
            let (bd, cd) = (
                gpu.zeros(width * pairs).unwrap(),
                gpu.zeros(width * pairs).unwrap(),
            );
            let (half_gate, half_fused) = (
                gpu.zeros(ff * pairs).unwrap(),
                gpu.zeros(ff * pairs).unwrap(),
            );
            let gate_half = |cmd: &Command| {
                gpu.op(
                    cmd,
                    "routed_iq2s_half",
                    &[&gw, &x, &routes, &half_gate],
                    &up_params,
                    ff,
                    pairs,
                )
                .unwrap();
            };
            let fuse_half = |cmd: &Command| {
                gpu.op(
                    cmd,
                    "routed_gate_up_iq2s_half",
                    &[&gw, &uw, &x, &routes, &half_fused],
                    &up_params,
                    ff,
                    pairs,
                )
                .unwrap();
            };
            let act = std::cell::RefCell::new(None);
            let separate = |cmd: &Command| {
                gpu.op(
                    cmd,
                    "project_routed_batch",
                    &[&gw, &x, &routes, &bg],
                    &up_params,
                    ff,
                    pairs,
                )
                .unwrap();
                gpu.op(
                    cmd,
                    "project_routed_batch",
                    &[&uw, &x, &routes, &bu],
                    &up_params,
                    ff,
                    pairs,
                )
                .unwrap();
                *act.borrow_mut() = Some(gpu.activation(cmd, &bg, &bu, ff * pairs).unwrap());
            };
            let gate_tg = |cmd: &Command| {
                gpu.op(
                    cmd,
                    "routed_iq2s_tg",
                    &[&gw, &x, &routes, &cg],
                    &up_params,
                    ff,
                    pairs,
                )
                .unwrap();
            };
            let gate_base = |cmd: &Command| {
                gpu.op(
                    cmd,
                    "project_routed_batch",
                    &[&gw, &x, &routes, &bg],
                    &up_params,
                    ff,
                    pairs,
                )
                .unwrap();
            };
            let fuse = |cmd: &Command| {
                gpu.op(
                    cmd,
                    "routed_gate_up_iq2s",
                    &[&gw, &uw, &x, &routes, &fused],
                    &up_params,
                    ff,
                    pairs,
                )
                .unwrap();
            };
            let fuse_tg = |cmd: &Command| {
                gpu.op(
                    cmd,
                    "routed_gate_up_iq2s_tg",
                    &[&gw, &uw, &x, &routes, &fused_tg],
                    &up_params,
                    ff,
                    pairs,
                )
                .unwrap();
            };
            let down_base = |cmd: &Command| {
                gpu.op(
                    cmd,
                    "project_routed_batch",
                    &[&dw, &dx, &routes, &bd],
                    &down_params,
                    width,
                    pairs,
                )
                .unwrap();
            };
            let down_lanes = |cmd: &Command| {
                gpu.op(
                    cmd,
                    "routed_q2_lanes8",
                    &[&dw, &dx, &routes, &cd],
                    &down_params,
                    width,
                    pairs,
                )
                .unwrap();
            };
            for body in [
                &separate as &dyn Fn(&Command),
                &gate_tg,
                &fuse,
                &fuse_tg,
                &down_base,
                &down_lanes,
                &gate_half,
                &fuse_half,
            ] {
                run(&gpu, body);
            }
            let reference_act = gpu.read_f32(act.borrow().as_ref().unwrap()).unwrap();
            let gate_equal = bits(&gpu.read_f32(&cg).unwrap()) == bits(&gpu.read_f32(&bg).unwrap());
            let fused_equal = bits(&gpu.read_f32(&fused).unwrap()) == bits(&reference_act);
            let fused_tg_equal = bits(&gpu.read_f32(&fused_tg).unwrap()) == bits(&reference_act);
            assert!(
                gate_equal && fused_equal && fused_tg_equal,
                "IQ2_S candidates must be bitwise equal"
            );
            // F64 reference for the sixteen-lane gate and fused SwiGLU candidates.
            let (half_g, half_act, base_g) = (
                gpu.read_f32(&half_gate).unwrap(),
                gpu.read_f32(&half_fused).unwrap(),
                gpu.read_f32(&bg).unwrap(),
            );
            let (mut gate_base_error, mut gate_half_error, mut act_base_error, mut act_half_error) =
                (0f64, 0f64, 0f64, 0f64);
            let gate_name = format!("blk.{layer}.ffn_gate_exps.weight");
            let up_name = format!("blk.{layer}.ffn_up_exps.weight");
            for (pair, &expert) in ids.iter().enumerate() {
                let rows = (expert as usize * ff..(expert as usize + 1) * ff).collect::<Vec<_>>();
                let gd = gpu
                    .dequant(&g.rows(&gpu.reads, &gate_name, &rows).unwrap())
                    .unwrap();
                let ud = gpu
                    .dequant(&g.rows(&gpu.reads, &up_name, &rows).unwrap())
                    .unwrap();
                let input = &x_input[pair / top * width..(pair / top + 1) * width];
                for row in 0..ff {
                    let dot = |m: &[f32]| {
                        m[row * width..(row + 1) * width]
                            .iter()
                            .zip(input)
                            .map(|(&a, &b)| f64::from(a) * f64::from(b))
                            .sum::<f64>()
                    };
                    let (ge, ue) = (dot(&gd), dot(&ud));
                    let ae = ge / (1.0 + (-ge).exp()) * ue;
                    let at = pair * ff + row;
                    gate_base_error = gate_base_error.max((f64::from(base_g[at]) - ge).abs());
                    gate_half_error = gate_half_error.max((f64::from(half_g[at]) - ge).abs());
                    act_base_error = act_base_error.max((f64::from(reference_act[at]) - ae).abs());
                    act_half_error = act_half_error.max((f64::from(half_act[at]) - ae).abs());
                    assert!(
                        (f64::from(half_g[at]) - ge).abs() <= 1e-4 + 5e-5 * ge.abs(),
                        "gate pair{pair} row{row}"
                    );
                    assert!(
                        (f64::from(half_act[at]) - ae).abs() <= 1e-4 + 1e-4 * ae.abs(),
                        "act pair{pair} row{row}"
                    );
                }
            }
            let (base_down, lanes_down) = (gpu.read_f32(&bd).unwrap(), gpu.read_f32(&cd).unwrap());
            // F64 reference over every routed expert row of the down projection.
            let (mut base_error, mut lanes_error, mut scale) = (0f64, 0f64, 0f64);
            for (pair, &expert) in ids.iter().enumerate() {
                let rows =
                    (expert as usize * width..(expert as usize + 1) * width).collect::<Vec<_>>();
                let decoded = gpu
                    .dequant(&g.rows(&gpu.reads, &down_name, &rows).unwrap())
                    .unwrap();
                for row in 0..width {
                    let expected = decoded[row * ff..(row + 1) * ff]
                        .iter()
                        .zip(&down_input[pair * ff..(pair + 1) * ff])
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum::<f64>();
                    let at = pair * width + row;
                    base_error = base_error.max((f64::from(base_down[at]) - expected).abs());
                    lanes_error = lanes_error.max((f64::from(lanes_down[at]) - expected).abs());
                    scale = scale.max(expected.abs());
                    assert!(
                        (f64::from(lanes_down[at]) - expected).abs()
                            <= 1e-4 + 5e-5 * expected.abs(),
                        "down pair{pair} row{row}"
                    );
                }
            }
            let down_equal = bits(&base_down) == bits(&lanes_down);
            println!(
                "ROUTED_VERIFY layer={layer} count={count} gate_half_us={:.1} fused_half_us={:.1} gate_f64 base={gate_base_error:.3e} half={gate_half_error:.3e} act_f64 base={act_base_error:.3e} half={act_half_error:.3e} separate_us={:.1} gate_base_us={:.1} gate_tg_us={:.1} fused_us={:.1} fused_tg_us={:.1} down_base_us={:.1} down_lanes8_us={:.1} down_bitwise={down_equal} down_f64_max_abs base={base_error:.3e} lanes8={lanes_error:.3e} scale={scale:.3e}",
                gpu_time(&gpu, &gate_half) * 1e6,
                gpu_time(&gpu, &fuse_half) * 1e6,
                gpu_time(&gpu, &separate) * 1e6,
                gpu_time(&gpu, &gate_base) * 1e6,
                gpu_time(&gpu, &gate_tg) * 1e6,
                gpu_time(&gpu, &fuse) * 1e6,
                gpu_time(&gpu, &fuse_tg) * 1e6,
                gpu_time(&gpu, &down_base) * 1e6,
                gpu_time(&gpu, &down_lanes) * 1e6,
            );
        }
    }
}

#[cfg(test)]
mod generic_flash_tests {
    use super::super::Gpu;

    /// Generic flash prefill matches dense attention, including head size 512.
    #[test]
    fn flash_prefill_matches_dense() {
        let gpu = Gpu::new().unwrap();
        let mut seed = 0x2545_f491_u32;
        let mut random = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 8) as f32 / (1 << 24) as f32 * 2. - 1.
                })
                .collect()
        };
        let (heads, kv) = (4, 2);
        for head in [256usize, 512] {
            for (position, count, peak) in
                [(0usize, 512usize, 1f32), (3072, 256, 1.), (3072, 256, 24.)]
            {
                let length = position + count;
                let qh: Vec<f32> = random(count * heads * head)
                    .iter()
                    .map(|x| x * peak)
                    .collect();
                let kh = random(length * kv * head);
                let vh = random(length * kv * head);
                let q = gpu.floats(&qh).unwrap();
                let (k, v) = (gpu.floats(&kh).unwrap(), gpu.floats(&vh).unwrap());
                let rows = heads * count;
                let scores = gpu.output(rows * length).unwrap();
                let probabilities = gpu.output(rows * length).unwrap();
                let dense = gpu.output(rows * head).unwrap();
                let flash = gpu.output(rows * head).unwrap();
                let p = [head, heads, kv, length, position].map(|x| x as u32);
                let cmd = gpu.command().unwrap();
                gpu.op(
                    &cmd,
                    "attention_scores_dense_batch",
                    &[&q, &k, &scores],
                    &p,
                    length,
                    rows,
                )
                .unwrap();
                gpu.op(
                    &cmd,
                    "softmax_dense_batch",
                    &[&scores, &probabilities],
                    &[length as u32, heads as u32, position as u32],
                    rows,
                    1,
                )
                .unwrap();
                gpu.op(
                    &cmd,
                    "attention_values_dense_batch",
                    &[&probabilities, &v, &dense],
                    &p,
                    head,
                    rows,
                )
                .unwrap();
                gpu.op(
                    &cmd,
                    &format!("flash_prefill_{head}"),
                    &[&q, &k, &v, &flash],
                    &[head, heads, kv, length, position, count, 0].map(|x| x as u32),
                    count.div_ceil(8) * 4,
                    heads,
                )
                .unwrap();
                gpu.submit(&cmd);
                gpu.wait(cmd).unwrap();
                let (a, b) = (gpu.read_f32(&dense).unwrap(), gpu.read_f32(&flash).unwrap());
                let error: f64 = a
                    .iter()
                    .zip(&b)
                    .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
                    .sum();
                let energy: f64 = a.iter().map(|&x| f64::from(x).powi(2)).sum();
                let relative = (error / energy).sqrt();
                // Double-precision reference for the last row of each head.
                let t = count - 1;
                let mut worst = [0f64; 2];
                for h in 0..heads {
                    let kh_ = h / (heads / kv);
                    let qrow = &qh[(t * heads + h) * head..][..head];
                    let n = position + t + 1;
                    let s: Vec<f64> = (0..n)
                        .map(|j| {
                            let krow = &kh[(j * kv + kh_) * head..][..head];
                            qrow.iter()
                                .zip(krow)
                                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                                .sum::<f64>()
                                / (head as f64).sqrt()
                        })
                        .collect();
                    let m = s.iter().cloned().fold(f64::MIN, f64::max);
                    let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
                    let z: f64 = e.iter().sum();
                    for d in 0..head {
                        let y: f64 = (0..n)
                            .map(|j| e[j] / z * f64::from(vh[(j * kv + kh_) * head + d]))
                            .sum();
                        let i = (t * heads + h) * head + d;
                        worst[0] = worst[0].max((f64::from(a[i]) - y).abs());
                        worst[1] = worst[1].max((f64::from(b[i]) - y).abs());
                    }
                }
                assert!(
                    relative < 2e-5,
                    "head {head} position {position}: relative L2 {relative}"
                );
                assert!(
                    worst[1] < 2e-5,
                    "head {head} position {position}: flash error {} vs f64",
                    worst[1]
                );
            }
        }
    }
}
