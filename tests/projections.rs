//! Real quantized rows exercise every specialized matrix-vector pipeline.
use ingot::{gguf::Gguf, metal::Gpu};
use std::{collections::BTreeSet, path::PathBuf};

#[test]
#[ignore = "requires QWEN_MODEL and GLM_MODEL plus Apple Silicon; reads a few real rows"]
fn all_real_formats_project_against_double_precision() {
    let gpu = Gpu::new().unwrap();
    let read_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    let mut checked = BTreeSet::new();
    for variable in ["QWEN_MODEL", "GLM_MODEL"] {
        let path = PathBuf::from(std::env::var_os(variable).expect(variable));
        let mut model = Gguf::open(&path).unwrap();
        let mut tensors: Vec<_> = model
            .tensors
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect();
        tensors.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, tensor) in tensors {
            if checked.contains(&tensor.kind) {
                continue;
            }
            let rows = tensor.bytes / tensor.stride;
            let matrix = model
                .rows(&read_pool, &name, if rows > 7 { &[0, 7] } else { &[0] })
                .unwrap();
            let decoded = gpu.dequant(&matrix).unwrap();
            let input: Vec<_> = (0..matrix.cols)
                .map(|i| ((i * 17 % 31) as f32 - 15.) * 0.03125)
                .collect();
            let actual = gpu.project(&matrix, &input).unwrap();
            for (row, &value) in actual.iter().enumerate() {
                let expected: f64 = decoded[row * matrix.cols..(row + 1) * matrix.cols]
                    .iter()
                    .zip(&input)
                    .map(|(&w, &x)| f64::from(w) * f64::from(x))
                    .sum();
                assert!(
                    (f64::from(value) - expected).abs() <= 1e-4 + 5e-5 * expected.abs(),
                    "type {} row {row}: {value} != {expected}",
                    tensor.kind
                );
            }
            checked.insert(tensor.kind);
            println!("type {}: {} real rows passed", tensor.kind, actual.len());
        }
    }
    assert_eq!(
        checked,
        BTreeSet::from([0, 1, 8, 10, 14, 16, 20, 21, 22, 23, 29, 30, 42])
    );
}
