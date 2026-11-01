//! Optional stage-boundary counters; no GPU submissions or waits live here.
//! API references: https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers
//! https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time
use super::*;
use objc2_foundation::NSRange;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

// Timestamp results are 8 bytes: stay within Apple's 32 KiB / 32-buffer limits.
const SAMPLES_PER_BUFFER: usize = 4096;
const MAX_BUFFERS: usize = 32;
type SampleBuffer = Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>;

#[derive(Default)]
struct KernelTotal {
    count: u64,
    ns: f64,
}
#[derive(Default)]
struct PhaseTotal {
    commands: u64,
    ns: f64,
    dropped_samples: u64,
    failed_samples: u64,
}

pub(super) struct KernelProfile {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    counter_set: Retained<ProtocolObject<dyn MTLCounterSet>>,
    output: File,
    free: Vec<SampleBuffer>,
    allocated: usize,
    kernels: BTreeMap<(&'static str, String), KernelTotal>,
    phases: BTreeMap<&'static str, PhaseTotal>,
}
impl KernelProfile {
    pub(super) fn new(device: &ProtocolObject<dyn MTLDevice>, path: &Path) -> Result<Self> {
        if !device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary) {
            return Err(io::Error::other(
                "INGOT_KERNEL_PROFILE requires Metal AtStageBoundary counter sampling",
            ));
        }
        let counter_set = device
            .counterSets()
            .and_then(|sets| {
                sets.iter().find(|set| {
                    // SAFETY: Metal's exported constant NSStrings live for the process.
                    set.name()
                        .isEqualToString(unsafe { MTLCommonCounterSetTimestamp })
                        && set.counters().iter().any(|counter| {
                            counter
                                .name()
                                .isEqualToString(unsafe { MTLCommonCounterTimestamp })
                        })
                })
            })
            .ok_or_else(|| io::Error::other("Metal timestamp counter set unavailable"))?;
        let output = File::create(path).map_err(|e| {
            io::Error::other(format!("kernel profile output {}: {e}", path.display()))
        })?;
        Ok(Self {
            device: Retained::from(device),
            counter_set,
            output,
            free: Vec::new(),
            allocated: 0,
            kernels: BTreeMap::new(),
            phases: BTreeMap::new(),
        })
    }

    fn buffer(&mut self) -> Result<Option<SampleBuffer>> {
        if let Some(buffer) = self.free.pop() {
            return Ok(Some(buffer));
        }
        if self.allocated == MAX_BUFFERS {
            return Ok(None);
        }
        let descriptor = MTLCounterSampleBufferDescriptor::new();
        descriptor.setCounterSet(Some(&self.counter_set));
        descriptor.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: a bounded 32 KiB timestamp sample buffer.
        unsafe { descriptor.setSampleCount(SAMPLES_PER_BUFFER) };
        let buffer = self
            .device
            .newCounterSampleBufferWithDescriptor_error(&descriptor)
            .map_err(|e| io::Error::other(format!("Metal counter sample buffer: {e}")))?;
        self.allocated += 1;
        Ok(Some(buffer))
    }

    fn write_json(&self) -> Result<()> {
        let mut out = BufWriter::new(&self.output);
        writeln!(out, "{{\"kernels\":[")?;
        for (i, ((phase, label), total)) in self.kernels.iter().enumerate() {
            // Both strings originate in our static stage/function names, never user input.
            writeln!(
                out,
                "{}{{\"phase\":\"{phase}\",\"label\":\"{label}\",\"count\":{},\"total_gpu_ns\":{},\"average_gpu_ns\":{}}}",
                if i == 0 { "" } else { "," },
                total.count,
                total.ns,
                total.ns / total.count as f64
            )?;
        }
        writeln!(out, "],\"phases\":{{")?;
        for (i, (phase, total)) in self.phases.iter().enumerate() {
            writeln!(
                out,
                "{}\"{phase}\":{{\"command_buffers\":{},\"command_buffer_gpu_ns\":{},\"dropped_samples\":{},\"failed_samples\":{}}}",
                if i == 0 { "" } else { "," },
                total.commands,
                total.ns,
                total.dropped_samples,
                total.failed_samples
            )?;
        }
        writeln!(out, "}}}}")?;
        out.flush()
    }
}
impl Drop for KernelProfile {
    fn drop(&mut self) {
        if let Err(error) = self.write_json() {
            eprintln!("kernel profile JSON write failed: {error}");
        }
    }
}

fn timestamps(device: &ProtocolObject<dyn MTLDevice>) -> (u64, u64) {
    let (mut cpu, mut gpu) = (0, 0);
    // SAFETY: two live writable MTLTimestamp values, copied synchronously.
    unsafe {
        device.sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu), NonNull::from(&mut gpu))
    };
    (cpu, gpu)
}

struct Chunk {
    buffer: SampleBuffer,
    // One entry per encoder; concurrent dispatch labels are combined at resolve.
    labels: Vec<Vec<String>>,
}
pub(super) struct CommandProfile {
    owner: Rc<RefCell<KernelProfile>>,
    phase: &'static str,
    before: (u64, u64),
    chunks: Vec<Chunk>,
    last_sampled: bool,
    dropped_samples: u64,
}
impl CommandProfile {
    pub(super) fn new(owner: Rc<RefCell<KernelProfile>>, phase: &'static str) -> Self {
        let before = timestamps(&owner.borrow().device);
        Self {
            owner,
            phase,
            before,
            chunks: Vec::new(),
            last_sampled: false,
            dropped_samples: 0,
        }
    }

    pub(super) fn descriptor(
        &mut self,
        dispatch: MTLDispatchType,
        label: Option<&str>,
    ) -> Result<Retained<MTLComputePassDescriptor>> {
        let descriptor = MTLComputePassDescriptor::new();
        descriptor.setDispatchType(dispatch);
        if self
            .chunks
            .last()
            .is_none_or(|chunk| chunk.labels.len() * 2 == SAMPLES_PER_BUFFER)
        {
            if let Some(buffer) = self.owner.borrow_mut().buffer()? {
                self.chunks.push(Chunk {
                    buffer,
                    labels: Vec::new(),
                });
            } else {
                // Never submit/wait early or overwrite in-flight samples to make room.
                self.dropped_samples += 2;
                self.last_sampled = false;
                return Ok(descriptor);
            }
        }
        let chunk = self.chunks.last_mut().unwrap();
        let start = chunk.labels.len() * 2;
        chunk
            .labels
            .push(label.into_iter().map(str::to_owned).collect());
        self.last_sampled = true;
        // SAFETY: attachment 0 exists; both indices fit this retained sample buffer.
        unsafe {
            let attachment = descriptor
                .sampleBufferAttachments()
                .objectAtIndexedSubscript(0);
            attachment.setSampleBuffer(Some(&chunk.buffer));
            attachment.setStartOfEncoderSampleIndex(start);
            attachment.setEndOfEncoderSampleIndex(start + 1);
        }
        Ok(descriptor)
    }

    pub(super) fn dispatch_label(&mut self, label: &str) {
        if self.last_sampled {
            self.chunks
                .last_mut()
                .unwrap()
                .labels
                .last_mut()
                .unwrap()
                .push(label.to_owned());
        }
    }

    // Called only after the existing completion wait (also from Command::drop).
    pub(super) fn finish(&mut self, cmd: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<()> {
        let mut owner = self.owner.borrow_mut();
        let after = timestamps(&owner.device);
        let phase = owner.phases.entry(self.phase).or_default();
        phase.commands += 1;
        phase.ns += (cmd.GPUEndTime() - cmd.GPUStartTime()).max(0.0) * 1e9;
        phase.dropped_samples += self.dropped_samples;
        let result = if cmd.status() == MTLCommandBufferStatus::Completed {
            self.resolve(after)
        } else {
            Err(io::Error::other("Metal kernel profile command failed"))
        };
        match result {
            Ok(samples) => {
                for (label, ns) in samples {
                    let total = owner.kernels.entry((self.phase, label)).or_default();
                    total.count += 1;
                    total.ns += ns;
                }
                Ok(())
            }
            Err(error) => {
                owner.phases.get_mut(self.phase).unwrap().failed_samples += self
                    .chunks
                    .iter()
                    .map(|chunk| chunk.labels.len() as u64 * 2)
                    .sum::<u64>();
                Err(error)
            }
        }
    }

    fn resolve(&self, after: (u64, u64)) -> Result<Vec<(String, f64)>> {
        let cpu_delta = after.0.checked_sub(self.before.0).filter(|&n| n > 0);
        let gpu_delta = after.1.checked_sub(self.before.1).filter(|&n| n > 0);
        let (Some(cpu_delta), Some(gpu_delta)) = (cpu_delta, gpu_delta) else {
            return Err(io::Error::other(
                "Metal kernel profile timestamp calibration failed",
            ));
        };
        // sampleTimestamps reports the CPU side in nanoseconds on Apple Silicon
        // (applying mach_timebase_info again inflated samples by 125/3).
        let ns_per_tick = cpu_delta as f64 / gpu_delta as f64;
        let mut samples = Vec::new();
        for chunk in &self.chunks {
            let count = chunk.labels.len() * 2;
            // SAFETY: these slots were allocated above and the command has completed.
            let data = unsafe { chunk.buffer.resolveCounterRange(NSRange::new(0, count)) }
                .ok_or_else(|| io::Error::other("Metal counter resolve failed"))?;
            // SAFETY: immutable NSData remains retained throughout this CPU-only read.
            let bytes = unsafe { data.as_bytes_unchecked() };
            if bytes.len() != count * std::mem::size_of::<MTLCounterResultTimestamp>() {
                return Err(io::Error::other("Metal counter resolve size mismatch"));
            }
            for (i, labels) in chunk.labels.iter().enumerate() {
                let read = |index: usize| {
                    // SAFETY: checked NSData length above; unaligned read avoids assuming NSData alignment.
                    unsafe {
                        bytes
                            .as_ptr()
                            .cast::<MTLCounterResultTimestamp>()
                            .add(index)
                            .read_unaligned()
                            .timestamp
                    }
                };
                let (start, end) = (read(i * 2), read(i * 2 + 1));
                if start == MTLCounterErrorValue || end == MTLCounterErrorValue || end < start {
                    return Err(io::Error::other(
                        "Metal kernel profile invalid timestamp sample",
                    ));
                }
                let mut labels = labels.clone();
                labels.sort_unstable();
                labels.dedup();
                samples.push((labels.join("+"), (end - start) as f64 * ns_per_tick));
            }
        }
        Ok(samples)
    }
}
impl Drop for CommandProfile {
    fn drop(&mut self) {
        self.owner
            .borrow_mut()
            .free
            .extend(self.chunks.drain(..).map(|chunk| chunk.buffer));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Apple Silicon Metal stage-boundary counters; run alone"]
    fn kernel_profile_stage_boundary_counters() {
        const TEST: &str = "metal::profile::tests::kernel_profile_stage_boundary_counters";
        if std::env::var_os("INGOT_KERNEL_PROFILE_TEST_CHILD").is_some() {
            let gpu = Gpu::new().unwrap();
            let gate = gpu.floats(&[1.0; 32]).unwrap();
            let up = gpu.floats(&[2.0; 32]).unwrap();
            let out = gpu.output(32).unwrap();
            let encode = |cmd: &Command| {
                gpu.encode(
                    cmd,
                    &gpu.swiglu,
                    &[&gate, &up, &out],
                    &[32],
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    1,
                )
                .unwrap();
            };
            gpu.set_profile_phase("prefill");
            let cmd = gpu.command().unwrap();
            for _ in 0..3 {
                encode(&cmd);
            }
            gpu.complete(cmd).unwrap();

            gpu.set_profile_phase("decode");
            let cmd = gpu.command().unwrap();
            // Cross a sample-buffer boundary without submitting or losing samples.
            for _ in 0..=SAMPLES_PER_BUFFER / 2 {
                encode(&cmd);
            }
            let weight = gpu.floats(&[1.0; 32]).unwrap();
            let projected = gpu.output(1).unwrap();
            gpu.encode(
                &cmd,
                &gpu.projections[&("quant_matvec", 0)],
                &[&weight, &gate, &projected],
                &[32, 1, 128, 0, 0],
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                1,
            )
            .unwrap();
            gpu.submit(&cmd);
            gpu.wait(cmd).unwrap();

            gpu.set_profile_phase("draft");
            let cmd = gpu.command().unwrap();
            gpu.independent_projections(&cmd, || {
                // Distinct outputs: truly independent dispatches, with a duplicate label.
                gpu.activation(&cmd, &gate, &up, 32)?;
                gpu.activation(&cmd, &gate, &up, 32)?;
                gpu.op(&cmd, "copy_into", &[&gate, &out], &[32, 0], 1, 1)
            })
            .unwrap();
            gpu.complete(cmd).unwrap();

            gpu.set_profile_phase("verify");
            let cmd = gpu.command().unwrap();
            encode(&cmd);
            encode(&cmd);
            gpu.submit(&cmd);
            // Exercise the existing drop wait; it must resolve and count exactly once.
            drop(cmd);
            let scope = gpu.profile_phase("draft");
            assert_eq!(gpu.profile_phase.get(), "draft");
            drop(scope);
            assert_eq!(gpu.profile_phase.get(), "verify");
            return; // Gpu/profile drop writes the JSON before this child exits.
        }

        // Child-local environment avoids unsafe process-wide set_var in a test harness.
        let path = std::env::temp_dir().join(format!(
            "ingot-kernel-profile-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", TEST, "--nocapture"])
            .env("INGOT_KERNEL_PROFILE", &path)
            .env("INGOT_KERNEL_PROFILE_TEST_CHILD", "1")
            .status()
            .unwrap();
        assert!(status.success());
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let kernels = json["kernels"].as_array().unwrap();
        for (phase, label, count) in [
            ("prefill", "swiglu", 3),
            ("decode", "swiglu", SAMPLES_PER_BUFFER as u64 / 2 + 1),
            ("decode", "quant_matvec/k0", 1),
            ("draft", "copy_into+swiglu", 1),
            ("verify", "swiglu", 2),
        ] {
            let kernel = kernels
                .iter()
                .find(|k| k["phase"] == phase && k["label"] == label)
                .unwrap();
            assert_eq!(kernel["count"].as_u64(), Some(count));
            let ns = kernel["total_gpu_ns"].as_f64().unwrap();
            assert!(ns > 0.0);
            let average = kernel["average_gpu_ns"].as_f64().unwrap();
            assert!((average - ns / count as f64).abs() <= ns * 1e-12);
        }
        assert_eq!(kernels.len(), 5);
        for phase in ["prefill", "decode", "draft", "verify"] {
            let total = &json["phases"][phase];
            assert_eq!(total["command_buffers"].as_u64(), Some(1));
            assert_eq!(total["dropped_samples"].as_u64(), Some(0));
            assert_eq!(total["failed_samples"].as_u64(), Some(0));
            let kernel_ns: f64 = kernels
                .iter()
                .filter(|k| k["phase"] == phase)
                .map(|k| k["total_gpu_ns"].as_f64().unwrap())
                .sum();
            // GPUStartTime/GPUEndTime advance in 24 MHz ticks; allow that granularity.
            assert!(
                kernel_ns <= total["command_buffer_gpu_ns"].as_f64().unwrap() * 1.01 + 100.0,
                "{phase}: sampled encoders exceed command-buffer GPU time"
            );
        }
        std::fs::remove_file(path).unwrap();
    }
}
