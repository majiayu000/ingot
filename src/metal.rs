//! The only unsafe boundary: typed host buffers and synchronous Metal execution.
use std::cell::RefCell;
use std::collections::HashMap;
use std::io;
use std::ops::Deref;
use std::ptr::NonNull;
use std::rc::{Rc, Weak};

use objc2::{
    class, msg_send,
    rc::Retained,
    runtime::{AnyObject, Bool, ProtocolObject},
};
#[link(name = "MetalPerformanceShaders", kind = "framework")]
unsafe extern "C" {}
use objc2_foundation::NSString;
use objc2_metal::*;

use crate::gguf::Matrix;
use crate::{Result, invalid};

#[path = "metal_generic.rs"]
mod generic_ops;
#[path = "metal_profile.rs"]
mod profile;
use profile::{CommandProfile, KernelProfile};

type NativeBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
#[derive(Clone)]
pub(crate) struct Buffer(Rc<Allocation>, usize, usize);
struct Allocation {
    // Drop Metal's view before releasing its backing mapping.
    raw: Option<NativeBuffer>,
    mapping: Option<memmap2::Mmap>,
    offset: usize,
    length: usize,
    pool: Weak<RefCell<Workspace>>,
    residency: Option<Retained<ProtocolObject<dyn MTLResidencySet>>>,
}
impl Deref for Buffer {
    type Target = ProtocolObject<dyn MTLBuffer>;
    fn deref(&self) -> &Self::Target {
        self.0.raw.as_ref().unwrap()
    }
}
impl Drop for Allocation {
    fn drop(&mut self) {
        if let Some(residency) = &self.residency {
            // Commands retain this allocation through completion. Remove Metal's
            // registration before dropping the last owner of the mapped pages.
            residency.removeAllocation(ProtocolObject::from_ref(&**self.raw.as_ref().unwrap()));
            residency.commit();
        }
        if let Some(pool) = self.pool.upgrade() {
            let raw = self.raw.take().unwrap();
            pool.borrow_mut()
                .free
                .entry(raw.length())
                .or_default()
                .push(raw);
        }
    }
}
impl Buffer {
    fn from_allocation(allocation: Allocation) -> Self {
        let length = allocation.length;
        Self(Rc::new(allocation), 0, length)
    }
    fn offset(&self) -> usize {
        self.0.offset + self.1
    }
    /// A shared, bounded tensor view. The backing allocation stays leased until
    /// every view and every in-flight command using it has been released.
    pub(crate) fn view_f32(&self, offset: usize, count: usize) -> Result<Self> {
        let start = offset
            .checked_mul(4)
            .ok_or_else(|| invalid("buffer view overflow"))?;
        let length = count
            .checked_mul(4)
            .ok_or_else(|| invalid("buffer view overflow"))?;
        if start > self.length() || length > self.length() - start {
            return Err(invalid("buffer view exceeds tensor"));
        }
        Ok(Self(self.0.clone(), self.1 + start, length))
    }
    pub(crate) fn contiguous_f32(inputs: &[Self], width: usize) -> Option<Self> {
        let first = inputs.first()?;
        let stride = width.checked_mul(4)?;
        let length = stride.checked_mul(inputs.len())?;
        if inputs.iter().enumerate().all(|(i, b)| {
            Rc::ptr_eq(&first.0, &b.0) && b.1 == first.1 + i * stride && b.length() >= stride
        }) {
            Some(Self(first.0.clone(), first.1, length))
        } else {
            None
        }
    }
    fn permanent(raw: NativeBuffer) -> Self {
        let length = raw.length();
        Self::from_allocation(Allocation {
            raw: Some(raw),
            mapping: None,
            offset: 0,
            length,
            pool: Weak::new(),
            residency: None,
        })
    }
    pub(crate) fn length(&self) -> usize {
        self.2
    }
    pub(crate) fn contents(&self) -> NonNull<std::ffi::c_void> {
        // SAFETY: every constructor validates offset + logical length within raw.
        unsafe {
            self.0
                .raw
                .as_ref()
                .unwrap()
                .contents()
                .byte_add(self.offset())
        }
    }
}
#[derive(Default)]
struct Workspace {
    limit: usize,
    managed: usize,
    peak: usize,
    allocations: u64,
    reuses: u64,
    free: HashMap<usize, Vec<NativeBuffer>>,
}
// Charge each allocation at the device page size, not its logical tensor length.
fn allocation_charge(length: usize) -> usize {
    length.div_ceil(16 * 1024) * (16 * 1024)
}

struct Pipeline {
    raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    name: String,
}
impl Deref for Pipeline {
    type Target = ProtocolObject<dyn MTLComputePipelineState>;
    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}
pub(crate) struct Command {
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    // Retain pool and mapping ownership through GPU completion, including errors.
    buffers: RefCell<Vec<Buffer>>,
    independent_encoder: RefCell<Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>>,
    failed: Rc<std::cell::Cell<bool>>,
    profile: RefCell<Option<CommandProfile>>,
}
impl Deref for Command {
    type Target = ProtocolObject<dyn MTLCommandBuffer>;
    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}
impl Command {
    fn finish_profile(&self) -> Result<()> {
        if let Some(mut profile) = self.profile.borrow_mut().take() {
            profile.finish(&self.raw)?;
        }
        Ok(())
    }
}
impl Drop for Command {
    fn drop(&mut self) {
        if matches!(
            self.raw.status(),
            MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled
        ) {
            self.raw.waitUntilCompleted();
        }
        if self.raw.status() == MTLCommandBufferStatus::Error {
            self.failed.set(true);
        }
        if self.profile.get_mut().is_some()
            && matches!(
                self.raw.status(),
                MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
            )
            && let Err(error) = self.finish_profile()
        {
            eprintln!("kernel profile completion failed: {error}");
        }
    }
}

pub struct Gpu {
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    residency_dirty: std::cell::Cell<bool>,
    reads: rayon::ThreadPool,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    projections: std::collections::HashMap<(&'static str, u32), Pipeline>,
    swiglu: Pipeline,
    combine: Pipeline,
    operations: std::collections::HashMap<&'static str, Pipeline>,
    failed: Rc<std::cell::Cell<bool>>,
    peak_bytes: std::cell::Cell<usize>,
    workspace: Rc<RefCell<Workspace>>,
    profile: Option<Rc<RefCell<KernelProfile>>>,
    profile_phase: std::cell::Cell<&'static str>,
}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.queue.removeResidencySet(&self.residency);
        self.residency.endResidency();
        self.residency.removeAllAllocations();
        self.residency.commit();
    }
}

pub(crate) struct ProfilePhase<'a> {
    phase: &'a std::cell::Cell<&'static str>,
    previous: &'static str,
}
impl Drop for ProfilePhase<'_> {
    fn drop(&mut self) {
        self.phase.set(self.previous);
    }
}
impl Gpu {
    pub(crate) fn set_profile_phase(&self, phase: &'static str) {
        self.profile_phase.set(phase);
    }
    pub(crate) fn profile_phase(&self, phase: &'static str) -> ProfilePhase<'_> {
        ProfilePhase {
            phase: &self.profile_phase,
            previous: self.profile_phase.replace(phase),
        }
    }
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| io::Error::other("Metal device unavailable"))?;
        let profile = std::env::var_os("INGOT_KERNEL_PROFILE")
            .map(|path| KernelProfile::new(&device, std::path::Path::new(&path)))
            .transpose()?
            .map(|profile| Rc::new(RefCell::new(profile)));
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| io::Error::other("Metal command queue unavailable"))?;
        let descriptor = MTLResidencySetDescriptor::new();
        unsafe {
            descriptor.setInitialCapacity(1024);
        }
        let residency = device
            .newResidencySetWithDescriptor_error(&descriptor)
            .map_err(|e| io::Error::other(format!("Metal weight residency set: {e}")))?;
        queue.addResidencySet(&residency);
        let source = format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
            include_str!("../kernels/quant_tables.metal"),
            include_str!("../kernels/iq3_mix.metal"),
            include_str!("../kernels/moe.metal"),
            include_str!("../kernels/qwen.metal"),
            include_str!("../kernels/glm.metal"),
            include_str!("../kernels/decode.metal"),
            include_str!("../kernels/prefill.metal"),
            include_str!("../kernels/generic.metal"),
            include_str!("../kernels/generic_arch.metal"),
            include_str!("../kernels/vision.metal")
        );
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options))
            .map_err(|e| io::Error::other(format!("Metal compile: {e}")))?;
        let pipeline = |name: &str| -> Result<Pipeline> {
            let f = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| io::Error::other(format!("missing Metal function: {name}")))?;
            device
                .newComputePipelineStateWithFunction_error(&f)
                .map(|raw| Pipeline {
                    raw,
                    name: match name {
                        "iq4_vector" => format!("{name}/k23"),
                        "q8_dense1" | "q8_dense2" | "q8_dense4" => format!("{name}/k8"),
                        "bf16_dense1" | "bf16_dense2" | "bf16_dense4" | "bf16_hc_split"
                        | "hc_down_inject" | "hc_up_collapse" => format!("{name}/k30"),
                        "project_routed_lut" => format!("{name}/k16"),
                        "project_routed_q2_packed" | "routed_q2_lanes8" => format!("{name}/k42"),
                        "routed_iq2s_tg"
                        | "routed_gate_up_iq2s"
                        | "routed_gate_up_iq2s_tg"
                        | "routed_iq2s_half"
                        | "routed_gate_up_iq2s_half" => format!("{name}/k22"),
                        "q4k_vector" | "q4k_vector2" | "q4k_vector3" | "q4k_vector4" => {
                            format!("{name}/k12")
                        }
                        "q5k_vector" | "q5k_vector2" | "q5k_vector3" | "q5k_vector4" => {
                            format!("{name}/k13")
                        }
                        "routed_iq2xs" => format!("{name}/k17"),
                        "routed_iq3xxs" => format!("{name}/k18"),
                        "routed_iq4nl" => format!("{name}/k20"),
                        "routed_iq3s" => format!("{name}/k21"),
                        _ => name.to_owned(),
                    },
                })
                .map_err(|e| io::Error::other(format!("Metal pipeline: {e}")))
        };
        let mut projections = std::collections::HashMap::new();
        for kind in [
            0u32, 1, 2, 8, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 30, 42,
        ] {
            let constants = MTLFunctionConstantValues::new();
            // SAFETY: constant 0 is a uint copied synchronously from a live u32.
            unsafe {
                constants.setConstantValue_type_atIndex(
                    NonNull::from(&kind).cast(),
                    MTLDataType::UInt,
                    0,
                );
            }
            for name in [
                "quant_matvec",
                "quant_matmul",
                "project_experts",
                "project_routed_batch",
                "quant_gemm",
                "quant_gemm_8",
                "quant_gemm_wide",
                "quant_gemm_wide_f32",
                "quant_expert_gemm",
                "quant_expert_gather",
                "quant_expert_packed",
                "quant_expert_gate_up",
                "routed_vec4",
                "decode_blocks16",
            ] {
                if name == "quant_gemm_wide_f32" && !matches!(kind, 8 | 30) {
                    continue;
                }
                let f = library
                    .newFunctionWithName_constantValues_error(&NSString::from_str(name), &constants)
                    .map_err(|e| io::Error::other(format!("Metal function {name}/{kind}: {e}")))?;
                let p = device
                    .newComputePipelineStateWithFunction_error(&f)
                    .map_err(|e| io::Error::other(format!("Metal pipeline {name}/{kind}: {e}")))?;
                let threads = if name == "quant_expert_gate_up" {
                    256
                } else {
                    128
                };
                if p.threadExecutionWidth() != 32 || p.maxTotalThreadsPerThreadgroup() < threads {
                    return Err(io::Error::other(
                        "quantized projections require four 32-lane Metal SIMD groups",
                    ));
                }
                projections.insert(
                    (name, kind),
                    Pipeline {
                        raw: p,
                        name: format!("{name}/k{kind}"),
                    },
                );
            }
        }
        let swiglu = pipeline("swiglu")?;
        let combine = pipeline("combine")?;
        let mut operations = std::collections::HashMap::new();
        for name in [
            "rms_wide",
            "hc_add_rms",
            "expert_layout",
            "expert_pairs",
            "iq3s_vector",
            "q6k_vector",
            "q4k_vector",
            "q4k_vector2",
            "q4k_vector3",
            "q4k_vector4",
            "q5k_vector",
            "q5k_vector2",
            "q5k_vector3",
            "q5k_vector4",
            "routed_iq2xs",
            "routed_iq3xxs",
            "routed_iq4nl",
            "routed_iq3s",
            "iq4nl_vector",
            "iq4_vector2",
            "iq4_vector3",
            "iq4_vector4",
            "route_top10",
            "iq4_vector",
            "bf16_dense1",
            "bf16_dense2",
            "bf16_dense4",
            "q8_dense1",
            "q8_dense2",
            "q8_dense4",
            "project_routed_lut",
            "project_routed_q2_packed",
            "routed_q2_lanes8",
            "routed_iq2s_tg",
            "routed_gate_up_iq2s",
            "routed_gate_up_iq2s_tg",
            "routed_iq2s_half",
            "routed_gate_up_iq2s_half",
            "attention_prefill_tiles",
            "attention_prefill_tail",
            "bf16_hc_split",
            "hc_down_inject",
            "hc_up_collapse",
            "hc_collapse_sigmoid",
            "copy_into",
            "greedy_select",
            "mtp_eh_input",
            "attention_online_parts",
            "attention_online_merge",
            "attention_scores_dense_batch",
            "softmax_dense_batch",
            "attention_values_dense_batch",
            "rope_batch",
            "extract_gate_batch",
            "pool_index_batch",
            "index_score_batch",
            "index_mask_batch",
            "index_mask_sorted",
            "compact_mask_ordered",
            "compact_mask",
            "attention_scores_compact",
            "softmax_compact",
            "attention_values_compact",
            "embedding_batch",
            "repeat_emb_batch",
            "hc_collapse_batch",
            "hc_add_batch",
            "convolution_parallel",
            "convolution_history",
            "convolution_batch",
            "rms_tokens",
            "delta_net_batch",
            "delta_gates_batch",
            "ple_gate_batch",
            "route_gpu_batch",
            "gather_rows",
            "scatter_rows",
            "combine_batch",
            "combine_packed",
            "embedding",
            "copy_range",
            "read_rows",
            "rms",
            "unary",
            "hc_collapse",
            "hc_add",
            "repeat_emb",
            "route_gpu",
            "mul_gate",
            "convolution",
            "delta_net",
            "ple_gate",
            "add_three",
            "add_rows",
            "bias_rows",
            "rope_generic",
            "scale_rows",
            "softcap_rows",
            "gelu_mul",
            "swiglu_fused",
            "copy_columns",
            "attention_scores_window",
            "moe_sum",
            "rope",
            "extract_gate",
            "cache_write",
            "pool_index",
            "index_score",
            "index_mask",
            "attention_scores",
            "softmax_rows",
            "attention_values",
            "glm_hc_gates",
            "glm_hc_pre",
            "glm_hc_post",
            "glm_hc_mean",
            "glm_add",
            "glm_kda",
            "glm_route",
            "glm_swiglu",
            "glm_merge",
            "glm_layer_norm",
            "glm_pool",
            "glm_index_score",
            "glm_index_mask",
            "glm_mla_scores",
            "glm_mla_scores_compact",
            "glm_mla_online_parts",
            "glm_mla_online_merge",
            "glm_index_mask_sorted",
            "vit_norm",
            "vit_bias_gelu",
            "vit_bias_add",
            "vit_qkv",
            "vit_attention",
            "rope_rows",
        ] {
            operations.insert(name, pipeline(name)?);
        }
        for name in generic_ops::OPERATIONS {
            operations.insert(name, pipeline(name)?);
        }
        Ok(Self {
            residency,
            residency_dirty: std::cell::Cell::new(false),
            reads: rayon::ThreadPoolBuilder::new()
                .num_threads(8)
                .stack_size(512 * 1024)
                .thread_name(|i| format!("expert-read-{i}"))
                .build()
                .map_err(io::Error::other)?,
            device,
            queue,
            projections,
            swiglu,
            combine,
            operations,
            failed: Rc::new(std::cell::Cell::new(false)),
            peak_bytes: std::cell::Cell::new(0),
            workspace: Rc::new(RefCell::new(Workspace::default())),
            profile,
            profile_phase: std::cell::Cell::new("decode"),
        })
    }

    pub(crate) fn empty_bytes(&self, length: usize) -> Result<Buffer> {
        if length == 0 || length > self.device.maxBufferLength() {
            return Err(invalid("Metal buffer exceeds device limit"));
        }
        let buffer = self
            .device
            .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| io::Error::other("Metal allocation failed"))?;
        self.observe_allocation();
        Ok(Buffer::permanent(buffer))
    }
    pub(crate) fn load_weight(
        &self,
        g: &mut crate::gguf::Gguf,
        t: &crate::gguf::Tensor,
    ) -> Result<Buffer> {
        let (_, offset, length) = t.mapping_layout()?;
        if length > self.device.maxBufferLength() {
            return Err(invalid("mapped weight exceeds Metal buffer limit"));
        }
        let mapping = g.map_tensor(t)?;
        // SAFETY: page-aligned read-only model storage, never bound as a kernel
        // output. Allocation owns the mapping; encoded commands retain Allocation
        // until completion. Metal gets no deallocator because Mmap owns unmapping.
        let raw = unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    NonNull::new(mapping.as_ptr().cast_mut().cast()).unwrap(),
                    length,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
        .ok_or_else(|| io::Error::other("Metal mapped weight creation failed"))?;
        self.observe_allocation();
        let buffer = Buffer::from_allocation(Allocation {
            raw: Some(raw),
            mapping: Some(mapping),
            offset,
            length: t.bytes,
            pool: Weak::new(),
            residency: Some(self.residency.clone()),
        });
        self.residency
            .addAllocation(ProtocolObject::from_ref(&*buffer));
        self.residency_dirty.set(true);
        Ok(buffer)
    }
    pub(crate) fn load_range(
        &self,
        g: &mut crate::gguf::Gguf,
        t: &crate::gguf::Tensor,
        source: usize,
        buffer: &Buffer,
        dest: usize,
        length: usize,
    ) -> Result<()> {
        if dest
            .checked_add(length)
            .is_none_or(|end| end > buffer.length())
        {
            return Err(invalid("slot buffer bounds"));
        }
        // SAFETY: the cache owns this byte range. The scheduler excludes the in-flight
        // slot, and waits for its GPU readers before reuse. Other slots are disjoint.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                buffer.contents().as_ptr().cast::<u8>().add(dest),
                length,
            )
        };
        g.read_range(t, source, bytes)
    }
    pub fn read_rows(
        &self,
        g: &mut crate::gguf::Gguf,
        name: &str,
        rows: &[usize],
    ) -> Result<Matrix> {
        g.rows(&self.reads, name, rows)
    }
    // A cache miss fills only the three disjoint tensors of one unpinned slot.
    pub(crate) fn load_expert(
        &self,
        g: &mut crate::gguf::Gguf,
        tensors: &[crate::gguf::Tensor; 3],
        buffers: &[Buffer; 3],
        count: usize,
        id: usize,
        slot: usize,
    ) -> Result<()> {
        let mut ranges = Vec::with_capacity(3);
        for (t, buffer) in tensors.iter().zip(buffers) {
            let size = t.bytes / count;
            let dest = slot * size;
            if dest
                .checked_add(size)
                .is_none_or(|end| end > buffer.length())
            {
                return Err(invalid("slot buffer bounds"));
            }
            // SAFETY: ExpertCache creates three distinct permanent buffers. This slot
            // is unpinned, all prior GPU readers have finished, and read_ranges joins
            // every worker before the slot is made valid or an error is returned.
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    buffer.contents().as_ptr().cast::<u8>().add(dest),
                    size,
                )
            };
            ranges.push((t, id * size, bytes));
        }
        g.read_ranges(&self.reads, ranges)
    }
    pub(crate) fn read_u32(&self, buffer: &Buffer, count: usize) -> Result<Vec<u32>> {
        if count.checked_mul(4).is_none_or(|n| n > buffer.length()) {
            return Err(invalid("readback bounds"));
        }
        // SAFETY: private callers wait for command completion before readback.
        Ok(
            unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u32>(), count) }
                .to_vec(),
        )
    }
    pub(crate) fn submit(&self, cmd: &Command) {
        self.residency.requestResidency();
        cmd.commit();
    }
    pub(crate) fn wait(&self, cmd: Command) -> Result<()> {
        cmd.waitUntilCompleted();
        if cmd.status() != MTLCommandBufferStatus::Completed {
            self.failed.set(true);
            return Err(io::Error::other(format!(
                "Metal execution failed: {:?}",
                cmd.error()
            )));
        }
        cmd.finish_profile()
    }
    pub(crate) fn zeros(&self, count: usize) -> Result<Buffer> {
        let b = self.empty_bytes(
            count
                .checked_mul(4)
                .ok_or_else(|| invalid("zero size overflow"))?,
        )?;
        // SAFETY: newly allocated shared resource has no GPU users.
        unsafe {
            std::ptr::write_bytes(b.contents().as_ptr().cast::<u8>(), 0, b.length());
        }
        Ok(b)
    }
    pub(crate) fn op(
        &self,
        cmd: &Command,
        name: &str,
        buffers: &[&Buffer],
        params: &[u32],
        width: usize,
        height: usize,
    ) -> Result<()> {
        let wide_point = width >= 8
            && matches!(
                name,
                "read_rows"
                    | "embedding"
                    | "embedding_batch"
                    | "repeat_emb"
                    | "repeat_emb_batch"
                    | "unary"
                    | "hc_collapse"
                    | "hc_collapse_batch"
                    | "hc_collapse_sigmoid"
                    | "hc_add"
                    | "hc_add_batch"
                    | "mul_gate"
                    | "copy_range"
                    | "convolution_parallel"
                    | "convolution_history"
                    | "copy_into"
                    | "extract_gate"
                    | "extract_gate_batch"
                    | "rope"
                    | "rope_batch"
                    | "cache_write"
                    | "pool_index"
                    | "pool_index_batch"
                    | "delta_gates_batch"
                    | "combine_batch"
                    | "combine_packed"
                    | "gather_rows"
                    | "scatter_rows"
                    | "mtp_eh_input"
                    | "add_three"
            );
        let wide_sort = matches!(
            name,
            "index_mask_sorted" | "hc_add_rms" | "expert_layout" | "expert_pairs"
        );
        let wide_rms = name == "rms" && params[0] >= 1024;
        let hc_projection = matches!(name, "hc_down_inject" | "hc_up_collapse");
        let wide_route =
            matches!(name, "route_gpu" | "route_gpu_batch") && params[0] == 512 && params[1] == 10;
        // The fused expert gate/up kernel runs eight SIMD groups per 64-row tile.
        let (width, gate_up) = if name == "quant_expert_gate_up" {
            (width.div_ceil(64) * 256, true)
        } else {
            (width, false)
        };
        let p = if wide_rms {
            self.operations.get("rms_wide")
        } else if wide_route {
            self.operations.get("route_top10")
        } else if name == "project_routed_batch" && params[3] == 29 {
            self.projections.get(&("routed_vec4", 29))
        } else if name == "project_routed_batch" && params[3] == 16 {
            self.operations.get("project_routed_lut")
        } else if name == "project_routed_batch" && params[3] == 42 {
            self.operations.get("routed_q2_lanes8")
        } else if name == "project_routed_batch" && params[3] == 17 {
            self.operations.get("routed_iq2xs")
        } else if name == "project_routed_batch" && params[3] == 18 {
            self.operations.get("routed_iq3xxs")
        } else if name == "project_routed_batch" && params[3] == 20 {
            self.operations.get("routed_iq4nl")
        } else if name == "project_routed_batch" && params[3] == 21 {
            self.operations.get("routed_iq3s")
        } else if matches!(
            name,
            "quant_matvec"
                | "quant_matmul"
                | "project_experts"
                | "project_routed_batch"
                | "quant_expert_gemm"
                | "quant_expert_gather"
                | "quant_expert_packed"
                | "quant_expert_gate_up"
        ) {
            self.projections.get(&(name, params[3]))
        } else {
            self.operations.get(name)
        }
        .ok_or_else(|| invalid("unknown Metal operation or projection format"))?;
        self.encode(
            cmd,
            p,
            buffers,
            params,
            MTLSize {
                width: if wide_sort {
                    width * 256
                } else if wide_point {
                    width * 32
                } else if wide_rms || wide_route || hc_projection {
                    width * 4
                } else {
                    width
                },
                height,
                depth: 1,
            },
            if wide_point || wide_sort || gate_up {
                256
            } else if wide_rms || wide_route || hc_projection {
                4
            } else if (name == "project_routed_batch"
                && matches!(params[3], 16 | 17 | 18 | 20 | 21 | 22 | 29 | 42))
                || matches!(
                    name,
                    "routed_q2_lanes8"
                        | "routed_iq2s_tg"
                        | "routed_gate_up_iq2s"
                        | "routed_gate_up_iq2s_tg"
                        | "routed_iq2s_half"
                        | "routed_gate_up_iq2s_half"
                )
            {
                16
            } else if matches!(
                name,
                "quant_expert_gemm" | "quant_expert_gather" | "quant_expert_packed"
            ) {
                64
            } else if matches!(
                name,
                "attention_prefill_tiles"
                    | "routed_q4k"
                    | "routed_q6k"
                    | "flash_prefill_64"
                    | "flash_prefill_64_tail"
                    | "flash_prefill_96"
                    | "flash_prefill_96_tail"
                    | "flash_prefill_128"
                    | "flash_prefill_128_tail"
                    | "flash_prefill_256"
                    | "flash_prefill_256_tail"
                    | "flash_prefill_512"
                    | "delta_net_batch"
                    | "attention_prefill_tail"
                    | "quant_matvec"
                    | "quant_matmul"
                    | "project_experts"
                    | "project_routed_batch"
                    | "quant_expert_gemm"
            ) {
                4
            } else {
                1
            },
        )
    }
    #[allow(clippy::too_many_arguments)] // Explicit kernel layout parameters.
    pub(crate) fn matvec(
        &self,
        cmd: &Command,
        w: &Buffer,
        cols: usize,
        rows: usize,
        stride: usize,
        kind: u32,
        x: &Buffer,
    ) -> Result<Buffer> {
        if matches!(kind, 2 | 8 | 12 | 13 | 14 | 20 | 21 | 23 | 30) {
            return self.matmul(cmd, w, cols, rows, stride, kind, 0, x, 1);
        }
        let out = self.output(rows)?;
        self.encode(
            cmd,
            self.projections
                .get(&("quant_matvec", kind))
                .ok_or_else(|| invalid("unsupported projection format"))?,
            &[w, x, &out],
            &[cols as u32, rows as u32, stride as u32, kind, 0],
            MTLSize {
                width: rows,
                height: 1,
                depth: 1,
            },
            4,
        )?;
        Ok(out)
    }
    #[allow(clippy::too_many_arguments)] // A quantized matrix, expert offset and token block.
    pub(crate) fn matmul(
        &self,
        cmd: &Command,
        w: &Buffer,
        cols: usize,
        rows: usize,
        stride: usize,
        kind: u32,
        expert: usize,
        x: &Buffer,
        count: usize,
    ) -> Result<Buffer> {
        let out = self.output(rows * count)?;
        if kind == 8 && count <= 4 && rows >= 128 {
            let nr = if cols >= 10240 && rows <= 512 {
                if count == 1 { 1 } else { 2 }
            } else {
                4
            };
            let kernel = match nr {
                1 => "q8_dense1",
                2 => "q8_dense2",
                _ => "q8_dense4",
            };
            self.encode(
                cmd,
                &self.operations[kernel],
                &[w, x, &out],
                &[
                    cols as u32,
                    rows as u32,
                    stride as u32,
                    kind,
                    expert as u32,
                    count as u32,
                ],
                MTLSize {
                    width: rows.div_ceil(4 * nr) * 4,
                    height: count,
                    depth: 1,
                },
                4,
            )?;
            return Ok(out);
        }
        if matches!(kind, 2 | 12 | 13 | 14 | 20 | 21 | 23) && count <= 4 {
            self.encode(
                cmd,
                &self.operations[match kind {
                    12 => ["q4k_vector", "q4k_vector2", "q4k_vector3", "q4k_vector4"][count - 1],
                    13 => ["q5k_vector", "q5k_vector2", "q5k_vector3", "q5k_vector4"][count - 1],
                    14 => "q6k_vector",
                    2 => "q4_0_vector",
                    20 => "iq4nl_vector",
                    21 => "iq3s_vector",
                    _ => ["iq4_vector", "iq4_vector2", "iq4_vector3", "iq4_vector4"][count - 1],
                }],
                &[w, x, &out],
                &[
                    cols as u32,
                    rows as u32,
                    stride as u32,
                    kind,
                    expert as u32,
                    count as u32,
                ],
                MTLSize {
                    width: rows.div_ceil(8) * 4,
                    height: if matches!(kind, 2 | 12 | 13 | 23) {
                        1
                    } else {
                        count
                    },
                    depth: 1,
                },
                4,
            )?;
            return Ok(out);
        }

        if kind == 30 && count <= 4 {
            let split =
                (cols == 10240 && (count <= 2 || rows < 128)) || (cols == 2560 && rows <= 512);
            let nr = if (cols == 2560 && matches!(rows, 2560 | 10240))
                || (cols == 10240 && rows >= 128 && count >= 3)
                || (cols == 2560 && rows > 128 && rows <= 512)
            {
                1
            } else if cols == 320 && count < 4 {
                2
            } else if cols == 2560 && rows >= 2560 && count < 4 {
                match count {
                    1 => 1,
                    2 => 2,
                    _ => 4,
                }
            } else {
                0
            };
            if split {
                self.encode(
                    cmd,
                    &self.operations["bf16_hc_split"],
                    &[w, x, &out],
                    &[cols as u32, rows as u32, stride as u32, expert as u32],
                    MTLSize {
                        width: rows * 4,
                        height: count,
                        depth: 1,
                    },
                    4,
                )?;
                return Ok(out);
            }
            if nr > 0 {
                let kernel = match nr {
                    1 => "bf16_dense1",
                    2 => "bf16_dense2",
                    _ => "bf16_dense4",
                };
                self.encode(
                    cmd,
                    &self.operations[kernel],
                    &[w, x, &out],
                    &[
                        cols as u32,
                        rows as u32,
                        stride as u32,
                        kind,
                        expert as u32,
                        count as u32,
                    ],
                    MTLSize {
                        width: rows.div_ceil(4 * nr) * 4,
                        height: count,
                        depth: 1,
                    },
                    4,
                )?;
                return Ok(out);
            }
        }
        if count == 4
            && rows >= 16
            && ((kind == 21
                && !((cols == 2560 && matches!(rows, 512 | 640))
                    || (cols == 6144 && rows == 2560)
                    || (cols == 2560 && rows >= 6144)))
                || (kind == 23 && rows >= 65536))
        {
            self.encode(
                cmd,
                &self.projections[&("quant_gemm_8", kind)],
                &[w, x, &out],
                &[
                    cols as u32,
                    rows as u32,
                    stride as u32,
                    kind,
                    expert as u32,
                    count as u32,
                ],
                MTLSize {
                    width: rows.div_ceil(16) * 8,
                    height: 1,
                    depth: 1,
                },
                8,
            )?;
            return Ok(out);
        }
        if count >= 8 && rows >= 16 {
            let wide = count >= 32;
            let wide_f32 = wide
                && matches!(
                    (kind, cols, rows),
                    (8, 5120, 2560)
                        | (8, 10240, 320)
                        | (8, 320, 10240)
                        | (8, 2560, 512)
                        | (30, 2560, 128)
                );
            self.encode(
                cmd,
                self.projections
                    .get(&(
                        if wide_f32 {
                            "quant_gemm_wide_f32"
                        } else if wide {
                            "quant_gemm_wide"
                        } else {
                            "quant_gemm"
                        },
                        kind,
                    ))
                    .ok_or_else(|| invalid("missing GEMM pipeline"))?,
                &[w, x, &out],
                &[
                    cols as u32,
                    rows as u32,
                    stride as u32,
                    kind,
                    expert as u32,
                    count as u32,
                ],
                MTLSize {
                    width: rows,
                    height: count.div_ceil(if wide { 32 } else { 16 }),
                    depth: 1,
                },
                if wide { 64 } else { 16 },
            )?;
            return Ok(out);
        }
        self.op(
            cmd,
            "quant_matmul",
            &[w, x, &out],
            &[
                cols as u32,
                rows as u32,
                stride as u32,
                kind,
                expert as u32,
                count as u32,
            ],
            rows,
            count.div_ceil(4),
        )?;
        Ok(out)
    }
    /// Large dense prefill only. Decode is charged to the same command and
    /// workspace as GEMM; no persistent expanded-weight cache is introduced.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dense_mps(
        &self,
        cmd: &Command,
        w: &Buffer,
        cols: usize,
        rows: usize,
        stride: usize,
        kind: u32,
        x: &Buffer,
        count: usize,
    ) -> Result<Buffer> {
        let decoded = self.output(rows * cols)?;
        let out = self.output(rows * count)?;
        self.op(
            cmd,
            "read_rows",
            &[w, &decoded],
            &[cols as u32, rows as u32, stride as u32, kind],
            (rows * cols).div_ceil(32),
            1,
        )?;
        let matrix = |b: &Buffer, n: usize, k: usize| -> Result<Retained<AnyObject>> {
            // SAFETY: MPSMatrixTypes.h uses NSUInteger dimensions/offset and
            // uint32_t MPSDataTypeFloat32. These are bounded F32 workspace buffers.
            unsafe {
                let descriptor: Option<Retained<AnyObject>> = msg_send![class!(MPSMatrixDescriptor),
                    matrixDescriptorWithRows: n, columns: k,
                    rowBytes: k * 4, dataType: 0x1000_0020u32];
                let descriptor =
                    descriptor.ok_or_else(|| invalid("MPS matrix descriptor allocation failed"))?;
                let allocated: objc2::rc::Allocated<AnyObject> =
                    msg_send![class!(MPSMatrix), alloc];
                let matrix: Option<Retained<AnyObject>> = msg_send![allocated,
                    initWithBuffer: &**b, offset: b.offset(), descriptor: &*descriptor];
                matrix.ok_or_else(|| invalid("MPS matrix allocation failed"))
            }
        };
        let lhs = matrix(x, count, cols)?;
        let rhs = matrix(&decoded, rows, cols)?;
        let product = matrix(&out, count, rows)?;
        // Retain Rust pool ownership through GPU completion, in addition to
        // Metal's resource references, so later allocations cannot reuse it.
        cmd.buffers.borrow_mut().extend([x.clone(), out.clone()]);
        // SAFETY: exact MPSMatrixMultiplication SDK signature. The preceding
        // decoder encoder has ended; tracked resources order decode then GEMM.
        unsafe {
            let allocated: objc2::rc::Allocated<AnyObject> =
                msg_send![class!(MPSMatrixMultiplication), alloc];
            let multiply: Option<Retained<AnyObject>> = msg_send![allocated,
                initWithDevice: &*self.device,
                transposeLeft: Bool::NO, transposeRight: Bool::YES,
                resultRows: count, resultColumns: rows, interiorColumns: cols,
                alpha: 1.0f64, beta: 0.0f64];
            let multiply = multiply.ok_or_else(|| invalid("MPS multiply allocation failed"))?;
            let _: () = msg_send![&*multiply, encodeToCommandBuffer: &*cmd.raw,
                leftMatrix: &*lhs, rightMatrix: &*rhs, resultMatrix: &*product];
        }
        self.observe_allocation();
        Ok(out)
    }

    pub(crate) fn activation(
        &self,
        cmd: &Command,
        gate: &Buffer,
        up: &Buffer,
        count: usize,
    ) -> Result<Buffer> {
        let out = self.output(count)?;
        self.encode(
            cmd,
            &self.swiglu,
            &[gate, up, &out],
            &[count as u32],
            MTLSize {
                width: if count >= 256 {
                    count
                } else {
                    count.div_ceil(32)
                },
                height: 1,
                depth: 1,
            },
            if count >= 256 { 256 } else { 1 },
        )?;
        Ok(out)
    }
    #[allow(clippy::too_many_arguments)] // Mirrors the five input bindings.
    pub(crate) fn merge(
        &self,
        cmd: &Command,
        routed: &Buffer,
        shared: &Buffer,
        weights: &Buffer,
        shared_logit: &Buffer,
        width: usize,
        top: usize,
    ) -> Result<Buffer> {
        let out = self.output(width)?;
        self.encode(
            cmd,
            &self.combine,
            &[routed, shared, weights, shared_logit, &out],
            &[width as u32, top as u32],
            MTLSize {
                width: width.div_ceil(32),
                height: 1,
                depth: 1,
            },
            1,
        )?;
        Ok(out)
    }
    pub(crate) fn complete(&self, cmd: Command) -> Result<()> {
        self.submit(&cmd);
        cmd.waitUntilCompleted();
        if cmd.status() != MTLCommandBufferStatus::Completed {
            self.failed.set(true);
            return Err(io::Error::other(format!(
                "Metal execution failed: {:?}",
                cmd.error()
            )));
        }
        cmd.finish_profile()
    }
    pub(crate) fn allocated_bytes(&self) -> usize {
        self.device.currentAllocatedSize()
    }
    fn observe_allocation(&self) {
        self.peak_bytes
            .set(self.peak_bytes.get().max(self.allocated_bytes()));
    }
    pub(crate) fn peak_allocated_bytes(&self) -> usize {
        self.observe_allocation();
        self.peak_bytes.get()
    }
    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    pub(crate) fn bytes(&self, bytes: &[u8]) -> Result<Buffer> {
        let ptr = NonNull::new(bytes.as_ptr().cast_mut().cast())
            .ok_or_else(|| invalid("empty buffer"))?;
        // SAFETY: Metal copies the entire valid slice before returning. Shared storage
        // is CPU/GPU accessible; no caller pointer is retained.
        let buffer = unsafe {
            self.device.newBufferWithBytes_length_options(
                ptr,
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| io::Error::other("Metal buffer allocation failed"))?;
        self.observe_allocation();
        Ok(Buffer::permanent(buffer))
    }

    pub(crate) fn floats(&self, values: &[f32]) -> Result<Buffer> {
        // SAFETY: f32 is plain data with no padding; the slice stays alive during the copy.
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values))
        };
        self.bytes(bytes)
    }

    pub(crate) fn output(&self, count: usize) -> Result<Buffer> {
        let bytes = count
            .checked_mul(4)
            .ok_or_else(|| invalid("output size overflow"))?;
        if bytes == 0 || bytes > self.device.maxBufferLength() {
            return Err(invalid("Metal buffer exceeds device limit"));
        }
        let mut pool = self.workspace.borrow_mut();
        if pool.limit == 0 {
            drop(pool);
            return self.empty_bytes(bytes);
        }
        let reused = pool.free.get_mut(&bytes).and_then(Vec::pop);
        let raw = if let Some(raw) = reused {
            pool.reuses += 1;
            raw
        } else {
            let charge = allocation_charge(bytes);
            // Evict only enough idle capacity for this allocation. Flushing all
            // sizes here makes long prefill shapes discard reusable decode buffers.
            while pool.managed + charge > pool.limit {
                let largest = pool
                    .free
                    .iter()
                    .filter(|(_, buffers)| !buffers.is_empty())
                    .map(|(&length, _)| length)
                    .max();
                let Some(length) = largest else {
                    break;
                };
                pool.free.get_mut(&length).unwrap().pop();
                pool.managed -= allocation_charge(length);
            }
            if pool.managed + charge > pool.limit {
                return Err(invalid(format!(
                    "workspace live allocation exceeds budget: {} + {charge} > {}",
                    pool.managed, pool.limit
                )));
            }
            let raw = self
                .device
                .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| io::Error::other("Metal workspace allocation failed"))?;
            pool.managed += charge;
            pool.peak = pool.peak.max(pool.managed);
            pool.allocations += 1;
            self.observe_allocation();
            raw
        };
        Ok(Buffer::from_allocation(Allocation {
            raw: Some(raw),
            mapping: None,
            offset: 0,
            length: bytes,
            pool: Rc::downgrade(&self.workspace),
            residency: None,
        }))
    }

    pub(crate) fn configure_workspace(&self, limit: usize) {
        self.workspace.borrow_mut().limit = limit;
    }
    /// Cumulative new allocations, reuse count, current managed bytes and peak bytes.
    pub(crate) fn workspace_statistics(&self) -> (u64, u64, usize, usize) {
        let w = self.workspace.borrow();
        (w.allocations, w.reuses, w.managed, w.peak)
    }

    pub(crate) fn command(&self) -> Result<Command> {
        if self.residency_dirty.replace(false) {
            self.residency.commit();
            self.residency.requestResidency();
        }
        if self.failed.get() {
            return Err(io::Error::other(
                "Metal engine requires reload after command failure",
            ));
        }
        let raw = self
            .queue
            .commandBuffer()
            .ok_or_else(|| io::Error::other("Metal command buffer unavailable"))?;
        Ok(Command {
            raw,
            buffers: RefCell::new(Vec::new()),
            independent_encoder: RefCell::new(None),
            failed: self.failed.clone(),
            profile: RefCell::new(
                self.profile
                    .as_ref()
                    .map(|profile| CommandProfile::new(profile.clone(), self.profile_phase.get())),
            ),
        })
    }

    // Only independent GPU dispatches may be encoded here: no consumer, submit or wait.
    // Callers finish this scope before using any produced output.
    pub(crate) fn independent_projections<T>(
        &self,
        cmd: &Command,
        body: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let encoder = if let Some(profile) = cmd.profile.borrow_mut().as_mut() {
            let descriptor = profile.descriptor(MTLDispatchType::Concurrent, None)?;
            cmd.computeCommandEncoderWithDescriptor(&descriptor)
        } else {
            cmd.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
        }
        .ok_or_else(|| io::Error::other("Metal compute encoder unavailable"))?;
        *cmd.independent_encoder.borrow_mut() = Some(encoder);
        struct EndScope<'a>(&'a Command);
        impl Drop for EndScope<'_> {
            fn drop(&mut self) {
                if let Some(encoder) = self.0.independent_encoder.borrow_mut().take() {
                    encoder.endEncoding();
                }
            }
        }
        let _end = EndScope(cmd);
        body()
    }

    fn encode(
        &self,
        cmd: &Command,
        pipeline: &Pipeline,
        buffers: &[&Buffer],
        params: &[u32],
        mut groups: MTLSize,
        row_tile: usize,
    ) -> Result<()> {
        if row_tile > 1 {
            groups.width = groups.width.div_ceil(row_tile);
        }
        cmd.buffers.borrow_mut().extend(
            buffers
                .iter()
                .filter(|b| b.0.pool.strong_count() != 0 || b.0.mapping.is_some())
                .map(|b| (*b).clone()),
        );
        let scope = cmd.independent_encoder.borrow();
        let standalone;
        let encoder = if let Some(encoder) = scope.as_ref() {
            if let Some(profile) = cmd.profile.borrow_mut().as_mut() {
                profile.dispatch_label(&pipeline.name);
            }
            encoder
        } else {
            standalone = if let Some(profile) = cmd.profile.borrow_mut().as_mut() {
                let descriptor =
                    profile.descriptor(MTLDispatchType::Serial, Some(&pipeline.name))?;
                cmd.computeCommandEncoderWithDescriptor(&descriptor)
            } else {
                cmd.computeCommandEncoder()
            }
            .ok_or_else(|| io::Error::other("Metal compute encoder unavailable"))?;
            &standalone
        };
        encoder.setComputePipelineState(pipeline);
        // SAFETY: Private callers match each kernel's binding order, element type and
        // validated dimensions. Buffers are retained by this normal command buffer;
        // params are copied by setBytes. Mapped bindings use their tensor offset.
        // Separate serial encoders
        // plus tracked resources order producers before consumers.
        unsafe {
            for (i, buffer) in buffers.iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(buffer), buffer.offset(), i);
            }
            encoder.setBytes_length_atIndex(
                NonNull::from(params).cast(),
                std::mem::size_of_val(params),
                buffers.len(),
            );
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                groups,
                MTLSize {
                    width: if row_tile == 256 {
                        256
                    } else if row_tile == 8 {
                        64
                    } else if row_tile == 32 {
                        256
                    } else if row_tile > 1 {
                        128
                    } else {
                        32
                    },
                    height: 1,
                    depth: 1,
                },
            );
        }
        if scope.is_none() {
            encoder.endEncoding();
        }
        Ok(())
    }

    fn encode_matvec(
        &self,
        cmd: &Command,
        matrix: &Matrix,
        x: &Buffer,
        out: &Buffer,
        per_expert: bool,
    ) -> Result<()> {
        let weights = self.bytes(&matrix.data)?;
        let params = [
            matrix.cols,
            matrix.rows,
            matrix.stride,
            matrix.kind as usize,
            if per_expert { matrix.cols } else { 0 },
        ]
        .map(|v| u32::try_from(v).map_err(invalid));
        let params = params.into_iter().collect::<Result<Vec<_>>>()?;
        self.encode(
            cmd,
            self.projections
                .get(&("quant_matvec", matrix.kind))
                .ok_or_else(|| invalid("unsupported projection format"))?,
            &[&weights, x, out],
            &params,
            MTLSize {
                width: matrix.rows,
                height: matrix.count,
                depth: 1,
            },
            4,
        )
    }

    pub(crate) fn finish_greedy(
        &self,
        cmd: Command,
        logits: &Buffer,
        rows: usize,
        vocabulary: usize,
        mapping: Option<&Buffer>,
    ) -> Result<Vec<u32>> {
        let chunks = vocabulary.div_ceil(1024);
        let partial = self.output(rows * chunks * 3)?;
        let result = self.output(rows * 2)?;
        for stage in 0..=1 {
            self.op(
                &cmd,
                "greedy_select",
                &[logits, mapping.unwrap_or(logits), &partial, &result],
                &[
                    vocabulary as u32,
                    rows as u32,
                    chunks as u32,
                    stage,
                    mapping.is_some() as u32,
                ],
                if stage == 0 { chunks } else { 1 },
                rows,
            )?;
        }
        self.complete(cmd)?;
        self.read_u32(&result, rows * 2)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| {
                if v[1] != 0 {
                    Err(invalid("non-finite logits"))
                } else {
                    Ok(v[0])
                }
            })
            .collect()
    }

    pub(crate) fn finish(&self, cmd: Command, output: &Buffer) -> Result<Vec<f32>> {
        self.submit(&cmd);
        cmd.waitUntilCompleted();
        if cmd.status() != MTLCommandBufferStatus::Completed {
            self.failed.set(true);
            return Err(io::Error::other(format!(
                "Metal execution failed: {:?}",
                cmd.error()
            )));
        }
        let values = self.read_f32(output)?;
        cmd.finish_profile()?;
        Ok(values)
    }
    // Call only after completing every command that writes this buffer.
    pub(crate) fn read_f32(&self, output: &Buffer) -> Result<Vec<f32>> {
        // SAFETY: The shared output has f32 layout, was fully initialized by the kernel,
        // and the GPU is finished. Copying prevents aliasing with later GPU commands.
        let values = unsafe {
            std::slice::from_raw_parts(
                output.contents().as_ptr().cast::<f32>(),
                output.length() / 4,
            )
        }
        .to_vec();
        if values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("Metal output contains non-finite values"));
        }
        Ok(values)
    }

    pub fn dequant(&self, matrix: &Matrix) -> Result<Vec<f32>> {
        self.validate_matrix(matrix)?;
        let w = self.bytes(&matrix.data)?;
        let count = matrix.cols * matrix.rows * matrix.count;
        let out = self.output(count)?;
        let cmd = self.command()?;
        self.op(
            &cmd,
            "read_rows",
            &[&w, &out],
            &[
                matrix.cols as u32,
                (matrix.rows * matrix.count) as u32,
                matrix.stride as u32,
                matrix.kind,
            ],
            count.div_ceil(32),
            1,
        )?;
        self.finish(cmd, &out)
    }
    fn validate_matrix(&self, matrix: &Matrix) -> Result<()> {
        let (block, bytes) = crate::gguf::geometry(matrix.kind)?;
        if matrix.cols == 0
            || matrix.rows == 0
            || matrix.count == 0
            || !matrix.cols.is_multiple_of(block)
            || matrix.stride != matrix.cols / block * bytes
            || matrix
                .stride
                .checked_mul(matrix.rows)
                .and_then(|n| n.checked_mul(matrix.count))
                != Some(matrix.data.len())
            || [matrix.cols, matrix.rows, matrix.count, matrix.stride]
                .iter()
                .any(|&n| u32::try_from(n).is_err())
        {
            return Err(invalid("invalid quantized matrix layout"));
        }
        Ok(())
    }
    pub fn project(&self, matrix: &Matrix, input: &[f32]) -> Result<Vec<f32>> {
        self.validate_matrix(matrix)?;
        if input.iter().any(|v| !v.is_finite()) || matrix.count != 1 || matrix.cols != input.len() {
            return Err(invalid("projection shape mismatch"));
        }
        let x = self.floats(input)?;
        let out = self.output(matrix.rows)?;
        let cmd = self.command()?;
        self.encode_matvec(&cmd, matrix, &x, &out, false)?;
        self.finish(cmd, &out)
    }

    fn ffn(&self, cmd: &Command, matrices: &[Matrix; 3], x: &Buffer) -> Result<Buffer> {
        let [gate, up, down] = matrices;
        if (gate.cols, gate.rows, gate.count) != (up.cols, up.rows, up.count)
            || (down.cols, down.rows, down.count) != (gate.rows, gate.cols, gate.count)
        {
            return Err(invalid("FFN matrix shapes do not match"));
        }
        let n = gate.rows * gate.count;
        let g = self.output(n)?;
        let u = self.output(n)?;
        let activation = self.output(n)?;
        let out = self.output(down.rows * down.count)?;
        self.encode_matvec(cmd, gate, x, &g, false)?;
        self.encode_matvec(cmd, up, x, &u, false)?;
        self.encode(
            cmd,
            &self.swiglu,
            &[&g, &u, &activation],
            &[u32::try_from(n).map_err(invalid)?],
            MTLSize {
                width: n.div_ceil(32),
                height: 1,
                depth: 1,
            },
            1,
        )?;
        self.encode_matvec(cmd, down, &activation, &out, true)?;
        Ok(out)
    }

    pub fn moe(
        &self,
        routed: &[Matrix; 3],
        shared: &[Matrix; 3],
        shared_gate: &Matrix,
        input: &[f32],
        weights: &[f32],
    ) -> Result<Vec<f32>> {
        for matrix in routed
            .iter()
            .chain(shared.iter())
            .chain(std::iter::once(shared_gate))
        {
            self.validate_matrix(matrix)?;
        }
        let width = input.len();
        if input.iter().chain(weights).any(|v| !v.is_finite())
            || routed[0].cols != width
            || shared[0].cols != width
            || shared[0].count != 1
            || routed[0].count != weights.len()
            || (shared_gate.cols, shared_gate.rows, shared_gate.count) != (width, 1, 1)
        {
            return Err(invalid("MoE input shape mismatch"));
        }
        let cmd = self.command()?;
        let x = self.floats(input)?;
        let routed_out = self.ffn(&cmd, routed, &x)?;
        let shared_out = self.ffn(&cmd, shared, &x)?;
        let shared_logit = self.output(1)?;
        self.encode_matvec(&cmd, shared_gate, &x, &shared_logit, false)?;
        let weights = self.floats(weights)?;
        let out = self.output(width)?;
        self.encode(
            &cmd,
            &self.combine,
            &[&routed_out, &shared_out, &weights, &shared_logit, &out],
            &[
                u32::try_from(width).map_err(invalid)?,
                u32::try_from(routed[0].count).map_err(invalid)?,
            ],
            MTLSize {
                width: width.div_ceil(32),
                height: 1,
                depth: 1,
            },
            1,
        )?;
        self.finish(cmd, &out)
    }
}

#[cfg(test)]
mod tests;
