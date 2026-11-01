# Rust 引擎路线（更新于 2026-10-13）

> **历史记录（2026-10-13 快照）**：文中“MTP 尚未实现”等描述已过时。当前能力、命令与限制见 [README.md](README.md)。

用户已授权按 [ARCHITECTURE.md](ARCHITECTURE.md) 实现自有 Rust / Metal 推理引擎。
保持一个 crate、直接模型前向、单模型和单活跃会话。

采用 objc2 系统绑定、GGUF 和固定版本 Hugging Face Tokenizers；改编有许可的 ggml
量化布局与码表。Rust 拥有模型执行、状态、权重访问、缓存和提交时机。
llama.cpp 与 MLX 只作为独立参考，不进入生产执行路径。

选择依据和拒绝的替代方案保留在架构文档第 2 节。没有引入通用计算图、异步运行时、
后端插件、HTTP 服务或自有权重容器。Qwen 和 GLM 只共享已经重复的量化权重与专家槽位，
模型过程和状态分别定义。

当前完整 Qwen 主干已经接通文本生成、分组预填充和有限内存专家调度。
[实际验收记录](NATIVE_VALIDATION.md) 区分实现能力、已测行为和剩余风险。
原始一层 FFN 结果仍保留在 [validation/rust-moe-layer-0.json](validation/rust-moe-layer-0.json)，
但不再作为整模型验收的替代品。

GLM 5.3 Flash 使用已校验的 ggml-org Q2_K_S 两分片。
独立 KDA、mHC、MLA/DSA 和路由已实现；完整短请求的 161 个位置与参考首选输出一致，
140 个生成 token 和 EOS 一致，11/16/24 GiB 预算及取消/状态复用检查通过。
结果见 [validation/native-glm.json](validation/native-glm.json)。GLM 长上下文质量尚待验证。
MTP 尚未实现；已列入 [性能补全 Spec](OPTIMIZATION_SPEC.md) 的必做能力，默认是否启用按实测收益决定。
模型权重遵循自己的许可；仓库不包含或再分发权重。

2026-10-13 第三轮增加四行 SIMD 投影、Q2_0/Q2_K/IQ1_M 的块信息复用，
以及对齐的 F16/BF16/F32 读取；计算继续使用 F32 和 Metal Safe math。
GLM 使用最多 8-token 分组预填充：KDA 独立投影和 dense/shared FFN 批量计算，
MoE 按专家聚合 token；卷积、KDA 递推与因果 DSA 仍按位置有序执行。
被遮蔽的注意力位置跳过点积，GLM 池化索引每个池计算一次名次。

跨层延迟提交与关闭系统文件缓存的实验没有进入当前实现。
命令行默认预算为 32 GiB；64 GiB 探针在本机当时负载下触发严重压缩，已停止并保留证据。
正式交替性能、逐位置 logits 回归及独立参考检查统一记录在
[validation/native-performance-3.json](validation/native-performance-3.json)。

本轮新增中英文/代码独立参考对照 135 个位置，top-1 全部一致；Qwen/GLM 各 2112-token
状态回归的 11 个检查点全词表 logits 与原版逐位一致。这不替代独立长上下文理解质量验收。
