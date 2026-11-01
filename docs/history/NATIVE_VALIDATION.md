# Native Rust / Metal 验收记录

> **历史记录（2026-10-12/13 验收）**：数据和结论对应当时的提交，此后的内核、MTP 和预填充改动未在本文重新验收。当前状态见 [README.md](../../README.md)。

日期：2026-10-12。目标为记录的两分片 Qwen3.8-Flash-Next GSQ-RCO IQ2_XS 和
GLM 5.3 Flash Q2_K_S，Apple M2 Max / 96 GiB、macOS 26.5.2。
原始大文件留在被忽略的 `evidence/`；可审阅摘要与实际输出见
[validation/native-qwen.json](../../validation/native-qwen.json)、
[validation/native-glm.json](../../validation/native-glm.json)。
权重摘要沿用 [macos/model-sha256.txt](../../macos/model-sha256.txt)。

## 当前完成边界

Rust 自有路径已执行 48 层完整前向和文本生成，包括 HC、Gated DeltaNet、QSA、
PLE、路由和共享专家、输出头。权重/算子/状态/缓存/提交均由 Rust 与 Metal 管理。
程序依赖检查只发现系统框架与库，没有 libllama/libggml，也不启动参考程序。

已实现 8-token 块预填充，稠密投影用量化矩阵乘法；路由后按专家分组多个 token。
GDN、卷积、PLE 历史与因果注意力依次更新位置。Decode 常驻路由留在 GPU；
流式路径只读取 top-k ID，在 GPU 使用的槽位之外补入下一专家。
GPU 完成前不会覆盖其权重槽位；缓存专家之间也检查取消。短读或取消不把部分专家标成可用。

首版只支持 greedy、单条用户消息、单活跃会话。Qwen 模板关闭思考；GLM 模板使用默认 Max 推理强度。
GLM 5.3 Flash 的独立主干已完成完整短请求数值/生成、状态及有限内存验收，详见末节。
MTP、随机采样、多模态和任意聊天模板没有实现。
下面的检查证明特定条件下的行为，不等于所有语料、模型变体或硬件的质量验收。

## 格式、分词与算子

- 12 种实际权重格式分别抽取真实行，与独立 ggml C 解量化或 NumPy 对照。
  最大误差记录在 JSON；每种均通过 `rtol=2e-6, atol=1e-8`。
- 28 组命令行分词用例覆盖中文、英文、代码、组合字符、表情、阿拉伯文、空白、
  长重复串与特殊 token。额外 Rust 用例覆盖 NUL/控制字节、解码和 chat 模板序列。
  BPE 固定 Hugging Face Tokenizers 0.22.1；不是自行重写 BPE。
- GPU 算子与独立双精度公式对照：GDN 跨调用递推/多头映射、逐通道膨胀卷积、
  GQA 注意力及 mask、QSA 2052/2055 位置预算和同分选择。
- 量化矩阵乘法检查 1/2/4/5/8-token 尾块与专家偏移。
- GGUF 检查截断文件头，以及已通过目录检查的文件随后截断时，位置读取返回
  `UnexpectedEof`；错误没有被转换成成功输出。

## 整模型数值与生成

固定 reference 为 memoriaru/llama.cpp `555d1ec9daa4564cb06adb39c469acfdcd1d1e93`。
权重保持原量化，Rust 激活、循环状态、KV 为 F32。Oracle 也使用 F32 K/V，并关闭
Flash Attention：该参考的 Flash 路径会临时把 F32 K/V 转为 F16，不能当作同精度比较。

4 个短探针位置 `[1, 109266, 248044, 96748]` 与 reference 的 top token 全部相同，
初始最大绝对 logit 误差约 0.00364。常驻与 8 GiB 缓存路径的全部 logits 逐位一致。

真实中文请求的 18-token prompt 加 29-token 输出，共 47 个 teacher-forced 位置：
纯 F32 reference 与 Rust 的 top token 全部相同；最终 29 个输出 token 和 EOS 一致。
最大绝对 logit 差 0.9685，相对 L2 为 0.0200；reference→Rust 平均 KL 为
`6.096e-5`，最大 KL 为 `6.968e-4`，平均总变差为 0.00248、最大为 0.01629。
这些概率指标补充 top-token 一致性，不能宣称所有 logits 位级相同。

原始 Flash reference 在位置 44 出现一次 greedy 分歧。逐层追踪发现早期第 9 层
路由的两个边界专家仅差约 `1e-5`，微小计算差异改变第十个专家并放大后续输出。
尝试改用 llama 的 L2 epsilon 公式没有消除该分歧，已撤回实验修改。
生产 L2 使用 MLX 模型的 `rsqrt(sum + epsilon)`。
关闭 reference Flash 的纯 F32 对照消除了这组生成的 greedy 分歧。
这解释当前案例；仍需更广语料与长上下文的独立质量对照。

实际输出：

> 我是通义千问，一个由阿里巴巴通义实验室独立开发的大语言模型，致力于为您提供智能、高效且富有同理心的帮助。

8 GiB 分组预填充生成与之前常驻/逐 token 生成的 token ID 相同。
真实权重 API 测试又比较了 1/2/4/8-token 块、PLE 重置标记和跨块状态，logits 逐位一致。
生成 EOS 248046 与 PLE 重置 248044 各自按自己的角色处理。

## 内存与状态行为

8 GiB 配置、上下文容量 8192、上述 18+29-token 请求：

| 观测项 | 实测字节 |
| --- | ---: |
| 配置预算 | 8,589,934,592 |
| 权重/状态/工作区/主机预留 | 8,556,992,512 |
| 请求结束的 Metal 分配 | 8,225,619,968 |
| 最大 RSS | 8,434,057,216 |
| 峰值 physical footprint | 8,415,631,840（7.838 GiB） |

峰值由直接运行二进制的 `/usr/bin/time -l` 测量，低于配置预算。
不能用包装它的 Cargo 进程 footprint 代替模型进程，也不把 RSS 与 Metal 分配相加。
最初未预留主机元数据开销的版本超出预算约 128 MiB；现已增加 256 MiB 主机预留并重测。
工作区随上下文的 score/probability 大小增长，短上下文至少预留 64 MiB。
系统文件缓存与其他应用占用不由这个预算控制。

已通过单会话租约、上下文越界先于状态变更、非法 token 后会话失效、取消后拒绝旧会话、
丢弃后新会话/缓存复用、输出回调错误传播、长度终止和已取消请求无输出。
实际线程在 GPU 工作期间设置取消，17–19 ms 左右返回；这是这次观测，不是延迟 SLA。
设备命令失败会要求重新装载 Engine；未人为触发驱动故障。

完整 **8192-token、8 GiB 配置 smoke 已通过**：16 个 512-token 检查点均产生有限
logits，最后已消费长度为 8192；随后追加输入被拒绝，历史长度不变。
前向累计 5010.744 s（83.51 分钟），测试总计 5012.54 s；保留全部慢样本。
这是重复中文段落的状态执行检查，不是长文本语言质量对照。当前吞吐仍需优化。

接近结束时读取 Darwin `proc_pid_rusage` V4，累计峰值为 8,454,003,192 字节
（7.873 GiB），进程累计磁盘读取计数为 41,957,273,600 字节。该采样先于进程退出，
不是最终精确峰值；读取计数也包含装载和其他文件，并非隔离的冷 SSD 测量。
[XNU 的计数来源](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/bsd_kern.c)
为进程 disk I/O statistics。系统 wired、压缩与 swap 快照另保存在 JSON 中，
它们属于全系统，不能归因或相加为引擎内存。

## 计时与读取

8 GiB 中文请求：装载 1.221 s，预填充 24.119 s，首 token 24.120 s，
完整请求 68.009 s，29 个 decode 步的 p50 为 1.101 s、p95 为 3.798 s。
读入 9,749,494,880 文件字节，专家读取耗时 9.297 s；缓存命中 11,979、未命中 6,751。
Prefill 计数单位为专家分组，decode 为一次选中专家，不能直接当成所有 token×专家的命中率。
未取得可信的 SSD 物理读取字节；文件缓存未清空，不能标成冷 SSD 成绩。

常驻专家、同一 Engine、同一个 18-token prompt，交替顺序 B/S/S/B/B/S：

| 样本 | 批量预填充 B（s） | 逐 token S（s） |
| --- | ---: | ---: |
| 0 / 1 | 23.288 | 21.005 |
| 3 / 2 | 12.669 | 45.225 |
| 4 / 5 | 14.971 | 49.682 |

六次最后位置 logits 全部逐位一致。首次 GPU 使用及系统负载未控制，全部慢样本保留。
分组路径在后两组更快，但样本波动较大；没有据此宣称稳定加速比或超过其他引擎。
也没有把旧 C++/Strata 的速度与 MTP 成绩算成 Rust 成绩。

## 复现

```sh
cargo build --release --locked
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
cargo fmt --check
MODEL=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
QWEN_MODEL="$MODEL" cargo test --release --locked --test qwen \
  state_cancellation_memory_and_reuse -- --ignored --nocapture --test-threads=1
QWEN_MODEL="$MODEL" cargo test --release --locked --test qwen \
  full_8192_context -- --ignored --nocapture --test-threads=1
QWEN_MODEL="$MODEL" cargo test --release --locked --test qwen \
  prefill_comparison -- --ignored --nocapture --test-threads=1
python3 tools/check_native.py --model "$MODEL" \
  --ggml build-macos/bin/libggml-base.dylib --report evidence/native-formats-tokenizer.json
```

真实模型检查逐个运行，避免同时装载。Cargo `--offline` 可在依赖已缓存时添加。
long-context 测试使用重复的已分词中文段落，总计 8192 个 token。

独立 oracle 在固定 CMake reference 构建完成后编译：

```sh
c++ -std=c++17 -O2 tools/qwen-oracle.cpp \
  -I build-macos/_deps/moe_llama-src/include \
  -I build-macos/_deps/moe_llama-src/ggml/include \
  -L build-macos/bin -Wl,-rpath,"$PWD/build-macos/bin" \
  -lllama -lggml -lggml-base -o evidence/qwen-oracle
QWEN_ORACLE_NO_FLASH=1 evidence/qwen-oracle "$MODEL" \
  evidence/chat.tokens evidence/reference.logits.f32 8192
target/release/ingot dev logits "$MODEL" evidence/chat.tokens \
  evidence/native.logits.f32 8192 64
```

`chat.tokens` 为 JSON 中的 prompt token 与输出 token 拼接后按 little-endian u32 编码。
两份输出按 `[position, vocabulary]` F32 排列，可以计算最大误差、softmax KL、总变差和 argmax。
Trace 使用 `QWEN_ORACLE_TRACE_DIR`、`QWEN_ORACLE_TRACE_POSITION` 或 Rust 的 trace 命令；
它属于诊断，不作为正常请求计时。

## 尚需完成

扩大 teacher-forced 语料、独立长上下文质量对照、可控缓存条件下的端到端性能基线。
MTP 需要完整状态恢复和目标验证，尚未实现。

## GLM 5.3 Flash 验收 — 2026-10-12

按用户授权删除 256 个确认的编译目录，可用空间由 41.43 GB 增加至 3.890 TB，净增加
3.849 TB。源码、Git 历史、已有模型及聊天记录未删除。此数值在 GLM 下载之前测量。
随后选定 [ggml-org Q2_K_S](https://huggingface.co/ggml-org/GLM-5.3-Flash-GGUF)，
固定修订为 `c259a9f17fe28ec50fecb0aa9f56971a0ae6eeaa`，两分片共 110,027,352,992 字节。
两个分片已完整下载，SHA-256 均与固定修订的发布记录一致：

| 分片 | 字节数 | SHA-256 |
| --- | ---: | --- |
| 00001 | 9,430,240 | `f334b5b84890a5c2ff7081c99ebe31aa7815250936bc5d96d19e31eab04da831` |
| 00002 | 110,017,922,752 | `61ed849acb67dc7832eda82e9fad0f0257edcd20ced1fc7640e5e329ed6951b4` |

原始日志、F32 logits 和测量记录保存在 `evidence/glm-20261012/`；可提交的结果摘要为
[validation/native-glm.json](../../validation/native-glm.json)。

已实现独立 45 层主干：34 层 KDA、11 层压缩 NoPE MLA/池化 DSA、4 路 mHC、
前 3 层 dense FFN、其余 288 选 8 MoE 与共享专家。GLM 当前按 token 顺序更新状态，
块末计算输出头；尚不宣称 GLM 已有 Qwen 的分组预填充。共享现有量化投影、权重读取
和逐层 LRU 专家槽位，不将 GLM 路由或状态套进 Qwen。

已完成的检查：

- 完整本地权重的 1383 个张量绑定通过。实际使用的 F32、BF16、Q8_0、Q2_K 对照独立
  ggml C 或 NumPy，7 次真实行检查最大绝对差均为 0：F32 张量仅有一行，其余抽取第 0/7 行。
- 30 组分词用例与固定 llama.cpp 的 GLM 分词对照一致。
- 官方 `tokenizer.json` 和 GGUF Jinja 模板产生的 5 组聊天序列，与 Rust 完全一致；
  另检查 NUL/控制字节解码及元数据定义的三个结束标记。
- 5 项 GLM GPU 算子检查通过独立双精度公式，包括跨步 KDA、mHC Sinkhorn/残差方向、
  MLA 分头吸收/值投影、池化和 2048 预算边界、路由偏置选择与限幅激活。
- 13 项普通测试、Clippy（所有目标、警告视为错误）、格式检查、发布构建和 Python 语法检查通过。
- 2 项 GLM 真实权重测试通过：顺序/分块输入、单会话限制、上下文边界、无效 token、
  取消后状态失效、新会话复用，以及 11/16/24 GiB 预算一致性。
- 共享槽位/Engine 接入之后，Qwen 的真实权重状态、取消、复用与 8 GiB 回归检查通过。

GLM 的参考单独固定为 ggml-org/llama.cpp `08ffba21e5c2f1728b6b405c05ff3913f65706d4`，
验证工具为 `tools/glm-oracle.cpp`。首次未修改的 mmap 装载向 Metal 注册了
104921.18 MiB，第一次 decode 因 GPU 内存不足退出（返回 3），未产生有效数值结果。
随后应用 [tools/glm-oracle-loading.patch](../../tools/glm-oracle-loading.patch)：仅给 gate/up/down
专家张量增加三个 `TENSOR_READ_LAZY` 标记，不改模型公式或算子。参考使用 load mode NONE、
lazy ON、4 GiB 专家 GPU 缓存、F32 状态及 KV、关闭 Flash Attention。
失败与成功记录分别保存；参考装载修复不会进入 Rust 运行依赖，也不替换原 Qwen 参考。

### 整模型数值与生成

先用 8 个输入位置检查全部 154880 维 logits，再对完整请求使用同一 token 序列逐位置对照：

| 检查 | 位置数 | argmax 一致 | 最大绝对差 | 相对 L2 |
| --- | ---: | ---: | ---: | ---: |
| 短探针 | 8 | 8/8 | 3.386e-5 | 1.621e-6 |
| 完整请求 teacher-forced | 161 | 161/161 | 3.648e-5 | 5.874e-7 |

两组 logits 均全部有限。完整请求平均 KL 为 9.594e-13，最大 KL 为 2.491e-11，
平均总变差为 3.161e-7。CLI 输入为「只回答：1+1等于几？」，使用 GLM 默认 Max 思考模板，
21 个 prompt token 后生成 140 个 token，输出包含思考文本，最终答案为 `2`。
参考预测与原生输出的全部 140 个 token 一致，最后均预测结束标记 `154827`，
原生请求以 `Eos` 停止，未触及 224-token 上限。完整文本和 token 序列保存在结果 JSON。

### 预算、状态与取消

上下文容量 64 时，11/16/24 GiB 预算分别提供每个 MoE 层 3/18/43 个专家槽位。
11 GiB 的 3 个槽位少于 top-8，实际触发单步内的换入换出；同一输入的最终 logits
在三种预算下逐字节一致。11/16 GiB 两个独立探针的全部 8 个位置也逐字节一致。
8 GiB 装载被明确拒绝。取消测试在请求开始 5 ms 后发出取消，9.245 ms 时返回；
旧会话随后拒绝执行，丢弃后新会话输出与初始状态一致。尚未提交工作的预取消请求也通过。

实际模型子进程的物理 footprint 峰值由 macOS `time -l` 记录，另采样进程磁盘计数。
11 GiB 探针峰值为 11,487,027,816 字节（10.70 GiB）；16 GiB 探针峰值为
16,694,767,832 字节（15.55 GiB）。完整生成峰值为 16,706,269,400 字节（15.56 GiB）。
包含全部预算测试的 API 测试进程累计峰值为 25,364,875,288 字节，低于其最大 24 GiB
预算；此值不是一个隔离的 24 GiB 请求测量。RSS、Metal 分配量和物理 footprint
是不同计数，不能相加。

### 本机时间与读取量

完整生成在 M2 Max / 96 GiB、256 上下文容量、16 GiB 引擎预算下测量：

| 项目 | 实测 |
| --- | ---: |
| 模型装载 | 1.847 s |
| 预填充 | 17.370 s |
| 请求开始至首 token（装载后开始计时） | 17.371 s |
| 请求总时间 | 129.764 s |
| CLI 墙钟时间（含装载，`time -l`） | 131.81 s |
| 140 次 decode 的 p50 / p95 | 0.797 / 0.865 s |
| 预算账本预留 / 结束时 Metal 分配 | 16,941,673,720 / 16,540,057,600 B |
| 专家逻辑读取量（不含装载） | 272,729,899,008 B |
| 专家缓存命中 / 未命中 | 21,068 / 33,028 |
| 专家读取调用累计耗时 | 52.510 s |
| 最后一次进程采样的累计磁盘读取计数 | 71,940,362,240 B |

读取调用部分与 GPU 计算重叠，累计耗时不等于纯等待时间。进程磁盘计数包括模型装载和
其他进程文件；逻辑读取可能命中系统文件缓存。未清空系统缓存，不能由这些数字推导
冷 SSD 吞吐。原生完整 teacher-forced 运行 133.228 s，参考运行 189.651 s；二者缓存预算
不同，运行顺序与系统缓存也未控制，不作公平速度对比或稳定加速结论。

### GLM 复现

```sh
GLM_MODEL=/path/to/GLM-5.3-Flash-Q2_K_S-00001-of-00002.gguf
cargo build --release --locked
GLM_MODEL="$GLM_MODEL" cargo test --release --locked --test glm \
  -- --ignored --nocapture --test-threads=1
target/release/ingot generate "$GLM_MODEL" '只回答：1+1等于几？' 224 256 16
```

参考源单独检出上述固定修订，再从本仓库目录构建：

```sh
GLM_REF=/path/to/ggml-org-llama.cpp
GLM_BUILD=/path/to/glm-reference-build
git -C "$GLM_REF" apply "$PWD/tools/glm-oracle-loading.patch"
cmake -S "$GLM_REF" -B "$GLM_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_ACCELERATE=ON \
  -DGGML_BLAS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF
cmake --build "$GLM_BUILD" --target llama -j8
mkdir -p evidence
c++ -std=c++17 -O2 tools/glm-oracle.cpp \
  -I "$GLM_REF/include" -I "$GLM_REF/ggml/include" \
  -L "$GLM_BUILD/bin" -Wl,-rpath,"$GLM_BUILD/bin" \
  -lllama -lggml -lggml-base -o evidence/glm-oracle
python3 tools/check_native.py --model "$GLM_MODEL" \
  --ggml "$GLM_BUILD/bin/libggml-base.dylib" --tokenizer-oracle evidence/glm-oracle \
  --report evidence/glm-formats-tokenizer.json
evidence/glm-oracle "$GLM_MODEL" evidence/glm-20261012/chat.tokens \
  evidence/glm-reference-repeat.f32 256
target/release/ingot dev logits "$GLM_MODEL" evidence/glm-20261012/chat.tokens \
  evidence/glm-native-repeat.f32 256 16
```

输入文件可从结果 JSON 的 `full_request.prompt_ids` 与 `output_ids` 拼接生成，按
little-endian u32 编码；输出按 `[161, 154880]` little-endian F32 比较。原生输出文件
必须不存在，避免覆盖已有证据。使用独立目录，逐个运行大模型检查。

本轮整模型证据覆盖短序列至 161 个位置。GLM 的长上下文质量、分组预填充优化、可控
冷 SSD 性能、MTP 与随机采样仍待后续工作；算子的池化/2048 边界检查不能替代整模型
长上下文质量验收。


## 第一轮性能优化 — 2026-10-12

本轮只修改量化投影的 pipeline 选择和 Qwen/GLM 的专家选择算子，未增加依赖或配置。
量化投影用 Metal function constant 固定权重类型；各 lane 的 F32 累加、SIMD 求和顺序
与解量化公式保持不变。专家选择由一个线程扫描改为一个 32-lane SIMD group 并行扫描，
同分仍选较小 expert ID。归一化按原来的顺序在 lane 0 累加；GLM 偏置仍只用于选择，
混合权重使用未加偏置的 sigmoid 分数与原有 2.5 缩放。

保持原量化、F32 激活/状态、Safe 数学模式、内存预算、缓存替换和读取重叠策略，MTP 关闭。
本轮采用现有 Metal API 的函数常量与 SIMD 归约，不用降低精度或扩大缓存获得速度改善。
固定参考的 Metal `mul_mv` 同样使用函数常量，但其实现或速度不能算作原生 Rust 成绩。

### 定位证据

私有诊断副本让每个 dispatch 单独提交并完成，再记录 GPUStartTime/GPUEndTime；
它会破坏正常提交与重叠，只用于定位算子，不能当作请求速度。
Qwen 输入 `[1, 109266]`、256 上下文容量、8 GiB 预算，共 8436 个 dispatch：

| 诊断项目 | 原版 GPU 时间 | 仅并行选择 | 加入类型专门化 |
| --- | ---: | ---: | ---: |
| 专家选择 | 0.20050 s | 0.00814 s | 0.00765 s |
| 矩阵向量投影 | 0.36105 s | 0.35826 s | 0.15992 s |
| 专家投影 | 0.29528 s | 0.28609 s | 0.13021 s |
| 全部 dispatch 时间之和 | 0.89564 s | 0.69837 s | 0.32930 s |

这三个诊断结果的全部 logits 逐字节一致。原版中专家选择约占此次诊断 GPU 时间的
22.4%，两个投影类别约占 73.3%；这些比例不能解释为正常请求的端到端耗时比例。
结果与完整请求记录见 [validation/native-performance.json](../../validation/native-performance.json)。

### 固定条件的完整请求

M2 Max / 96 GiB，两个版本都使用 256 上下文容量。Qwen 预算 8 GiB，GLM 预算 16 GiB。
每个模型、每个版本先保留一个 1-token 预热请求；随后按 A/B、B/A、A/B 顺序完成三组
完整请求，每次新建 Engine，不同时运行本任务的 GPU 检查。A 为任务开始前保存的原版
二进制，B 为本轮发布构建；各自 SHA-256 和全部命令保存在 JSON。

Qwen 输入「用一句中文介绍你自己。」、上限 32 token，每次均输出 29 token 后 Eos。
GLM 输入「只回答：1+1等于几？」、上限 224 token，每次均输出 140 token 后 Eos，最终
答案为 `2`，包括默认 Max 思考文本。各模型全部六次的输出 ID、文本和停止原因一致。

| 项目 | Qwen A → B | GLM A → B |
| --- | ---: | ---: |
| 生成吞吐（全部 decode 步数 / 全部 decode 耗时） | 1.847 → 3.178 token/s | 0.486 → 0.958 token/s |
| 请求平均总时间（不含装载） | 22.056 → 13.294 s | 330.391 → 184.411 s |
| 请求平均时间变化 | −39.7% | −44.2% |
| 平均预填充 | 6.331 → 4.145 s | 42.077 → 38.134 s |
| 平均首 token（装载后计时） | 6.332 → 4.145 s | 42.077 → 38.134 s |
| 单次生成吞吐范围 | 1.787–1.896 → 2.835–3.588 token/s | 0.449–0.530 → 0.875–1.113 token/s |
| 三次中的最大物理 footprint | 8,405,244,360 → 8,403,769,800 B | 16,710,136,072 → 16,705,941,792 B |

两个模型的新版本峰值都低于本轮配置预算，预留与结束时 Metal 分配量也与原版一致。
物理峰值取实际 CLI 子进程的 `time -l` 退出统计，不用包装进程或 RSS 代替。

保留全部正式样本，包括较慢的新版 Qwen 和旧版 GLM。没有清空系统文件缓存，系统负载
未控制，三组输入对照不能证明长期稳定速度或所有语料的改善。GLM 本轮每次进程累计
磁盘读取约 118–123 GB，比上一节旧测量更多；这是采样进程计数，含装载及其他文件，
不能单独证明冷 SSD 性能或把全部波动归因于读取。此前 0.66 / 1.25 token/s 的单次记录
属于不同运行条件，不能用于计算本轮加速倍数。首次着色器编译的冷启动成本未单独测量。

普通测试 14 项及 Clippy（所有目标、警告为错误）通过。新增 13 种真实权重格式的矩阵
向量投影检查：抽取共 25 行，对照解量化后以双精度计算的点积；全部通过。Qwen 路由
覆盖 1/31/33/288/512 个专家、跨 lane 同分、尾部与大 logits；GLM 路由覆盖偏置选择、
无偏权重、缩放和 1/33/288 个专家。8-token 分组、expert 偏移与 MLA 分头投影检查保留。

真实格式检查的复现入口（两个主分片都必须提供）：

```sh
QWEN_MODEL=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
GLM_MODEL=/path/to/GLM-5.3-Flash-Q2_K_S-00001-of-00002.gguf \
  cargo test --release --locked --test projections -- --ignored --nocapture --test-threads=1
```

### 优化后的回归与参考对照

发布构建重新完成 Qwen 的会话状态、分块、取消、复用、预算与错误合同检查，以及 GLM
的状态/取消/复用/11、16、24 GiB 预算和 tokenizer 检查。Qwen 全部 47 个位置、GLM
全部 161 个位置的完整 F32 logits，与优化前逐字节一致。参考实现的所有位置 top-1
分别为 47/47、161/161；29/140 个输出 ID 及最终 Eos 也一致。Qwen 与参考的最大绝对
差仍为 0.968502、相对 L2 为 0.020011；GLM 为 3.648e-5、5.874e-7。top-1 一致不能
替代更广语料的质量验收。此前 Qwen 8192 容量检查未在本轮重跑，GLM 长上下文质量仍待验证。

另做单次标量 teacher-forced 参考对照：双方每次输入 1 token，每个位置计算完整输出头，
F32 KV、Flash Attention 关闭，容量 256，MTP 关闭；此工作量不同于上面的分组生成请求。
固定参考修订沿用前述记录，只修改验证包装程序的专家缓存参数：Qwen 每 bank 62 个
slot，144 个专家权重池均成功建立、无跳过，专家权重容量 4,293,376,000 B；GLM 全局
专家缓存设为 6,242,697,216 B，对齐原生每层 18 slot 的合计容量。GLM 仍使用前述仅修改
三个专家装载标记的补丁，数学和算子未改。

| 完整 teacher-forced 输入 | 原生优化版 CLI 耗时 | llama.cpp 参考 CLI 耗时 | 原生 / 参考物理峰值 |
| --- | ---: | ---: | ---: |
| Qwen，47 个位置 | 25.26 s | 7.81 s | 7.816 / 4.299 GiB |
| GLM，161 个位置 | 125.65 s | 230.89 s | 15.556 / 16.479 GiB |

耗时含装载、逐位置前向及 F32 文件输出，取实际子进程 `time -l`；每种实现只有一个样本。
专家权重容量对齐，整体硬内存预算未对齐：mmap/装载和缓存分配、替换策略不同，GLM
参考物理峰值超过原生 16 GiB 预算。系统文件缓存未清空，跨框架顺序未交替；Qwen 原生
进程采样累计读取约 12.45 GB，而参考约 45 MB，包含装载及其他文件。这些差异无法从
单次耗时里扣除，不能把表格用于公平排名或证明整体超过 llama.cpp。

完整命令、两个版本的二进制 SHA-256、冻结源码 SHA-256、全部正式与预热样本、数值对照
和完成的检查写入 [validation/native-performance.json](../../validation/native-performance.json)。
本机原始日志、tokens、F32 输出、验证包装程序和文件清单保存在 Git 忽略的
`evidence/native-performance-20261012/`，无模型权重或发布二进制进入提交。

## 第二轮性能优化 — 2026-10-12

本轮修改 `src/weights.rs`、`src/qwen.rs`、`src/glm.rs`：连续命中的专家共享一次 GPU
提交，单 token 的 gate/up、激活、down 按多专家数组执行，仍按原 rank 写回。Qwen
分组预填充保留原来的专家分组与矩阵投影，只把连续命中的组放在同一 command buffer。
投影内核、量化格式、F32/Safe 算术和累加顺序均未改，也没有增加生产依赖或配置。
减少提交和同步开销的取舍参照 [Apple 的 command buffer 指南](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html)。

缓存仍按访问顺序更新 LRU。预读期间保护当前批次使用的全部槽位；当批次占满缓存时，
下一次读取推迟到 GPU 完成之后。读取错误仍在等待 GPU 完成之后传播，部分读取的槽位
保持无效。新增 GPU 测试同时验证多槽位预读保护、满保护集合、命中和真实短读失败。

### 三组完整请求对照

硬件、模型、prompt、输出上限和 256 上下文容量沿用第一轮。Qwen 预算 8 GiB、输出
29 token；GLM 预算 16 GiB、输出 140 token，均到 Eos。A 是第一轮优化后的保存二进制，
B 是本轮发布构建。两版分别先跑一次完整请求预热，保留并排除预热样本；正式顺序仍为
A/B、B/A、A/B，每次新建 Engine，不并行运行本任务的大模型检查。

| 项目 | Qwen A → B | GLM A → B |
| --- | ---: | ---: |
| 生成吞吐（全部 decode 步数 / 全部 decode 耗时） | 3.225 → 4.686 token/s | 1.377 → 1.569 token/s |
| 请求平均总时间（不含装载） | 13.519 → 10.450 s | 118.291 → 104.652 s |
| 请求平均时间变化 | −22.7% | −11.5% |
| 平均预填充 | 4.504 → 4.239 s | 16.563 → 15.361 s |
| 平均装载 | 1.160 → 1.174 s | 2.457 → 2.499 s |
| 单次生成吞吐范围 | 3.148–3.298 → 4.541–4.770 token/s | 1.362–1.388 → 1.564–1.573 token/s |
| 三次中的最大物理 footprint | 8,403,605,960 → 8,407,849,392 B | 16,705,614,088 → 16,708,514,032 B |

每个模型全部六次的输出 ID、文本、Eos、预留字节、Metal 分配、缓存命中/缺失次数和
逻辑读取量一致。物理峰值来自实际 CLI 子进程 `time -l`，两个模型都低于配置预算。
系统文件缓存未清空、负载未控制，只覆盖两个固定输入，所有样本均保留。旧版在本轮
也比第一轮记录更快，不能混用两轮数据计算累计加速或将变化全部归因于某个缓存因素。

普通测试 15 项、Clippy（全部目标、警告为错误）、格式检查和发布构建通过。Qwen
会话状态/分块/取消/错误/复用检查，以及 GLM 状态/取消/复用/11、16、24 GiB 预算和
tokenizer 检查通过。重新输出的 Qwen 全部 47 个位置、GLM 全部 161 个位置 F32 logits
与上一轮逐字节一致，因此继承上一轮固定参考的数值对照；本轮没有重跑参考性能测试。
8192-token Qwen 检查未重跑，GLM 长上下文质量仍待验证。

### 较大缓存与同一引擎复用

另用本轮 B 做单次较大预算探索。先在 GLM 正式测试之后运行 GLM 24/32 GiB，再运行
Qwen 16/40 GiB；每次新建 Engine，文件缓存未清空。以下保留全部样本，改变预算的
结果不计入上面的代码优化收益。请求时间均不含装载。

| 模型 / 预算 | 装载 | 预填充 | 完整请求 | 生成吞吐 | 物理峰值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen / 16 GiB | 1.40 s | 4.67 s | 10.51 s | 4.98 token/s | 15.89 GiB |
| Qwen / 40 GiB | 7.85 s | 7.32 s | 12.27 s | 5.88 token/s | 36.87 GiB |
| GLM / 24 GiB | 2.48 s | 16.58 s | 95.85 s | 1.77 token/s | 23.64 GiB |
| GLM / 32 GiB | 2.39 s | 41.53 s | 172.74 s | 1.07 token/s | 31.73 GiB |

Qwen 40 GiB 可让全部路由专家权重驻留，PLE 表仍按需读取。GLM 更大缓存降低逻辑
读取量，但 32 GiB 的这个请求前半段明显较慢，原因尚未隔离；不能仅凭缓存容量预测
完整请求速度。Qwen 40 GiB 也出现初次预填充和 decode 停顿，不能只保留后面的快步骤。

为进一步观察复用，使用私有 API 验证程序，每个模型只装载一次 Engine，随后新建会话
连续做三次完整请求。该程序使用空流式输出回调，在请求计时后打印结果；核心 Rust/
Metal 源码与 B 相同。它与逐 token 打印的 CLI 接口不同，也没有控制系统缓存和负载，
不能把以下数据与 CLI 样本合并计算加速。三个请求全部保留，没有丢弃首次慢请求。

| 模型 / 预算 | 第一次请求 / 吞吐 | 第二次请求 / 吞吐 | 第三次请求 / 吞吐 |
| --- | ---: | ---: | ---: |
| Qwen / 40 GiB | 23.74 s / 7.24 token/s | 6.31 s / 7.41 token/s | 8.81 s / 7.02 token/s |
| GLM / 32 GiB | 84.54 s / 2.11 token/s | 78.52 s / 2.09 token/s | 79.76 s / 2.05 token/s |

一次性装载耗时分别为 Qwen 8.77 s、GLM 2.55 s；每条 Statistics 中的 load 字段重复
记录这个初始值，不表示每个请求都重新装载。整组三次请求的最大物理 footprint 分别
为 36.88 / 31.73 GiB，低于预算。各预算与全部复用请求的输出 ID、文本和 Eos 都与
低预算对照一致。32 GiB GLM 的两类实验差异未能单独归因于缓存复用、首次执行或系统
负载，仍需更可控的测量才能推荐稳定速度或最佳预算。

完整命令、二进制/源码 SHA-256、全部样本与检查见
[validation/native-performance-2.json](../../validation/native-performance-2.json)。原始日志、F32
输出、复用验证程序及 SHA-256 清单保存在 Git 忽略的 `evidence/native-performance-2-20261012/`。
量化块复用、进一步算子融合、GLM 分组投影预填充、MTP/随机采样仍是后续工作。

## 第三轮实现与性能验收 — 2026-10-13

本轮源码在同一台 M2 Max / 96 GiB 上编译和测试。基线是开始时完整但未提交的源码，
已保存 `evidence/native-performance-3-20261013/baseline-source.tar.gz`，不是 Git HEAD 的早期单层实现。
实现、SHA-256、命令、全部样本和数值分布见 [完整 JSON](../../validation/native-performance-3.json)。

验收中发现独立源码副本共用 Cargo target 导致导出器复用中间版本，已隔离构建目录，
重跑全部相关数值回归；旧产物另行标注并排除。正式测速的原版/候选内核身份核对正确，
隔离重建候选与测速候选的所有 `__TEXT` 节（含机器码和内核常量）逐字节一致。

四行 SIMD 投影、Q2_0/Q2_K/IQ1_M 块信息复用及对齐读取保留 F32/Safe math。
GLM 新增 8-token 分组预填充，按专家聚合 token，批量计算 KDA 独立投影和 dense/shared FFN；
循环、卷积和因果注意力仍按位置更新。稀疏 mask 跳过无效点积，GLM 每个池只计算一次排名。
跨层合并提交实验已撤回；F_NOCACHE 私有实验和 64 GiB GLM 探针停止记录均保留。
命令行默认预算现为 32 GiB。

正常命令提交的独立诊断覆盖四次 Qwen 请求：GPU 命令区间累计 19.34 → 15.61 s；
候选临时分配调用累计 2.08 s，约为请求总时间 22.09 s 的 9.4%。CPU 等待包含 GPU 时间，
这些数值不能相加或当作硬件占用率。基于这项测量，本轮没有增加缓冲区池的所有权/生命周期抽象。
诊断产物只存在于 evidence，不进入运行版本。

每个进程先完成一次预热，再计一个请求；按 A/B、B/A、A/B 顺序，各取三个正式样本。
所有预热、正式样本和波动均保留，完整输出 token 与 EOS 一致。上下文容量 256，
Qwen 输出 29 token，GLM 默认 Max 思考输出 140 token；decode 是总步数 / 总 decode 时间。
请求时间含 prefill，不含 Engine 装载。文件缓存、后台负载未隔离，Qwen 40 GiB 部分样本有
Godot 进程并行。完整 JSON 提供 median、范围、样本方差和逐 token 延迟。

| 条件 | decode token/s | 提升 | 平均请求 s | 候选峰值 GiB |
| --- | ---: | ---: | ---: | ---: |
| Qwen / 8 GiB | 5.58 → 6.27 | +12.3% | 9.03 → 7.86 | 7.846 |
| Qwen / 40 GiB | 7.21 → 7.95 | +10.4% | 6.86 → 5.75 | 36.884 |
| GLM / 32 GiB | 2.22 → 2.63 | +18.8% | 74.18 → 62.88 | 31.736 |

Qwen 中英文/代码 69 个位置、GLM 66 个位置的优化前后全词表 logits 均逐位一致；
各短输入额外检查不同分块方式。两个模型的 2112-token 状态回归均在 1、8、64、256、512、
1024、2047、2048、2052、2055、2112 共 11 个位置逐位一致，所有 logits 有限、峰值低于预算。
这些长序列使用重复中英代码片段，是状态/边界回归，不是长文本理解质量验收。
独立固定版本 reference 对新增短语料核对了 135 个完整位置，top token 一致 135/135；
每组最大误差、KL、总变差及任何不一致位置均在 JSON 中保留，不把近似浮点结果称为位级一致。

真实权重的 13 种投影格式、独立解量化/分词对照、矩阵行尾/批尾/专家偏移、稀疏池尾/同分选择、
1/8-token GLM 执行中取消、会话复用和预算回归均完成；fmt、Clippy `-D warnings`、release build
及测试通过。测试过的 CLI 二进制位于 `target/release/ingot`，依赖检查未引入 llama/ggml。

复测工具：`examples/bench.rs`、`examples/validate.rs`、`tools/bench_native.py` 和
`tools/check_native_regression.py`。原始 token 文件、reference 构建脚本与所有日志位于
`evidence/native-performance-3-20261013`。第三轮没有重跑完整 8192-token smoke；独立长上下文
质量对照、MTP 和随机采样尚未完成。
