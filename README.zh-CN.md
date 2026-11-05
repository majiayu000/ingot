# Ingot

[![CI](https://github.com/majiayu000/ingot/actions/workflows/ci.yml/badge.svg)](https://github.com/majiayu000/ingot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

在你的 Apple Silicon Mac 上运行 AI 模型，可以用终端、浏览器，也可以接入你常用的应用。
Ingot 是一个 Rust 程序，使用自己的 Metal kernel，运行时不依赖 llama.cpp 或 MLX。

[English](README.md) ·
[快速开始（英文）](docs/QUICKSTART.md) ·
[参考手册（英文）](docs/REFERENCE.md) ·
[性能数据（英文）](docs/PERFORMANCE.md)

## 开始使用

需要 Apple Silicon Mac 和 macOS 15 及以上。目前还没有预编译版本，需要先用 Rust 1.88 及以上和
Xcode 命令行工具构建：

```sh
git clone https://github.com/majiayu000/ingot.git
cd ingot
scripts/package-app.sh       # 生成 Mac 应用和磁盘映像，在 dist/ 目录
scripts/install.sh           # 或者只安装命令行工具
```

**Mac 应用：** 打开 `dist/Ingot-0.1.0.dmg`，把 Ingot 拖到“应用程序”。打开后会看到按这台 Mac
评估过的模型列表，选一个，等下载完成，就可以开始对话。聊天、项目分组、PDF 和文本附件、自定义模型、
设置都在应用里。Ingot 会留在菜单栏，方便其他应用连接。

**命令行：** 运行 `ingot`，选择模型并确认下载。

你的对话内容只留在自己的 Mac 上。

## 功能

- **随处对话：** 可以在 Mac 应用和终端使用；运行 `ingot serve` 后，也可以打开内置网页
  <http://127.0.0.1:8080/>。
- **接入常用工具：** 提供兼容 OpenAI、Anthropic 和 Ollama 的 API，可连接 Open WebUI、
  Cursor、Claude Code，以及任何带 Ollama 设置的应用。
- **运行超大模型：** 支持 Qwen3.8-Flash-Next 和 GLM-5.3-Flash，内存放不下的专家层从
  SSD 流式读取。
- **运行常见 GGUF 模型：** 支持 Llama、Qwen2、Qwen3、Qwen3-MoE、Gemma 3、Gemma 4
  和 Phi-3 系列文件，可从内置列表下载，也可直接从 Hugging Face 下载。
- **发现和定制模型：** 在应用里搜索 Hugging Face，保存带自己提示词和参数的自定义模型，也可以导入 Modelfile。
- **启动你的工具：** `ingot launch claude`、`codex` 或 `opencode`。
- **细节齐全：** 工具调用、思考等级、图片输入（Qwen3.8）、API 密钥、Prometheus
  指标、后台服务，以及可续传并经 SHA-256 校验的下载。

## 选哪个模型

打开应用，或运行 `ingot` 后直接按回车。Ingot 会检查这台 Mac 的内存和可用磁盘空间，推荐放得下的最佳模型，
并选好上下文长度。已下载且放得下的模型会优先使用。

举例，下面是上下文为 8192 token 时各模型需要的内存：

| 模型 | 下载大小 | 内存 |
| --- | ---: | ---: |
| `gemma3-4b-it-qat-q4_0` | 2.4 GiB | 5.0 GiB |
| `qwen3-8b-q4_k_m` | 4.7 GiB | 7.5 GiB |
| `qwen3-30b-a3b-q4_k_m` | 17.3 GiB | 19.3 GiB |
| `qwen3.8-iq2_xs` | 63.4 GiB | 41 GiB |

这里的内存指 GPU 可用内存，比 Mac 的总内存少。`ingot` 启动时的第一屏会显示你这台 Mac 的数值。

```sh
ingot models                # 查看全部模型，以及哪些放得下
ingot list                  # 查看已有模型
ingot ps                    # 查看已加载的模型
ingot stop qwen3-8b-q4_k_m  # 立即释放它的内存
ingot run qwen3-8b-q4_k_m   # 跳过菜单
```

其他受支持的 GGUF 可以用 `ingot pull hf:OWNER/REPO/FILE.gguf` 下载。

## 连接应用

```sh
ingot serve qwen3-8b-q4_k_m
```

| 客户端 | 设置 |
| --- | --- |
| 浏览器 | 打开 <http://127.0.0.1:8080/> |
| Open WebUI、Cursor 等 OpenAI 客户端 | Base URL 填 `http://127.0.0.1:8080/v1`，模型名从 `/v1/models` 获取 |
| 带 Ollama 设置的应用 | 服务器填 `http://127.0.0.1:8080` |
| Claude Code | [配置指南](docs/QUICKSTART.md#claude-code) |

Mac 应用会自动在 11435 端口启动这个服务，可以在菜单栏图标里选 **Copy API Address** 复制地址。
服务默认只监听本机。要从手机或其他电脑访问，请看
[远程访问](docs/REFERENCE.md#remote-access-and-api-keys)。

## 性能

以下数据都在同一台 M2 Max（38 核 GPU，96 GiB）上测得。耗时比小于 1 表示 Ingot 更快。

| 对比对象 | 测试 | 结果 |
| --- | --- | --- |
| [Strata](docs/PERFORMANCE.md#vs-strata-on-mac) | Qwen3.8，相同 GGUF 和 MTP head，1K、4K 提示词 | 请求耗时比 0.79–0.92；生成速度快 15–47% |
| Strata | 同上，16K 提示词，不开 MTP | 打平（0.98–1.00）；Strata 的提示词处理快 4–7% |
| [oMLX](docs/PERFORMANCE.md) | Qwen3.8，量化方式不同 | 1K 为 0.82，4K 为 1.01；oMLX 的 4-bit 权重质量得分更高 |
| [llama.cpp](docs/ENGINE.md#generic-architectures) | 7 个标准 GGUF 模型 | logits 误差约 1e-4；生成速度相差 ±7% 以内；提示词处理慢 8–20% |

Ingot 最强的是专门优化过的 Qwen3.8 和 GLM-5.3。标准 GGUF 模型结果正确、速度接近，
但提示词处理仍然比 llama.cpp 慢。

## 当前状态

Ingot 还是早期项目。

- 只在一台机器上测试过，其他芯片、内存大小和 macOS 版本都没有测试。
- 服务一次只运行一个推理请求，不支持 TLS。
- GLM 不支持工具调用。图片输入需要 Qwen3.8 及其视觉投影模型。
- 其他 GGUF 架构会直接报错。

完整限制见 [引擎文档](docs/ENGINE.md#status-and-limitations)。

## 文档

除特别注明外，文档都是英文。

| 指南 | 内容 |
| --- | --- |
| [快速开始](docs/QUICKSTART.md) | 安装、第一次对话、选模型、连接应用和设备 |
| [故障排查](docs/TROUBLESHOOTING.md) | 常见错误及解决办法 |
| [AI 安装说明](docs/AI_SETUP.md) | 交给 AI 编程助手执行的步骤 |
| [参考手册](docs/REFERENCE.md) | 全部命令、选项和 API 细节 |
| [引擎](docs/ENGINE.md) | 引擎原理和支持的权重 |
| [性能](docs/PERFORMANCE.md) | 完整测试方法和结果 |
| [开发](docs/DEVELOPMENT.md) | 构建、验证和仓库结构 |
| [架构](docs/ARCHITECTURE.md) | 设计与取舍（中文） |
| [来源](PROVENANCE.md) | 代码来源和改编内容 |
| [历史记录](docs/history/) | 带日期的存档，未随当前代码更新 |

## 许可证

MIT，见 [LICENSE](LICENSE)。量化格式和部分 kernel 技术改编自 ggml/llama.cpp（MIT 许可），
详见 [kernels/NOTICE](kernels/NOTICE) 和 [macos/NOTICE](macos/NOTICE)。模型权重不包含在内，
遵循各自的许可证。
