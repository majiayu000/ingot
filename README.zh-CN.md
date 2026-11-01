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

需要 Apple Silicon Mac、macOS 15 及以上、Rust 1.88 及以上和 Xcode 命令行工具。
安装脚本会逐项检查。目前还没有预编译版本。

```sh
git clone https://github.com/majiayu000/ingot.git
cd ingot
scripts/install.sh
ingot
```

选择模型，确认下载，就可以开始对话。你的对话内容只留在自己的 Mac 上。

## 功能

- **随处对话：** 可以在终端使用；运行 `ingot serve` 后，也可以打开内置网页
  <http://127.0.0.1:8080/>。
- **接入常用工具：** 提供兼容 OpenAI 和 Anthropic 的 API，可连接 Open WebUI、
  Cursor、Claude Code 等客户端。
- **运行超大模型：** 支持 Qwen3.8-Flash-Next 和 GLM-5.3-Flash，内存放不下的专家层从
  SSD 流式读取。
- **运行常见 GGUF 模型：** 支持 Llama、Qwen2、Qwen3、Qwen3-MoE、Gemma 3、Gemma 4
  和 Phi-3 系列文件，可从内置列表下载，也可直接从 Hugging Face 下载。
- **细节齐全：** 工具调用、思考等级、图片输入（Qwen3.8）、API 密钥、Prometheus
  指标、后台服务，以及可续传并经 SHA-256 校验的下载。

## 选哪个模型

| 你的 Mac | 模型 | 下载大小 |
| --- | --- | ---: |
| 16 GB | `qwen3-8b-q4_k_m` | 4.7 GiB |
| 32 GB | `qwen3-30b-a3b-q4_k_m` | 17.3 GiB |
| 64 GB 及以上 | `qwen3.8-iq2_xs` | 63.4 GiB |

```sh
ingot run qwen3-8b-q4_k_m   # 需要时下载，然后开始对话
ingot models                # 查看全部模型，以及哪些适合你的 Mac
```

列表里还有 Gemma、Phi、Llama 3.1 和 GLM-5.3。其他受支持的 GGUF 可以用
`ingot pull hf:OWNER/REPO/FILE.gguf` 下载。

## 连接应用

```sh
ingot serve qwen3-8b-q4_k_m
```

| 客户端 | 设置 |
| --- | --- |
| 浏览器 | 打开 <http://127.0.0.1:8080/> |
| Open WebUI、Cursor 等 OpenAI 客户端 | Base URL 填 `http://127.0.0.1:8080/v1`，模型名从 `/v1/models` 获取 |
| Claude Code | [配置指南](docs/QUICKSTART.md#claude-code) |

服务默认只监听本机。要从手机或其他电脑访问，请看
[远程访问](docs/REFERENCE.md#remote-access-and-api-keys)。

## 性能

以下数据都在同一台 M2 Max（38 核 GPU，96 GiB）上测得。耗时比小于 1 表示 Ingot 更快。

| 对比对象 | 测试 | 结果 |
| --- | --- | --- |
| [Strata](docs/PERFORMANCE.md#vs-strata-on-mac) | Qwen3.8，相同 GGUF 和 MTP head，1K、4K 提示词 | 请求耗时比 0.79–0.92；生成速度快 15–47% |
| Strata | 同上，16K 提示词，不开 MTP | 打平（0.98–1.00）；Strata 的提示词处理快 4–7% |
| [oMLX](docs/PERFORMANCE.md) | Qwen3.8，量化方式不同 | 1K 为 0.82，4K 为 1.01；oMLX 的 4-bit 权重质量得分更高 |
| [llama.cpp](docs/ENGINE.md#generic-architectures) | 7 个标准 GGUF 模型 | logits 误差约 1e-4；生成速度相差 ±7% 以内；提示词处理慢 8–19%，Gemma 4 慢 53% |

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
