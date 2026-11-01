# Ingot

[English](README.md)

在你自己的 Apple Silicon Mac 上本地运行 AI 对话模型：Qwen3.8-Flash-Next、
GLM-5.3-Flash，以及常见的 Llama、Qwen2、Qwen3、Qwen3-MoE GGUF 模型。

## 从这里开始

ingot 在你的 Mac 上私密地运行 AI 对话模型：你输入的内容不会离开你的电脑。
需要一台 Apple Silicon Mac（M1 或更新）和 macOS 15 及以上版本。

1. **安装：** 克隆本仓库并运行 `scripts/install.sh`。
2. **对话：** 运行 `ingot`，选择一个模型，确认下载，然后开始输入。
3. **按内存选模型：** 16 GB 的 Mac → `qwen3-8b-q4_k_m`（4.7 GiB），
   32 GB → `qwen3-30b-a3b-q4_k_m`（17.3 GiB），64 GB 及以上 →
   `qwen3.8-iq2_xs`（63.4 GiB）。`ingot models` 会显示哪些模型适合你的 Mac。
4. **在浏览器或其他应用中使用：** 运行 `ingot serve`，然后打开
   <http://127.0.0.1:8080/>；也可以让 Open WebUI、Cursor 或 Claude Code 连接它。

### 安装

```sh
git clone https://github.com/majiayu000/ingot.git
cd ingot
scripts/install.sh
```

安装脚本不会使用 `sudo`，也不会要求输入密码。它会检查 Xcode 命令行工具和
Rust（1.88 及以上），缺少时会告诉你如何安装。仓库目前是私有的，需要 GitHub
访问权限。

### 第一次对话

```sh
ingot
```

看到 **Ready!** 后输入消息并回车。`/reset` 开始新对话，`/exit` 退出，
Ctrl-C 停止当前回答。下载中断时重新运行同一条命令即可继续，文件会自动校验。

## 指南（英文）

- [快速开始](docs/QUICKSTART.md)：安装、第一次对话、按内存选择模型、浏览器、
  Open WebUI、Cursor、Claude Code、手机和其他电脑。
- [出问题了？](docs/TROUBLESHOOTING.md)：常见错误信息及解决办法。
- [让 AI 帮你安装](docs/AI_SETUP.md)：交给 AI 编程助手执行的安装说明。

完整的技术说明见 [README.md](README.md)（英文）；设计文档
[ARCHITECTURE.md](ARCHITECTURE.md) 为中文。

这是一个早期项目，目前只在一台 M2 Max（96 GiB）上测试过。
