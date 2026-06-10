# 电商场景 AIGC 带货视频生成系统

商家上传商品素材并填写卖点、使用场景、目标人群后，系统自动完成素材理解、商品身份约束、带货分镜规划、Seedance 渲染、本地预览降级、内容审核和结果展示。

当前版本是可运行 MVP / Demo。没有火山方舟密钥时，项目会自动走本地预览降级，方便评审或同学下载后直接体验完整工程链路。

## 一键启动

### Windows

双击 `start.bat`，或在命令行执行：

```bat
start.bat
```

如果项目放在 `\\wsl.localhost\...` 路径下，`start.bat` 会自动转到 WSL 内执行 `start.sh`，避免 Windows `cmd` 不支持 UNC 当前目录导致闪退。

### macOS / Linux

```bash
chmod +x start.sh
./start.sh
```

启动后访问：

```text
http://127.0.0.1:8010
```

脚本会自动创建 `.venv`、安装依赖并启动 Demo。如果没有配置 `ARK_API_KEY`，会默认设置 `AIGC_DISABLE_LLM=1` 和 `AIGC_DISABLE_VIDEO_MODEL=1`，使用本地降级链路。

## Docker 启动

```bash
docker compose up --build
```

然后访问：

```text
http://127.0.0.1:8010
```

如需调用真实模型，把环境变量写入本机环境或 `.env`：

```bash
cp .env.example .env
ARK_API_KEY=...
ARK_TEXT_ENDPOINT_ID=...
ARK_VIDEO_ENDPOINT_ID=...
AIGC_DISABLE_LLM=0
AIGC_DISABLE_VIDEO_MODEL=0
```

## 手动启动

```bash
python -m pip install -e ".[dev]"
python task_creation_demo_app.py
```

## 环境变量

- `ARK_API_KEY`：火山方舟 API Key。
- `ARK_TEXT_ENDPOINT_ID`：文本/多模态模型 endpoint。
- `ARK_VIDEO_ENDPOINT_ID`：Seedance 视频模型 endpoint。
- `AIGC_DISABLE_LLM=1`：禁用 LLM，使用规则兜底。
- `AIGC_DISABLE_VIDEO_MODEL=1`：禁用真实视频模型，使用本地预览降级。
- `AIGC_DISABLE_BACKGROUND_REMOVAL=1`：禁用 rembg 抠图。
- `HOST`：服务监听地址，默认 `127.0.0.1`，Docker 中为 `0.0.0.0`。
- `PORT`：服务端口，默认 `8010`。

## 项目结构

- `task_creation_demo_app.py`：FastAPI 页面、素材上传、任务启动、任务详情和结果展示。
- `video_task_module.py`：任务领域模型、状态流转和内存仓储。
- `agent/`：视频生成核心后端，包括素材预处理、需求结构化、分镜规划、prompt 安全、Seedance 渲染、内容修复和最终检查。
- `prompt_skill_library/`：prompt skill 样例、反例和最终视频 prompt 结构规范。
- `tests/`：核心回归测试，覆盖素材绑定、商品身份一致性、prompt 安全、内容修复和前端字段传递。
- `docs/`：模块说明、项目背景、视频工作流和提交材料。
- `docs/submission/`：评审提交资料，包括架构说明、演示脚本、评测方案、提交清单和最终提报内容。

## 核心链路

```text
用户输入 + 上传素材
  -> 素材预处理和主商品确认
  -> 需求结构化 + 多模态素材分析
  -> 商品上下文和素材能力计划
  -> LLM / prompt skill 分镜规划
  -> 可拍性审核和素材匹配
  -> Seedance 渲染或本地预览降级
  -> 内容审核、局部修复和最终检查
```

## 测试

```bash
python -m pytest -q
```

## 提交资料

评审优先阅读：

- `docs/submission/final_submission.md`
- `docs/submission/system_architecture.md`
- `docs/submission/demo_script.md`
- `docs/submission/evaluation_plan.md`
