# API 与复核接口说明

本项目的 Demo 服务默认运行在 `http://127.0.0.1:8010`。以下接口用于评审复核、运行状态检查和任务结果导出，不会触发视频重新生成。

## 健康检查

```http
GET /api/health
```

返回当前服务实例和关键配置状态。接口只返回布尔值，不泄露 API Key 或 endpoint 明文。

主要字段：

- `status`：服务状态，正常为 `ok`。
- `server_pid`：当前进程 PID。
- `run_instance_id`：本次服务启动实例 ID。
- `started_at`：服务启动时间。
- `port`：当前端口，默认 `8010`。
- `upload_root`：上传和产物目录。
- `disable_llm`：是否禁用文本 / 多模态模型。
- `disable_video_model`：是否禁用视频模型。
- `ark_text_configured`：是否已配置火山方舟文本 / 多模态能力。
- `ark_video_configured`：是否已配置火山方舟视频能力。
- `task_count`：当前内存仓储中的任务数量。

示例：

```bash
curl http://127.0.0.1:8010/api/health
```

## 任务状态轮询

```http
GET /api/tasks/{task_id}
```

前端详情页使用该接口轮询任务状态。返回任务状态、当前阶段、进度、事件和阶段性 `workflow_result`。该接口只读，不会启动或重试任务。

## 任务报告导出

```http
GET /tasks/{task_id}/report.json
```

返回单个任务的可复核 JSON 报告，适合随演示视频或截图一起提交。

主要字段：

- `generated_at`：报告导出时间。
- `task`：任务基础信息、用户输入、上传素材、进度事件和状态。
- `workflow_result`：剧本、分镜、创作计划、渲染结果、A/B 候选、审核和最终检查等结构化结果。
- `artifact_dir`：本地 artifact 目录，通常为 `.uploads/{task_id}/artifacts`。
- `video_urls`：A/B 成片摘要，包含视频 URL、文件路径、渲染模式和耗时。

示例：

```bash
curl http://127.0.0.1:8010/tasks/task_xxx/report.json
```

## 复核建议

1. 先访问 `/api/health`，确认服务实例、端口、模型开关和任务数量。
2. 在页面完成一次任务后，打开任务详情页左侧的“导出任务报告”。
3. 对照报告中的 `script_plan`、`storyboard`、`creation_plan`、`render_result`、`content_review` 和 `trace_summary` 检查生成链路。
4. 如需复查模型 prompt 或分阶段产物，查看 `artifact_dir` 指向的本地目录。
