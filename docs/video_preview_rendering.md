# 视频预览生成模块规划

## 1. 模块目标

这个模块保留为 Seedance 不可用时的本地预览兜底。正常链路已经接入 Seedance 和混合渲染策略。

这一步用于验证：

- 上传文件是否真的保存到本地
- 多模态模型是否能读取图片内容
- 剧本和分镜是否能转成可观看的视频草稿
- 页面是否能展示最终视频结果

## 2. 实现边界

新增模块：

- `agent/simple_video_renderer.py`

职责：

- 读取分镜列表
- 选择已上传图片素材
- 为每个分镜生成一段静态画面
- 叠加字幕
- 合成为 MP4

不负责：

- 调用 Seedance 图生视频
- 生成真实镜头运动
- TTS 配音
- BGM
- 高级剪辑时间线

## 3. 为什么保留本地 MP4 预览

Seedance 视频生成是异步、耗时、并发低、成本更高的能力。当前如果直接接视频生成模型，容易把问题混在一起：

- 是素材没保存好？
- 是多模态理解不准？
- 是分镜不合理？
- 是视频模型接口失败？

所以项目仍然保留 `render_mode=local_preview` 作为失败兜底。正常链路优先执行 Seedance 商品镜渲染；统一空背景承接镜则使用本地淡入商品锚点的方式生成。

## 4. 输入输出

输入：

- `task_id`
- `storyboard`
- `asset_matching`
- `output_dir`

输出：

```json
{
  "render_mode": "local_preview",
  "video_path": ".uploads/task_xxx/preview.mp4",
  "video_url": "/uploads/task_xxx/preview.mp4",
  "success": true,
  "error": null
}
```

## 5. 审核策略

本地渲染前做规则检查：

- 是否存在分镜
- 是否存在可用图片素材
- 每个分镜是否有字幕
- 输出目录是否可写

如果渲染失败，不打回剧本或分镜，而是进入 `render_failed` 状态，因为这类失败通常来自依赖、路径或素材格式。
