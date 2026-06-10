# 提交材料总览

## 基础信息

- 项目名称：电商场景 AIGC 带货视频生成系统
- 参赛课题：电商场景 AIGC 带货视频生成系统
- 当前完成度：可运行 MVP / Demo
- 一句话价值：商家上传商品素材并填写卖点后，系统自动生成可预览的带货短视频草稿，并保留分镜、素材绑定、渲染和审核链路。

## 交付入口

- 在线 Demo 链接：待补充实际部署地址；本地体验见根目录 `README.md`。
- 演示视频链接：待补充公开录屏链接。
- 源码仓库链接：待补充 GitHub / GitLab 地址、分支和最后提交记录。
- 本地启动：`python -m pip install -e ".[dev]"` 后运行 `python task_creation_demo_app.py`。

## 必备文档

- [项目 README](../../README.md)
- [完赛项目提报内容](final_submission.md)
- [系统架构说明](system_architecture.md)
- [API 与复核接口说明](api_reference.md)
- [演示脚本](demo_script.md)
- [评测方案与样例](evaluation_plan.md)
- [提交清单](submission_checklist.md)

## 核心功能

- 商品素材上传与任务创建。
- 商品身份卡、多模态素材理解和素材能力计划。
- LLM + prompt skill 驱动的带货分镜规划。
- Seedance 文生视频 / 图生视频渲染与本地预览降级。
- 内容审核、局部修复、最终检查和任务状态反馈。

## 复核方式

1. 按根目录 `README.md` 安装依赖并启动 Demo。
2. 在页面填写商品类型、卖点、目标人群、使用场景和风格信息。
3. 上传商品图，提交任务并等待进度完成。
4. 在任务详情页查看 A/B 候选视频、分镜、素材绑定和审核结果。
5. 打开任务详情页左侧“导出任务报告”，复核结构化 JSON、视频路径和 artifact 目录。
6. 运行 `python -m pytest -q` 验证核心回归测试。
