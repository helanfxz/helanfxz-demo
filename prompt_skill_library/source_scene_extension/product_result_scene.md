---
id: source_scene_extension.product_result_scene
strategy: source_scene_extension
required_slots:
  - product_type
  - appearance
  - result_visual
  - result_action
  - scene_goal
forbidden_if:
  - requested_scene_transfer == true
  - product_presence != required
  - source_asset_missing == true
  - action_count > 1
failure_tags:
  - no_product
  - scene_conflict
  - prompt_intent_lost
  - identity_drift
  - bad_physics
success_stats:
  default: {success: 0, failure: 0}
purpose: A conservative fidelity shot 3, product-visible commerce result proof.
---
# A 保守策略：素材场景商品结果证明镜

## 用途

用于 A_conservative_fidelity 的第三个 5 秒图生视频镜头。这个模板修复一个明确踩过的坑：保守策略不能把带货结果镜写成无商品生活空镜。A 保守的是动作幅度、场景变化和身份风险，不是让该出现商品的镜头没有商品。

本镜继续从真实素材首帧出发，允许少量同场景生活道具或承托关系扩展，但不切到完全不同地点，不让模型自由重建商品。

## Prompt 模板

这是 5 秒图生视频，竖屏 9:16，真实写实的商品带货短视频。第一帧仍使用上传素材中的真实{{product_type}}，商品主体必须清楚可见。这个镜头是带货结果证明镜，不是无商品铺垫镜头，也不是纯生活空镜。

镜头目标：{{scene_goal}}。画面必须一直保留同一件{{product_type}}作为主体，商品身份：{{appearance}}。商品必须来自上传素材首帧，保持同一件商品，不重绘为类似商品，不新增第二个同类商品。

视觉方向：{{result_visual}}。可以在首帧素材场景边缘轻微扩展生活道具、承托物、桌面边缘、背包一角或自然光变化，但这些元素只能服务商品结果表达，不能替代商品主体，也不能把画面带到完全不同地点。

动作安排：0.0-1.0 秒承接素材首帧，先稳定确认商品；1.0-4.0 秒只执行一个低风险动作：{{result_action}}；4.0-5.0 秒动作停稳，最后一帧商品仍清楚可见，观众能从商品和周边道具关系理解卖点结果。

商品自带 logo、标识或字样只保持首帧已有外观，不要改写、重画或发明新字母。不要新增非商品自带文字、字幕、UI、水印或购物按钮。商品始终贴合真实承托关系运动，轮廓、比例和结构保持稳定；如果出现手部或道具接触，必须先接触再移动，支撑关系清楚。
