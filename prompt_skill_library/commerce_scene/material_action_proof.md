---
id: commerce_scene.material_action_proof
strategy: ideal_commerce_scene
required_slots:
  - product_type
  - appearance
  - source_action
  - first_point
  - style
forbidden_if:
  - requested_scene_transfer == true
  - action_count > 1
  - source_asset_missing == true
failure_tags:
  - bad_physics
  - identity_drift
  - extra_product
  - scene_conflict
  - prompt_intent_lost
success_stats:
  default: {success: 0, failure: 0}
purpose: B ideal commerce scene shot 2, source-frame material action proof.
---
# B 候选：素材场景动作证明镜

## 用途

用于 B_ideal_commerce_scene 的第二个 5 秒图生视频镜头。它从上传素材首帧出发，但比第一镜更主动：用一个清楚、连续、物理关系明确的商品动作来证明卖点。

这个模板重点解决之前的失败：一个镜头塞太多动作会像快进；动作没有接触关系会出现商品自己跳动；跨地点会把素材场景和剧情场景混在一起。因此本镜只允许一个动作，不换地点，动作完成后停稳。

## Prompt 模板

这是 5 秒图生视频，竖屏 9:16，真实写实的商品带货短视频。第一帧仍然来自上传素材中的同一件{{product_type}}，本镜头不换地点，不硬切到户外、办公室入口、玄关之外的新空间，也不把多个时间地点混进同一镜头。

本镜头只发生一个主要动作：{{source_action}} 这个动作要让观众从画面里看出卖点「{{first_point}}」，不要只靠字幕解释。

0.0-1.0 秒：承接素材首帧，商品位置、承托面、背景和光线保持稳定，先让观众确认这是同一件商品。  
1.0-2.0 秒：如果有人手参与，手必须先真实接触商品的稳定部位，再开始施力；如果没有手参与，只允许商品周围环境、光影或承托物轻微变化，商品始终保持真实承托关系。  
2.0-4.0 秒：执行唯一动作，运动路径短而连续，商品保持刚体感，颜色、轮廓、结构和标识区域不被大幅重绘。  
4.0-5.0 秒：动作结束并停稳，最后一帧商品仍清楚可见，能从商品位置、手部接触或周边道具关系理解卖点。

商品外观必须保持：{{appearance}}。商品自带 logo、标识或字样只保持首帧已有外观，不要新增、改写或重画。不要新增第二个同类主商品，不要新增非商品自带文字、字幕、UI、水印或购物按钮。

手部、承托面、背包、桌面或其他道具必须有真实接触和支撑关系；商品轮廓、比例、材质和结构保持稳定，运动只来自手部或承托物的真实带动。背景只允许在首帧素材场景基础上轻微扩展，例如桌面边缘、背包一角或自然光，不要跨到全新的生活场景。

整体风格：{{style}}。
