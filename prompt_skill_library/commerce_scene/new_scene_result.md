---
id: commerce_scene.new_scene_result
strategy: ideal_commerce_scene
required_slots:
  - product_type
  - appearance
  - result_place
  - human
  - result_state
  - result_action
  - second_point
  - identity_clause
  - style
forbidden_if:
  - render_strategy == image_to_video
  - product_presence == forbidden
  - action_count > 1
  - logo_text_risk >= 4
failure_tags:
  - wrong_product
  - logo_changed
  - shape_changed
  - second_product_generated
  - scene_not_expressed
  - prompt_intent_lost
success_stats:
  default: {success: 0, failure: 0}
purpose: B ideal commerce scene shot 3, independent new-scene result proof.
---
# B 候选：新场景结果证明镜

## 用途

用于 B_ideal_commerce_scene 的第三个 5 秒文生视频镜头。这个镜头故意不使用素材图作为首帧，而是测试视频模型是否能根据素材理解出的商品外观，在一个新的带货使用场景里生成“结果状态”。

这个模板必须把时间和空间边界说清楚：这是硬切后的新镜头，新地点、新时间，不承接上一镜的桌面、室内房间、手拿动作或背景。它不是“从上一镜走到新地点”，也不是在原素材背景里加一个人物。

## Prompt 模板

这是 5 秒文生视频，竖屏 9:16，真实写实的商品带货短视频。这个镜头是硬切后的新分镜，新地点、新时间，不使用素材图作为第一帧，也不承接上一镜的桌面、房间、手部动作、包装画面或背景物体。

镜头一开始已经在{{result_place}}，不要表现从上一场景走过来的过程。整个 5 秒都发生在这个新地点，不切回素材场景，不跨到第二个地点。画面主体是{{human}}和同一件{{product_type}}的使用结果状态。

商品外观根据素材理解来复刻：{{appearance}}。{{identity_clause}} 商品必须是画面里的主商品之一，不能变成泛生活场景里的无关道具，也不能被包、手或人物完全遮住。

画面具体状态：{{result_state}}。

0.0-1.0 秒：建立新场景，商品已经处在结果状态里，位置清楚可见，不表现拿起、塞入、取出或从上一镜移动过来。  
1.0-2.5 秒：人物或道具只做轻微辅助动作，动作必须围绕商品结果状态展开：{{result_action}}  
2.5-4.0 秒：镜头稳定展示商品与场景道具的关系，让观众从画面理解「{{second_point}}」，不是只出现一个泛生活背景。  
4.0-5.0 秒：停在清楚的商品结果画面，商品仍只有一个主商品，颜色、轮廓、关键结构和可复刻标识区域保持一致。

不要新增非商品自带文字、字幕、UI、购物按钮或水印；不要出现第二个同类主商品；不要把商品改成其他品类；不要让错误 logo、随机字母或品牌幻觉成为画面焦点。

整体风格：{{style}}。
