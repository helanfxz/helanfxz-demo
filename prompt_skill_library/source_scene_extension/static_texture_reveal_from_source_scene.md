---
id: source_scene_extension.static_texture_reveal_from_source_scene
strategy: source_scene_extension
required_slots:
  - source_scene_description
  - product_description
  - product_location
  - texture_focus
  - end_state
forbidden_if:
  - requested_scene_transfer == true
failure_tags:
  - too_static
  - shot_redundant
success_stats:
  default: {success: 0, failure: 0}
---
# 原素材场景静态质感展示

## 用途
当动作风险较高时，用原素材首帧做保真质感展示。

## 适用条件
素材图质量较高，需要展示 logo、材质、结构或文字，但不适合复杂动作。

## Prompt 模板
竖屏 9:16，写实商品短视频，第一帧就是参考图里的真实画面。

画面环境必须顺着第一帧来理解：{source_scene_description}。主商品是{product_description}，位于{product_location}。

这个镜头不做大位移，只展示商品真实质感和细节。地点不变化，不切换到新场景。重点是{texture_focus}。

0.0-1.0 秒：商品保持在原位置，镜头稳定，观众能看清整体轮廓。
1.0-2.5 秒：镜头缓慢推近或轻微侧移，光线自然扫过材质，商品结构不改变。
2.5-4.0 秒：保持商品主体清晰，背景轻微虚化，不新增第二个商品。
4.0-5.0 秒：最后一帧展示{end_state}。
