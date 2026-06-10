---
id: commerce_scene.source_confirm
strategy: ideal_commerce_scene
required_slots:
  - product_type
  - appearance
  - source_place
  - style
forbidden_if:
  - requested_scene_transfer == true
  - action_count > 0
  - source_asset_missing == true
failure_tags:
  - identity_drift
  - extra_product
  - scene_conflict
  - too_static
  - logo_changed
success_stats:
  default: {success: 0, failure: 0}
purpose: B ideal commerce scene shot 1, source-frame identity confirmation.
---
# B 候选：素材首帧商品确认镜

## 用途

用于 B_ideal_commerce_scene 的第一个 5 秒图生视频镜头。这个镜头不负责讲完整剧情，只负责让观众先确认上传素材里的真实商品身份，避免后续较大胆的带货场景变成另一个商品。

这个模板保留了之前实跑得到的关键边界：第一帧就是素材，不换地点，不塞复杂动作，不新增非商品自带文字，同时允许商品自带 logo、标识或字样按首帧保持。

## Prompt 模板

这是 5 秒图生视频，竖屏 9:16，真实写实的商品带货短视频。第一帧就是上传素材中的同一件{{product_type}}，先不要改变地点，也不要把素材重建成新房间、新桌面、户外或棚拍背景。

0.0-1.0 秒：完全保持首帧构图和商品位置，让观众看清真实商品。商品不能跳动、不能自己移动、不能变成另一个类似商品。  
1.0-3.5 秒：只让{{source_place}}里的自然光、阴影和反射发生很轻微的生活化变化，镜头可以有非常轻的呼吸感或缓慢推近，但仍然停留在同一个素材空间。  
3.5-5.0 秒：画面稳定在商品清晰近景，观众能确认商品的颜色、轮廓、材质、结构和标识区域。

商品外观必须保持：{{appearance}}。画面里可以有桌面、手边普通道具和柔和阴影，但不要新增第二个同类商品，不要新增非商品自带文字、字幕、UI、水印或购物按钮；商品自带 logo、标识或字样只保持首帧已有外观，不要改写、重画或发明新字母。

整体风格：{{style}}。这个镜头只负责确认真实商品身份，不讲复杂剧情，不表现拿起、放入、饮用、打开、旋转或跨地点移动。
