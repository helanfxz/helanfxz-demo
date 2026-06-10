# Prompt Skill Library

This library stores validated prompt-building references for product video shots.
Skills are organized by shot capability, not by hard-coded product type.

The experiment runner reads the YAML-like front matter at the top of each skill
file, validates slot coverage, and then composes a Chinese natural-language
prompt. The JSON/contract fields are internal only; Seedance receives only the
final prompt text.

Each skill must define:

- `id`: stable skill id.
- `strategy`: strategy family.
- `required_slots`: fields that must be available before composition.
- `forbidden_if`: risk/contract conditions that make the skill invalid.
- `failure_tags`: known failure modes.
- `success_stats`: empirical counters by risk profile.

Current first-pass skills:

- `shared/commerce_expression_strategies.md`
- `shared/anti_patterns.md`
- `shared/prompt_block_spec.md`
- `source_scene_extension/hand_pickup_from_source_scene.md`
- `source_scene_extension/hinge_or_fold_adjust_from_source_scene.md`
- `source_scene_extension/static_texture_reveal_from_source_scene.md`
- `source_scene_extension/product_result_scene.md`
- `commerce_scene/source_confirm.md`
- `commerce_scene/material_action_proof.md`
- `commerce_scene/new_scene_result.md`
- `new_scene_text_reconstruction/backpack_side_pocket_commute.md`

`shared/commerce_expression_strategies.md` is the strategy-selection reference
for the LLM/composer. It lists reusable commerce expression structures such as
direct benefit proof, usage result demo, premium texture reveal, comparison,
operation proof, and problem-solution pairing. These are examples and decision
guidance, not product-type if/else templates. Code may validate risk, conflict,
slot coverage, and prompt cleanliness, but creative structure should be chosen
from material understanding, user goal, visual provability, and product risk.
