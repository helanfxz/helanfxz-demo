# Scoring Rubric

Risk scores are 0-5. LLM output may provide candidate observations, but final
scores must be computed by deterministic rules and calibrated with review data.

- `identity_risk`: how strictly the product identity must be preserved.
- `logo_text_risk`: risk from logo, label, packaging text, screen text, or exact marks.
- `motion_structure_risk`: risk from hinge, fold, fixed axis, assembly, wearing, or flexible parts.
- `physics_risk`: risk from hand contact, lifting, insertion, walking, liquid, or multi-object interaction.
- `scene_conflict_risk`: conflict between source image scene and requested shot scene.
- `text_reconstruction_score`: whether product can be reconstructed by text description.
- `anchor_required_score`: whether image anchoring is required.

