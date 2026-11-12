# Manual labels

Rule-based and reviewed decisions that the pipeline applies. This folder **is tracked in git**, unlike
`data/raw`, `data/interim` and `data/processed`. The rules are documented in `docs/decisions.md`.

## Files you fill in

| File | What to do |
|------|------------|
| `entity_review.csv` | `is_named_person` = yes/no for person-like entities, answered by rule D3 (`scripts/apply_rules.py`): every named person is excluded |
| `duplicate_review.csv` | `decision` = merge_into_a / merge_into_b / keep_separate, answered by rule D2 (`scripts/apply_rules.py`) |
| `topic_labels.csv` | `final_label` per topic (D4: proposed from top terms and representative incidents, reviewed by the author); the same label on two topics merges them |
