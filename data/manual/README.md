# Manual labels

Rule-based and reviewed decisions that the pipeline applies. This folder **is tracked in git**, unlike
`data/raw`, `data/interim` and `data/processed`. The rules are documented in `docs/decisions.md`.

## Files you fill in

| File | What to do |
|------|------------|
| `entity_review.csv` | `is_private_individual` = yes/no for person-like entities (Phase 2) |
| `duplicate_review.csv` | `decision` = merge_into_a / merge_into_b / keep_separate (Phase 2) |
| `topic_labels.csv` | `final_label` per topic; the same label on two topics merges them (Phase 3/4) |
