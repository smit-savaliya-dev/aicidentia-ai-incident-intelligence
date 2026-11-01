# Manual labels

Place the 150 hand-labelled incidents here (e.g. `hand_labels_150.csv`). This folder **is tracked in git**,
unlike `data/raw`, `data/interim` and `data/processed`.

Suggested columns: `incident_id`, `hand_label`, `notes`.

## Files you fill in

| File | What to do |
|------|------------|
| `entity_review.csv` | `is_private_individual` = yes/no for person-like entities (Phase 2) |
| `duplicate_review.csv` | `decision` = merge_into_a / merge_into_b / keep_separate (Phase 2) |
| `topic_labels.csv` | `final_label` per topic; the same label on two topics merges them (Phase 3/4) |
| `label_audit_sample.csv` | blind annotation of 150 incidents: `my_label` (one of the 7 MIT domain names, or 1-7), `my_confidence` (1 low, 2 medium, 3 high), `notes` (Phase 6). Do not look up the MIT label first; guidelines are in `docs/label_guidelines.md`. |
