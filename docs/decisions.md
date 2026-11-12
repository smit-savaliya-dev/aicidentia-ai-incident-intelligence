# Decisions and rules

Every rule applied to the data or the analysis, why it was chosen, and its effect in numbers. Numbers come
from `reports/results.json` (key in brackets) so they can be cited and re-checked. Earlier analytic rules
(model-selection rules, deployment rules) are stated in the RQ documents; the ones that shape the final results
are repeated here so the report can cite one place.

## D1. No manual label audit

**What.** The planned blind audit (150 MIT-labelled incidents hand-annotated by the author) was dropped. The
template, its git-ignored answer key, the annotation app, the label guidelines, the dashboard's
"hide labels while the audit is open" masking, the refresh step that waited for annotations, and their tests were
removed. None of the 150 items had been annotated.

**Why.** A single annotator gives weak evidence: with one person there is no inter-annotator agreement, so a
disagreement with the MIT label cannot be attributed to the label rather than to the annotator's own errors or
reading of the taxonomy. Reliability is instead assessed with independent signals that already exist in AIID
(human CSETv1 and GMF annotations, editor-linked incident pairs, and confident learning on out-of-fold
predictions; see D7-D9).

**Effect.** RQ3's second half is reframed as "How reliable are the AI-generated risk labels, judged against
independent signals already in AIID?". The dashboard shows every MIT label. The small-cell rule for network
communities (domain mix hidden for communities with fewer than 20 incidents) is kept as a general disclosure
rule, not as audit protection.
