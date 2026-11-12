# Datasheet: AI Incident Database (AIID) snapshot used by AICIDENTIA

_Structure follows Gebru et al., "Datasheets for Datasets". Facts about this snapshot (20261026) are generated from the audit (`reports/results.json`, keys `audit.*`). Statements about how AIID operates beyond what is visible in the files are marked **[AIID documentation, not verified in the snapshot]**._

## 1. Motivation

- **Purpose.** AIID is a public repository of reported harms and near-harms involving AI systems, intended to help practitioners and researchers learn from past failures (McGregor, 2021). AICIDENTIA reuses it to study themes, trends, actors and risk domains.
- **Who made it.** Maintained by the AIID community and editors **[AIID documentation, not verified in the snapshot]**; the snapshot lists editor IDs per incident and a submissions/candidates workflow (collections `submissions`, `candidates`, `quickadd`).
- **Funding.** Not recorded in the files.

## 2. Composition

- **Instances.** 1,775 incidents (Excel export; the snapshot additionally holds 2 merged duplicates), 8,023 reports (812 of them empty stubs), 7,609 entities.
- **Time span.** Incidents dated 1983-09-26 to 2026-10-22 (date the harm occurred); 2026 is partial. Only 97.4% of incidents are dated 2015 or later; the recommended trend window is 2015-2025.
- **Fields.** Identity (title, description, date, alleged deployer/developer/harmed parties, implicated systems, report count) for every incident; three optional taxonomies: MIT (84.3%, automatically generated), CSETv1 (207 incidents, human-annotated), GMF (41 incidents).
- **Text.** Incident title and a short editor-written description for all incidents; full article text for linked reports in the snapshot only.
- **Sampling.** The database is not a random sample of AI harms. It contains incidents someone submitted and an editor accepted, so it is a sample of **reported and editorially accepted** incidents.
- **Missing data.** Taxonomy labels are missing for many incidents (no taxonomy: 279); MIT coverage falls to 88.6% in 2025 and 34.7% in 2026.
- **Confidential or sensitive content.** Reports are public news articles, but incidents involve real people and organisations, including named individuals among affected parties and sometimes minors. Treat all entities as *alleged* parties.

## 3. Collection process

- **Acquisition.** Reports are submitted (by users or via quick-add) and linked to incidents by editors; the snapshot shows the submission, candidate and quick-add collections (quick-add rows: 56, submissions: 59 in this snapshot). The source URL, domain and download date are stored per report.
- **Who is represented.** Reports are overwhelmingly English (95.6%) and from mainstream outlets (top domains: theguardian.com, nytimes.com, washingtonpost.com, reuters.com, theverge.com). Incidents outside English-language news are underrepresented.
- **Labelling.** CSETv1 and GMF are annotated by people (CSETv1 by several annotators with peer review; fields recorded in the snapshot). MIT AI Risk Repository domain/causal labels were **generated automatically** (taxonomy record: `automatedClassifications = true`; description credits Simon Mylius's automatic classification work); the MIT taxonomy itself is described in Slattery et al., arXiv:2408.12622 (cited in the taxonomy record).

## 4. Preprocessing, cleaning and labelling in AICIDENTIA

- Raw files are never modified. Phase 2 produces `data/processed/*.parquet`: entity normalisation via `config/entity_aliases.yaml`, merged-duplicate resolution, stub-report removal, and an entity-type column. The cleaning steps are documented in `docs/data_audit.md` and tested.
- Rule-based decisions on duplicates, person entities and topic names are recorded in `data/manual/` and `docs/decisions.md`.

## 5. Uses

- **Intended in this project.** Topic modelling and trend analysis, network analysis, risk-domain classification and label-reliability assessment, similar-incident retrieval, a read-only dashboard.
- **Out of scope / inappropriate.** Estimating the real-world frequency of AI harm; ranking organisations by culpability; individual-level inference about named people; treating MIT labels as ground truth.
- **Known biases and gaps.**
  - *Reporting bias:* only incidents that reached news media and an editor; English-language and Western outlets dominate (95.6% English).
  - *Growth artefact:* incidents per year rise from 24 (2015) to 463 (2025); this reflects database collection effort as well as events.
  - *Thin evidence per incident:* 40% of incidents have a single report.
  - *Label provenance:* the only full-coverage risk-domain label is machine-generated; human taxonomies cover about 11% (CSETv1) and 2% (GMF).
  - *Entity heterogeneity:* the `deployer`, `developer` and `harmed` fields mix organisations, generic actor classes, technologies and abstract values; the Excel strings are lossy.
  - *Allegation, not finding:* AIID records alleged developers and deployers; presence in the data is not a determination of fault.

## 6. Distribution

- Public at https://incidentdatabase.ai/. The snapshot's `license.txt` states: report contents are subject to their own intellectual property rights; unless otherwise noted, the database is shared under **CC BY-SA 4.0** (https://creativecommons.org/licenses/by-sa/4.0/). Raw data is git-ignored in this repository; derived artefacts that embed AIID text should carry the same attribution and licence terms.

## 7. Maintenance

- AIID is continuously updated, so this analysis is tied to one frozen snapshot (`snapshot_date: 20261026` in `config/config.yaml`), with SHA-256 hashes recorded in `docs/data_manifest.md`. Later AIID edits, merges or relabelling will not appear.

## 8. Citation

- McGregor, S. (2021). Preventing Repeated Real World AI Failures by Cataloging Incidents: The AI Incident Database. *Proceedings of the AAAI Conference on Artificial Intelligence* (IAAI 2021). **Check the volume, pages and DOI against the publisher's record before putting this in the IEEE report.**
- Gebru, T., et al. (2021). Datasheets for Datasets. *Communications of the ACM*. (Format reference.)
- Data: AI Incident Database, snapshot 20261026, https://incidentdatabase.ai/ (CC BY-SA 4.0).
