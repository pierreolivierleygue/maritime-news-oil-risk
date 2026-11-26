# Maritime-risk article annotation

Development v0.2 · 17 November 2026 · Rules, not a validated classifier

## Annotation policy — updated 12 November 2026

The user has chosen to proceed with **AI-reviewed reference labels**. Human review
is optional and is not a gate for annotation, model fitting or evaluation. Keep
`status = ai_annotated`, `annotator_type = ai`, and `human_reviewed = false` accurate;
these fields describe provenance, not a requirement to obtain human review.
The label definitions below use v0.2. This policy supersedes earlier project
wording requiring manually reviewed evaluation ground truth.

Development annotations can inform fitting and rule refinement. For held-out NLP
evaluation, freeze the guide first and assign AI labels without seeing classifier
predictions or financial outcomes. Describe precision/recall/F1 as agreement with
the AI reference labels; they do not independently establish event accuracy.
Retain family-aware chronological separation and the later financial test period.
Exclude a reviewed `null` from fitting/scoring that label only; do not turn it
into 0 or discard its other resolved labels automatically. No positive financial
result or human sign-off is required to continue.

## Purpose

Turn candidate articles into defensible labels before fitting keyword, TF–IDF or sentence-embedding models. An oil-industry article is not automatically maritime-risk news. Label the article's reporting, not your knowledge of subsequent events or what oil prices did afterward.

Use `document_id` and `record_version_hash` to identify the exact text reviewed. Keep annotations separate from the generated corpus, which is rebuilt on import. Do not put licensed article text in public documentation.

## Three primary labels

Use `1` for yes, `0` for no and `null` for unresolved; record a brief explanation for uncertain cases. Threat and disruption can both apply.

| Field | Assign 1 when… | Do not assign 1 merely because… |
|---|---|---|
| `relevant` | The article substantively discusses security risk, interruption, avoidance, or restoration of maritime supply routes in Hormuz or the Red Sea/Bab el-Mandeb corridor. Reporting that a feared interruption has not occurred is also relevant. | It mentions a coastal energy project, general oil prices, or a business named “Red Sea”. |
| `threat` | It reports a current prospective threat to these routes: a closure warning, threatened attacks, or a concrete risk of interruption. | It uses broadly negative language, mentions past conflict, or discusses ordinary commercial uncertainty. |
| `disruption` | It reports an actual current/recent operational interruption: a shipping suspension, forced rerouting, blocked passage, or interruption to a vessel or infrastructure operation linked to the scoped maritime risk. | An attack occurred without reported operational consequences, or prices rose. |

When `relevant = 0`, set the other two labels to 0. For relevant articles, threat and disruption may both be 0—for example a report of restored passage or historical context. Distinguish interruption of a vessel/route from a reduction in aggregate oil supply: the latter requires separate evidence.

## Context fields

- `location`: Hormuz / Red Sea–Bab el-Mandeb / both / uncertain.
- `temporal_focus`: current development / historical background / mixed / uncertain. Old attacks quoted as background do not alone establish a current threat or disruption.
- `deescalation`: 1 / 0 / null for resumed passage, removed restrictions or easing threats. This is initially descriptive, not an extra financial model.
- `rationale`: a short explanation in your own words; identify the supporting paragraph locally when useful.
- `annotator`, `annotated_at`, `guide_version`: record who made the decision, when and under which rules.

Read negations carefully. “No substantial interruption despite tensions” can be relevant risk reporting while `disruption = 0`. A manufacturer's scheduled maintenance shutdown does not establish maritime-route disruption. A pipeline bypass may be relevant if the article substantively explains current route threats; its mere geographic location is insufficient.

## Scope and evidence decisions — v0.2

These are project conventions chosen on development material before modelling,
not claims about whether an event actually happened. Apply each test to the
article's own text; never fill gaps with another article or later event knowledge.

### Security mechanism and geography

- Include attacks, piracy, coercive detention, conflict-related restrictions and
  their effects on scoped maritime supply operations. Include a secondary hazard
  only when the article explicitly links it to conflict/security conditions and
  possible or actual route/port interruption, as with conflict-prevented maintenance
  of a storage tanker. An attack need not establish operational disruption.
- Exclude purely accidental groundings, weather, mechanical failures and routine
  commercial/maintenance problems from the primary security-risk labels. The
  accidental Ever Given and Affinity V Suez cases are therefore irrelevant here,
  even though their blockages were real. They remain in the corpus as negatives.
- Primary geography remains Hormuz and the Red Sea/Bab el-Mandeb corridor. Suez
  alone is not a third scoped chokepoint. An adjacent approach or port qualifies
  only if the text explicitly connects the incident or risk to the scoped passage
  (for example an attack described as near Hormuz, or Gulf of Aden piracy explicitly
  linked to the Bab el-Mandeb/Red Sea shipping corridor).
- “The Gulf”, Iranian territorial waters, a country name, or an incident off Oman
  is insufficient by itself. Do not extend Hormuz to the whole Arabian Sea. A
  Hormuz threat elsewhere in an article cannot supply a missing location for a
  different seizure. Military/civilian status may be disputed; the reported
  route-security link, rather than a conclusively established ship status, governs
  inclusion.
- A substantive claim must explain a route-security risk, operational consequence,
  protective response, or restoration. Its position or word count does not decide
  relevance. Incidental maps, itineraries, historical analogies and a standalone
  background mention in an unrelated story are insufficient. An explicit causal
  explanation that a current conflict could imperil shipping can be substantive.

### Current and recent operations

- Count an interruption that the article explicitly says is ongoing at publication,
  regardless of its start date. “Still held”, continuing blockage, or an active
  effort to release the detained vessel qualifies.
- Also count an interruption explicitly reported as a new/current development or
  as recent. With explicit dates, use the preceding **30 calendar days**, including
  publication day, for a completed event. This is a fixed development convention,
  not a tuned forecasting parameter or an asserted optimal window.
- Undated “today”, “yesterday”, “this/last week”, “recent days/weeks” or a clearly
  current seizure report qualifies as narrative recency. Record that the exact
  event date is unknown; do not invent one. “Recent months” alone does not qualify.
  Explicit dates override loose recency adjectives. Unverified archive publication
  timing remains a separate limitation; do not infer event dates from issue dates.
- “There have been seizures”, “following intermittent detentions” or a dated old
  attack recounted as background is insufficient without an explicit recent or
  continuing operation. A current concern about those events can support threat
  while disruption remains 0.
- A report that a still-detained vessel has now departed describes both the
  continuing-until-release interruption and restoration: disruption=1 and
  deescalation=1. A retrospective reminder of a long-ended detention does not.
  An intended release, permission with continuing detention, or hoped-for reopening
  does not establish resumed passage. Explicit removal of a restriction or current
  easing of risk can support de-escalation without completed physical passage;
  keep that basis distinct in the rationale.

### Negative evidence and uncertainty

- Labels describe **qualifying evidence in this article**. Assign 0 when a required
  scope, location, timing or operational-consequence link is absent. Preserve the
  reason (for example `recency_not_established` or `location_not_linked`) in the
  review record. This is not a claim of verified normal operation or a known event
  location/date. Do not recode every existing null automatically; inspect its cause.
- Reserve null for conflicting or unreadable evidence that prevents a decision
  after applying these rules. Uncertain attribution alone does not force null
  when the article clearly reports an eligible interruption. Preserve attribution
  and uncertainty notes. Irrelevant articles have threat/disruption/deescalation=0
  and null location/temporal context.
- Threat needs a prospective risk, warning or concrete protective response; an
  already-realised seizure does not automatically add threat=1. A recommendation
  or hypothetical rerouting is not actual disruption. Damage, casualties or oil
  leakage alone do not establish interrupted operations without a stated
  consequence such as evacuation, detention, stopped passage or forced diversion.

The v0.2 consistency pass covers all 58 existing development annotations, including
the first specialist batch. Original decisions and pre-revision files are retained;
current labels carry guide version, revision reason and validation hashes. The
decision report is [ANNOTATION_RULE_DECISIONS.md](ANNOTATION_RULE_DECISIONS.md).

## First exercise and evaluation discipline

The original local batch was prepared under v0.1 (historical workflow):

```sh
.venv/bin/python scripts/prepare_annotation_batch.py
```

Read `data/annotations/proquest_dev_001/reading_packet.md` and record reviewed
labels in the adjacent `annotations.jsonl`, or discuss each article in chat before
saving decisions. The manifest records the selected versions and sampling method.
The batch includes one previously inspected development example and nine seeded
draws across 2019–2022 and the two available early-period sources. This deliberate
development selection is not a representative evaluation sample. Labels start as
null with `status = unreviewed`; distinguish this from a reviewed but unresolved
decision. Set status to `reviewed` after human review, recording annotator and time.
For AI-only annotation, use `status = ai_annotated`, `annotator_type = ai`, and
`human_reviewed = false`, with the assistant identity, timestamp, evidence and
rationale. These first-batch labels are development labels. Held-out AI reference
labels can be prepared separately under the policy above. Preserve the original AI decisions separately if a later review
changes them. For irrelevant articles, `location` and `temporal_focus` may be null
(not applicable); explain unresolved primary labels in the rationale instead.
The preparation script refuses to replace an existing batch. Daily corpus imports
do not touch this directory. Keep a private backup of your annotations: this local
directory contains licensed text and is excluded from Git.

1. Review ten articles from the available **2019–2022** records, without checking OVX movements. Use them to refine these rules, including difficult negative examples.
2. Discuss ambiguous decisions and record rule changes before scaling to the planned 100–150 labels. Revisit earlier labels whenever definitions change.
3. Treat this first exercise as development only. The current early-period sample is dominated by trade publications; it is not a representative evaluation set.
4. Once collection coverage is established, reserve a later, representative NLP evaluation sample. Keep duplicated/near-duplicated stories together and freeze the text pipeline before financial evaluation. Do not use 2024–2025 articles to tune rules for the proposed final financial test; examples already inspected from those years must be acknowledged as exploratory exposure when finalising the split.

Start with full-article labels, then fix the exact headline/opening-text input rule before modelling. Audit whether the evidence supporting a label survives that truncation; otherwise classifier errors may reflect missing input rather than poor text representation. Never silently change a label to match a model prediction.

Single-annotator labels and a small corpus remain limitations. Useful outputs include class counts, ambiguous cases and confusion matrices; a high headline accuracy on mostly irrelevant articles would be insufficient.
