# Orchestration Patterns — Lobster AI 1.2

Use the local Python engine to offload biological workflows, not to hide a failed
analysis behind plausible prose. Cloud workflows use the supported npm client;
the two clients have different setup and authentication contracts.

## Local Setup and Session Continuity

Assume `lobster` resolves to the Python engine, or replace it with
`uv run python -m lobster` in an environment containing the engine.

```bash
lobster config-test --json
lobster agents list
```

Do not install packages or write provider credentials automatically when these
checks fail. Ask the user to approve setup; see [CLI reference](cli-reference.md).
Keep the same workspace and session on every query and inspection command:

```bash
WORKSPACE="./analysis"
SESSION="study"
lobster query -w "$WORKSPACE" --session-id "$SESSION" --json \
  "Inspect sample metadata and propose a QC plan before making changes"
lobster command "workspace list" -w "$WORKSPACE" --session-id "$SESSION" --json
```

`$WORKSPACE` is the actual root, not the parent of a nested hidden workspace.

## Capture Both Failure Signals

A local JSON response can be present even when the process exits nonzero. Capture
that response without allowing shell `set -e` to discard it:

```bash
status=0
RESULT=$(lobster query -w "$WORKSPACE" --session-id "$SESSION" --json \
  "Run the approved QC workflow" 2>"$WORKSPACE/query.stderr.log") || status=$?

if [ "$status" -ne 0 ] || ! printf '%s\n' "$RESULT" | jq -e '.success == true' >/dev/null; then
  printf '%s\n' "$RESULT" | jq -r '.error // .response // "No structured result"'
  # Keep stderr for private diagnosis; review/redact it before sharing.
  exit 1
fi
printf '%s\n' "$RESULT" | jq -r '.response'
```

The workspace must exist before redirecting stderr there. Do not infer success
from parseable JSON, reassuring prose or exit status alone. Missing/invalid JSON
is a failure to obtain a structured result, not permission to fabricate one.

## Inspect Without Inference

```bash
lobster command data -w "$WORKSPACE" --session-id "$SESSION" --json
lobster command modalities -w "$WORKSPACE" --session-id "$SESSION" --json
lobster command files -w "$WORKSPACE" --session-id "$SESSION" --json
lobster command plots -w "$WORKSPACE" --session-id "$SESSION" --json
lobster command "queue list" -w "$WORKSPACE" --session-id "$SESSION" --json
```

These are supported local commands without an LLM call. A successful inspection
command can still report an empty workspace. An inventory-read error is not an
empty or complete inventory. Inspect the returned tables, warnings and messages
before scheduling an analysis.

## Count-Safe Bulk RNA Workflow

1. Discover a dataset; inspect its metadata and data representations before
   downloading. Do not assume an accession contains a raw count matrix.
2. Request GEO download through
   `execute_download_from_queue(..., data_requirement="gene_counts")`. This
   tool option requires a suitable complete series-level bulk gene-count matrix
   with exact sample identity; it is not a CLI flag.
3. Read any `Requirement not met` refusal and candidate reasons. Ambiguity,
   missing counts, sample mismatch or changed numeric values require a new choice,
   not repeated unchanged downloads or substituting TPM/CPM.
4. Confirm biological replicate metadata and the contrast before DESeq2.
5. Preserve raw counts through QC/aggregation. Valid `layers['counts']` is
   preferred; absent that layer, `X` must pass count validation. An invalid counts
   layer is refused. Do not round normalized values to circumvent the check.
6. Inspect returned tables, count-source report and paths. Numeric whole-number
   values alone are not proof that an upstream pipeline produced raw counts.

Example requests, after choosing appropriate data and receiving approval:

```bash
lobster query -w "$WORKSPACE" --session-id "$SESSION" --json \
  "Download the selected GEO series with data_requirement='gene_counts'; report any refusal"
lobster query -w "$WORKSPACE" --session-id "$SESSION" --json \
  "Assess sample QC and raw-count availability; show replicate and condition metadata"
lobster query -w "$WORKSPACE" --session-id "$SESSION" --json \
  "Run the agreed DESeq2 contrast using raw counts, then export the result table"
```

A natural-language request does not itself establish that the expected tool ran.
Check the response and resulting data/files before proceeding.

## Figure Workflow — Local Engine

Use retained analysis results to draw; do not recompute DE, clustering or chemistry
just to change a figure. Ready-made templates return editable Plotly figures and
never implicitly save them. Run `assess_data_quality` before `create_qc_plots` and
draw the full QC-assessed modality; drawing does not repair missing metrics or
accept DE/small-gene subsets as QC inputs.

A custom-code figure is retained only by explicit `submit_figure`. Creating
`fig`, calling a template or using Plotly display is not auto-capture. Check the
submission/save outcome, then inspect `lobster command plots --json` with the same
workspace/session. Preview-only results are not saved revisions.

For an edit request, use the exact saved `<figure_id>@<revision_id>`:

```python
# Inside execute_custom_code; replace the placeholder with a literal saved ref.
fig = load_figure("<figure_id>@<revision_id>")
fig.update_layout(title="Updated title")
submit_figure(fig, parent="<figure_id>@<revision_id>", intent="Change the title")
```

Filtering/restyling an existing figure is still an edit. Preserve its original
revision and check that the reply reports a newly saved revision. To retain
regeneration data for a new figure, supply `source` explicitly. Omitted config on
an edit inherits the parent's config; explicit `{}` clears it.

Interactive `/plots` and `/plot <identifier>` provide saved listing/opening.
Template discovery occurs through `list_templates()`, `get_template(name)` and
`show_template_source(name)` inside custom code, not an invented slash command.
These local figure behaviors do not assert Cloud feature availability.

## Outputs and Notebook Export

```bash
lobster command files -w "$WORKSPACE" --session-id "$SESSION" --json
lobster command "pipeline export" -w "$WORKSPACE" --session-id "$SESSION" --json
find "$WORKSPACE/exports" -type f
```

Default/relative DE and annotation outputs use the workspace `exports/` directory.
Read returned paths for notebooks and other outputs; do not assume a root glob.
Default notebook export curates recorded steps; replay mode selects those marked
replayable and can refuse failed or unreplayable data-changing steps. Custom code
using attempt-only figure helpers is excluded from executable export. Inspect the
omission summary and any record-only steps. Export alone does not mean the notebook
was run or its scientific outputs checked.

## Cost, Privacy and Recovery

Bedrock cache points are opt-in (`LOBSTER_BEDROCK_PROMPT_CACHE=1`), intended for
direct own-AWS Anthropic inference, and off by default. No hit-rate or saving
follows merely from enabling them. The local price snapshot is offline list-price
data; exact provider/model keys and unknown rates matter.

`/tokens` shows aggregate and per-agent values, not completeness or pricing
snapshots. Detailed callback summaries expose `cost_complete`, `usage_complete`,
`unpriced_invocations`, `unpriced_models`, and `attempt_coverage="callback_only"`.
If pricing is incomplete, the priced sum is a lower bound; withhold derived cost
metrics. Check usage completeness separately. Callback-only coverage is not a
complete provider bill. See [cli-reference.md](cli-reference.md#code-tool-and-token-evidence)
for code access and detailed configuration; execution runs as Lobster's own OS user.

Do not recover a terminal inference refusal by silently switching credentials,
providers or authority. Auth/policy/budget/context/accounting failures need an
authorized resolution. Preserve diagnostic logs privately; fixed-category
provider refusals are not a guarantee that every log is safe to share.

## npm Cloud Orchestration

Use npm `@omicsos/lobster`, not Python/Go Cloud compatibility:

```bash
lobster cloud login
lobster cloud status
lobster cloud query --json "Describe the available session data"
lobster cloud query --json --session-id latest "Continue the approved workflow"
lobster cloud chat
```

For headless API-key access, supply `OMICSOS_API_KEY` from an approved secret
source. Browser/device login requires the user's participation. Preserve returned
session UUIDs; prefer an explicit UUID over `latest` when coordinating multiple
sessions. Check JSON `success` and error/finish information as well as process
status. The npm query path is programmatic, not interactive-only. Do not use a
stored credential's existence or a grep for account text as an authentication test.
