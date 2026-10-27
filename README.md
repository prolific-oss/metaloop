# MetaLoop

Code, per-item model outputs and human-baseline results for **MetaLoop: Benchmarking the Full Metacognitive Loop in LLMs** (John Burden, Nora Petrova and Jerome Wynne; NeurIPS 2026 Evaluations and Datasets track).

MetaLoop tests whether a model's confidence drives what it does next. Seven tasks cover monitoring (how well confidence tracks correctness) and control (abstaining, switching strategy, catching and fixing errors), and six human studies on Prolific provide baselines for five of them. This repository holds the code and released results from which every number, table and figure in the paper is recomputed (after fetching the source datasets), and the code to run the benchmark on new models.

## Contents

| Path | What it holds |
|---|---|
| `tasks/<task>/` | Runner, scorer and specification (`task_spec.md`) per task, and per-item results in `results_v2/`. The Source-Monitoring passages and their labelled statements are in `t3_source_monitoring/passages/`, the Strategy-Switching problems in `t4_strategy_switching/generators/` |
| `src/` | Shared model client (`llm.py`), run and scoring helpers, statistics |
| `config/models.yaml` | Every model's API identifier, host and inference settings, and the judge panel |
| `config/paraphrases.yaml` | The prompt wordings of the robustness appendix |
| `data/` | Question-bank manifest (pointers into the source datasets, no item text) and the underspecified Deferral items |
| `results_v2_metrics/` | Per-model metrics and the paper's numbers (`camera_ready_numbers.json`) |
| `human_studies/` | What participants saw: Prolific listings, on-task instructions, study app configurations |
| `paper/` | Scripts that produce the paper's numbers, tables and figures, and their outputs |
| `scripts/` | Question-bank fetch, scoring and metric scripts, the run drivers, `reproduce.sh` |

Task directories keep the numbering used during development:

| Directory | Task in the paper |
|---|---|
| `t2_confidence_wagering` | Wagering |
| `t3_source_monitoring` | Source-Monitoring |
| `t4_strategy_switching` | Strategy-Switching |
| `t5_selective_deferral` | Deferral |
| `t6_error_detection` | Self-Review |
| `t6_control2` | Results of Self-Review Control 2 (a fresh re-attempt of each error); its runner is `t6_error_detection/run_v2_control2.py` |
| `t7_consistency` | Consistency |
| `t8_loop` | The-Loop (`results_v2/`) and the summary-feedback comparison (`results_v2_design_a/`) |

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # add API keys only if you will run models
```

### The question bank

The bank's items come from GPQA Diamond, MMLU, MMLU-Pro, ARC, SimpleQA and BoolQ. We do not redistribute them: `data/qbank_manifest.json` records, for each item, its source dataset, revision and row, and `scripts/fetch_qbank.py` rebuilds the bank from the canonical copies under each dataset's own licence. GPQA Diamond is gated, so first accept its terms at <https://huggingface.co/datasets/Idavidrein/gpqa> and log in with `.venv/bin/hf auth login` (or export `HF_TOKEN`). Without access the script skips GPQA, and `reproduce.sh` stops with a message.

```bash
.venv/bin/python scripts/fetch_qbank.py     # writes data/question_bank_v2.json (gitignored)
```

Every GPQA item's answer options are in a fixed order, balanced across positions; the order is part of the manifest.

## Reproducing the paper

```bash
scripts/reproduce.sh
```

This makes no API calls. It fetches the bank if needed, recomputes every number and table from the per-item scored results and checks them against the shipped values, then redraws the four figures and reruns Self-Review Control 1. The human numbers come from the shipped summary until the participant-level files are added (see Human data). `scripts/reproduce.sh --quick` also runs a small live pilot of six tasks (all but The-Loop) for one model (`MODEL=...`, default `gpt-4.1`), which needs API keys.

The check covers the analysis, from scored records to the paper. Answers can also be re-extracted from the released responses with the scoring scripts, except where a response was removed (see Results format), and new runs go through the same code; neither is part of `reproduce.sh`.

## Running models

```bash
scripts/run_model_v2.sh <model_key>         # every task for one model, then scoring inputs, Control 2, The-Loop
scripts/score_all_v2.sh <model_key> ...     # scoring and metrics -> results_v2_metrics/{qa,review}/<model_key>.json
```

Model keys are the entries of `config/models.yaml`. To add a model, add an entry there with its route (`anthropic`, `openai`, `google`, or OpenRouter by default), API identifier and settings. Runs are resumable: a rerun only repeats records whose status is `error`. Each runner takes `--pilot --limit N` for a small trial run.

Answer extraction uses a regex for each prompt's output format, with a three-judge panel (Claude Sonnet 4.6, GPT-5.4 and Gemini 3.1 Pro) for answers the regex cannot read; Strategy-Switching transcripts are graded by Claude Opus 4.6. The judges' verdicts on the paper's runs are cached (`judge_cache*.jsonl`, keyed by a hash of the judge prompt), so rescoring those runs needs no judge calls for the responses that are released.

Two of the paper's models cannot be relied on to be available. Claude Sonnet 4 was reached through OpenRouter's Amazon Bedrock host because Anthropic's API no longer served it, and as of 19 October 2026 neither Anthropic's API nor OpenRouter serves it, so its outputs cannot be regenerated. Gemini 2.5 Pro was listed for retirement on 20 October 2026. The released outputs are the record for those models.

## Results format

Each `results_v2/<model>.jsonl` has one record per (item, condition, repeat): the model's response text, its status (`ok`, `truncated`, `empty`, `blocked`, `error`), finish reason, the provider that served it and the model version it reported, token usage, and the hash of the model configuration used. `results_v2/scored/<model>.jsonl` adds the extracted answer, correctness, confidence and the extraction method. The-Loop records are whole sessions, with the per-item history.

To keep source-dataset items out of the repository, records do not include prompts, questions or answer options (the runners rebuild prompts from the bank). Model responses are kept as generated, with two exceptions where the text is removed (`null`): every response to a GPQA Diamond item, following the GPQA authors' request that the items not be redistributed in plain text, and any other response that restates its question verbatim. Scored fields are kept for all of them, so all results recompute. Responses can still quote answer options or parts of a question. The configuration hash in each record refers to the configuration file of the full run, which also listed models not reported in the paper; the released `config/models.yaml` holds the paper's models with unchanged settings.

## Human data

The six studies (S1A and S1B Consistency, S2 Self-Review, S3 Wagering, S4 Deferral, S5 Source-Monitoring) are summarised in `results_v2_metrics/camera_ready_numbers.json` (`human`), which holds the per-study numbers reported in the paper. The participant-level responses are not yet included; we plan to add them, de-identified, once participants have been told how the data will be used and given the chance to opt out. `human_studies/README.md` gives the participant-facing instructions of each study.

## Licence and citation

Copyright 2026 Prolific. The code and MetaLoop's own data are released under CC BY 4.0 (`LICENSE`). The source datasets are not included and keep their own licences (GPQA Diamond CC BY 4.0, MMLU and MMLU-Pro MIT, ARC CC BY-SA 4.0, SimpleQA MIT, BoolQ CC BY-SA 3.0), which also apply to any parts of their items quoted in model responses.

```bibtex
@inproceedings{burden2026metaloop,
  title     = {{MetaLoop}: Benchmarking the Full Metacognitive Loop in {LLMs}},
  author    = {Burden, John and Petrova, Nora and Wynne, Jerome},
  booktitle = {Advances in Neural Information Processing Systems (Evaluations and Datasets Track)},
  year      = {2026}
}
```
