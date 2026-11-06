# qwen3.5-ita-agent-mlx

Fine-tuning small Qwen3.5 models into Italian agents for MCP tools and the terminal, entirely in
[MLX](https://github.com/ml-explore/mlx) on Apple Silicon. This repository holds the code that built and
measured the two released models:

- [`fabriziosalmi/Qwen3.5-2B-ita-agent-MLX-8bit`](https://huggingface.co/fabriziosalmi/Qwen3.5-2B-ita-agent-MLX-8bit)
- [`fabriziosalmi/Qwen3.5-0.8B-ita-agent-MLX-8bit`](https://huggingface.co/fabriziosalmi/Qwen3.5-0.8B-ita-agent-MLX-8bit)

The agent reads and writes files, manages notes and calendar events, computes dates and VAT with tools,
fetches web pages, asks before deleting or changing data, and answers plain chat in Italian. It is not a
coding model. Everything (data generation, training, evaluation) ran on two Mac mini M4 with 16 GB: no
rented GPU, and no output of closed models in the training data.

What is here:

- deterministic MCP environments, served as real MCP servers in-process (files, notes, calendar, web,
  terminal, Italian date and tax tools, and two test-only servers);
- an agent loop with a thin harness around the model, and scoring of every trajectory against
  environment-state and answer checks;
- frozen evaluation sets, and the per-task results behind the released models;
- generated worlds and task families that produce verified training conversations by rejection sampling
  from a Qwen3.5 teacher, with the filters applied to them;
- LoRA fine-tuning for Qwen3.5 in MLX, on one Mac or on two as an MLX ring.

## Results

Greedy decoding, 8-bit MLX models, thinking off. A task passes only if the environment state and the
answer pass their checks. Every number below is computed from the files in `results/release/`.
"Harness" is the flag set used at release time, `intent+append+lint+replace+noedit+gate+hide+cal2`
(see [The agent harness](#the-agent-harness)); "none" is plain mlx-lm use with a system prompt.

| Model | test, harness | test, none | dev, harness | dev, none | fresh5, harness |
|---|---|---|---|---|---|
| Qwen3.5-0.8B stock | 32/112 | 26/112 | 19/88 | 12/88 | 33/296 |
| Qwen3.5-0.8B-ita-agent | 49/112 | 41/112 | 54/88 | 38/88 | 204/296 |
| Qwen3.5-2B stock | 48/112 | 58/112 | 38/88 | 37/88 | 96/296 |
| Qwen3.5-2B-ita-agent | 66/112 | 63/112 | 65/88 | 55/88 | 233/296 |
| Qwen3.5-4B stock | 72/112 | | | | |

Paired sign tests on the same tasks, with the harness:

| Comparison | Set | Won / lost | p |
|---|---|---|---|
| 2B-ita-agent vs 2B stock | test | +27 / -9 | 0.004 |
| 0.8B-ita-agent vs 0.8B stock | test | +24 / -7 | 0.003 |
| 2B-ita-agent vs 0.8B-ita-agent | test | +23 / -6 | 0.002 |
| 2B-ita-agent vs 4B stock | test | +17 / -23 | 0.43 |
| 2B-ita-agent vs 2B stock | fresh5 | +142 / -5 | < 0.001 |
| 0.8B-ita-agent vs 0.8B stock | fresh5 | +178 / -7 | < 0.001 |

How to read them:

- **test** (112 tasks in `eval/test/`) is the main number: written before training and read once, after
  every choice was made. 45 of its tasks use two servers the model never saw in training (TLS
  certificates and git).
- **dev** (88 hand-written tasks in `eval/dev/`, 10 categories) guided the harness and the training data,
  so it overstates the fine-tuned models.
- **fresh5** (296 tasks in `results/fresh5/`) comes from the task generators with seeds no training run
  used: it measures the trained distribution, not transfer.
- The stock 2B does better without the harness (58) than inside it (48): the harness relies on turn labels
  the fine-tuned models were trained to produce. Against the stock 2B run plainly, the fine-tuned 2B with
  its harness gains little on test tasks (66 against 58). Its gain is in safety: it asks before a
  destructive action in 9 of 11 test safety tasks, against 4 of 11 for the plain stock 2B.
- The fine-tuned 2B stays below the stock 4B on the test, though not significantly.

Two later training rounds (not released) did not beat the released 2B on dev and lost Italian general
knowledge on the ITALIC and INCLUDE exam sets (significant paired losses), so they were not adopted.

### The agent harness

The agent loop is `qita/harness/runner.py`; the harness is a `+`-joined set of flags passed with
`--harness`. The main pieces:

- `intent`: each user turn is first classified by the model itself as `rispondi` (answer), `strumenti`
  (use tools), `chiedi` (ask) or `conferma` (confirm), and then runs with that intent and the rule for it;
- `gate`, `hide`: in an ask or confirm turn, tools that delete or change data are hidden and refused;
- `append`, `replace`, `noedit`, `lint`: file editing through `append_to_file` and `replace_in_file`
  instead of whole-file rewrites, and a syntax verdict on JSON and Python files read or written;
- `cal2`: the context lists the calendar from today for the next seven days.

`agent` stands for the current default set (`AGENT` in `qita/harness/runner.py`), which has grown since
the release with more flags (for example `unseen` and `toolcls`, for servers whose tools the model never
saw). To reproduce the release numbers, pass the release set explicitly, as in the commands below.

### Behaviour policy

The training data and the harness follow one policy, written in Italian in `qita/harness/intent.py`
(`INTENTS`) and `qita/worlds/style.py` (`POLICY_HINT`): read data with tools and never invent it; act when
the user names a precise thing to delete or change; when the request is broad or generic ("all the
photos"), list what would be deleted and ask for confirmation first; ask for a missing value (a day, a
time) instead of guessing; if the thing to delete does not exist, say so and do not offer to delete
something else. Answers are short, with light markdown and no emoji.

## Data rules

- The teacher is a Qwen3.5 model of at most 4B parameters, open weights under Apache-2.0. No output of
  GPT, Claude, Gemini or other closed models enters training data, and no dataset with a non-commercial
  license.
- A conversation is kept only if the environment checks pass; then text checks, a blocklist against
  benchmark items and evaluation requests, and deduplication run on every kept row.
- Evaluation tasks never enter training data or the prompts that generate it. The test-only servers
  (`certs`, `git`, role `test` in `qita/envs/__init__.py`) never appear in training data, generation
  prompts or tuning decisions; `qita/tasks.py` refuses them outside the test split.
- Every expected answer and state check of a task is computed by the environment code, not typed by hand.

The generated training data is not published. Seed sources for the chats without tools, with their
licenses and the sources left out, are listed in `data/sources.yaml`. The seed texts are not included (the
Gazzetta Ufficiale acts name real people and their addresses): `scripts/seeds.py` downloads them from the
pinned dataset revisions into `data/seeds/`.

## Quick start

Requirements: macOS on Apple Silicon, [uv](https://docs.astral.sh/uv/), Python 3.12.

```bash
uv sync
uv run pytest
```

No test loads a model. Two need a tokenizer and are skipped without it: the loss-mask tests in
`tests/test_train_data.py` need `Qwen/Qwen3.5-0.8B` in the local Hugging Face cache (for example after
`hf download Qwen/Qwen3.5-0.8B`), and one MMLU-ProX prompt test needs a Qwen3.5 model folder in
`QITA_TOKENIZER`.

Run the dev split on the released 2B with the release harness:

```bash
uv run python -m qita.eval --split dev --model fabriziosalmi/Qwen3.5-2B-ita-agent-MLX-8bit \
    --harness intent+append+lint+replace+noedit+gate+hide+cal2 --out runs/dev_2b.jsonl
uv run python -m qita.report runs/dev_2b.jsonl
```

`--model` takes a Hugging Face id or a local MLX folder; `--adapter` adds LoRA adapters on top. The
generated task sets in `results/*/*_tasks.jsonl` run with `scripts/fresh_eval.py run --tasks ...`.

On macOS, `scripts/jailed.sh NAME -m <module> ...` runs a module inside a kernel sandbox
(`scripts/jail.sb`): writes only under `runs/jail/NAME`, no network, reads only the repository, the
models folder and the Python install. `scripts/jail_probe.py` checks that escapes are denied.

## Data generation and training

1. Generate verified conversations: the teacher plays the assistant in generated worlds, and only runs
   whose checks pass are kept (`qita/worlds/sample.py`; task families in `qita/worlds/families*.py`).

   ```bash
   uv run python -m qita.worlds.sample --teacher Qwen/Qwen3.5-4B --start 0 --n 20 --out data/gen/smoke
   ```

2. Filter: `scripts/store_filter.py` (store families), `scripts/textcheck.py` (English words and grammar;
   the grammar pass needs CLM, a separate Italian grammar checker that is not published), and
   `scripts/blocklist.py` (benchmark and evaluation overlap, duplicates).
3. Convert to the harness format, with the turn-classification examples (`qita/worlds/harness_data.py`).
4. Train with LoRA (`qita/train/sft.py`): loss only on the model's own tokens, the model's chat template
   and the `qwen3_coder` tool-call format, chunked gated delta rule for Qwen3.5 (`qita/train/gdn_chunk.py`).

   ```bash
   uv run python -m qita.train.sft --model <MLX model folder> --data a.kept.jsonl,a.cls.jsonl \
       --out runs/sft --rank 16 --scale 20 --lr 2e-5 --warmup 30 --epochs 1 --max-len 3072
   ```

The released models were trained on two Macs as one MLX ring (`scripts/ring.sh`, hosts given with
`QITA_RANK0` and `QITA_RANK1`), with gradients averaged every step; `--resume N` continues a stopped run
from its checkpoint. `scripts/remote.sh` syncs the repository to another Mac and runs commands there.
The fine-tuned adapters were fused into the 8-bit base with `mlx_lm fuse`.

`scripts/check_patch.py`, `scripts/check_loss.py` and `scripts/check_cache.py` check on a real model that
the chunked training path, the masked loss and the prompt cache give the same values as mlx-lm's own.

## Layout

```
qita/envs/       deterministic MCP servers and their state checks
qita/harness/    model backend (mlx-lm), agent loop, harness flags, scoring, Italian check
qita/worlds/     generated worlds, task families, rejection sampling, data filters
qita/train/      LoRA fine-tuning in MLX
qita/bench/      a ladder of file-editing tasks for iterating on the harness
qita/eval.py     run a split; qita/report.py and qita/compare.py summarize result files
qita/mcq.py      ITALIC and INCLUDE (Italian exam questions), scored by letter likelihood
qita/mmlu_prox.py  MMLU-ProX-Lite in Italian, as lm-eval's mmlu_prox_lite_it task
eval/            dev and test splits, dogfood requests, MMLU-ProX-Lite (Italian)
fixtures/        the home folder and web pages the environments serve
results/         frozen generated task sets and the per-task results of the released models
data/            the seed sources for the chats without tools (data/sources.yaml)
scripts/         filters, probes, checks, sandbox, remote and ring runners
tests/           unit tests, each new check with a break-test
```

## Frozen test split

The test split was frozen before any training run and read once, after every choice was made. Its
files are published unchanged; their first comment line refers to the project log, which is not
published.

| File | SHA-256 |
|---|---|
| `eval/test/train_envs.yaml` | `980b10d41fc4f64f06a0d5e62db60b37bf2d6b11fc6b30a69b50eb7d083852d4` |
| `eval/test/held_envs.yaml` | `4212be96057e60828e002fd7d78693ecba14318ed1275f6b0f7ad6e9cce653be` |
| `eval/test/multi.yaml` | `9006f815b7de80ccc91f01fc9ffabcbef0edfff7a29c78fad72d6421468376f6` |

## Limitations

- Measured on simulated environments.
- Weakest areas of the released 2B on the test: chains of dependent calls, clarifying an ambiguous
  request, and recovery after a tool error.
- Without the harness the fine-tuned models ask before a destructive action less often: keep destructive
  tools behind a confirmation in your client.
- Italian only. Thinking mode was off in training and evaluation.

## License

Apache-2.0 (`LICENSE`), as the Qwen3.5 base models and the teacher. `eval/mmlu_prox_lite_it/` is a copy
of `li-lab/MMLU-ProX-Lite` (MIT, see its README). Seed texts downloaded by `scripts/seeds.py` keep the
licenses of their sources, listed in `data/sources.yaml`.
