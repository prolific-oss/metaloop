# T4: Strategy Switching

Measures whether a language model can detect when its current approach is failing and pivot to a different strategy. Each task prefills the model with work that contains an error (without acknowledging it), then gives the model one opportunity to respond. The model should recognise the error and switch to the correct approach.

## Task types

| Task | Trap approach | Correct pivot |
|------|--------------|---------------|
| **Wordle** | Guess candidate words from a cluster | Guess a non-candidate distinguisher that tests multiple letters |
| **Mastermind** | Guess candidate codes from a cluster | Guess a non-candidate distinguisher code |
| **Dijkstra** | Apply Dijkstra's algorithm (settles nodes before discovering negative edge) | Recognise negative weights invalidate Dijkstra; use Bellman-Ford |
| **Maze** | Wall-following / right-hand rule (loops back to start) | BFS or flood-fill (exit is on disconnected interior island) |
| **Sphere** | Euclidean distance on coordinate pairs | Great circle / Haversine (coordinates are lat/long) |
| **Cipher** | Monoalphabetic frequency analysis (produces gibberish) | Vigenère / Kasiski (cipher is polyalphabetic) |
| **Knights** | Knight/knave 2-role deduction (all assignments contradict) | Introduce spy role (3-role model) |
| **Graph coloring** | Greedy coloring uses K+1 colors | Backtrack and recolor earlier nodes to use at most K |

8 task types, 10 instances each, 80 problems total.

## Prefill design

Each problem includes `prefill_messages`: a conversation where the model has already applied the wrong approach. The prefill contains the error but does **not** acknowledge it. A "Please continue." user message follows the prefilled assistant work.

For wordle and mastermind (multi-turn games), the prefill includes opening guesses with feedback plus one trap candidate guess. For single-turn tasks, the prefill shows the wrong approach's working (e.g., Euclidean distances that don't match references, knight/knave assignments that contradict).

## Task generation

Each task type has a generator that produces 10 instances using `seed=42` for reproducibility. Tasks are generated once; the runner loads the fixed instances shipped in `data/strategy_switching_problems.json`.

## How a task runs

1. Load `prefill_messages` from `tasks.json`
2. Append "Please continue." if the last prefill message is from the assistant
3. Call the model once (single turn, `max_tokens=16384`, thinking budget 8000)
4. Parse the response for a SUBMIT line
5. Send the transcript + correct answer to the LLM judge (Opus 4.6)

Wordle and mastermind get 2 post-prefill turns (enough for distinguisher + answer, not enough to brute-force candidates).

## How a task is scored

**LLM-as-judge (primary).** The judge (Opus 4.6) receives the task context, the correct answer, and the model's post-prefill response. It answers three questions:

- **SUBMITTED**: Did the model output a clear answer?
- **SWITCHED**: Did the model recognise the error and switch strategy?
- **CORRECT**: Is the submitted answer correct?

**Programmatic detection (secondary).** Each task module also has a `detect_switch(result, task)` function that checks the result concretely (e.g., wordle: any guess outside candidate set; dijkstra: declared distance closer to Bellman-Ford than Dijkstra; knights: any person assigned "spy"). Both signals are recorded.

### Outcome categories

| Outcome | Switched? | Correct? | Score |
|---------|-----------|----------|-------|
| `SWITCHED_CORRECT` | Yes | Yes | 1.0 |
| `SWITCHED_WRONG` | Yes | No | 0.5 |
| `NO_SWITCH_CORRECT` | No | Yes | 0.25 |
| `PERSEVERATED` | No | No | 0.0 |
| `NO_SUBMISSION` | — | — | 0.0 |

Scoring is based on the LLM judge's assessment.

## Running

```bash
# Single model (raw model name, no alias resolution)
python run_strategy_switching.py claude-haiku-4-5-20251001

# All registered models (parallel across models, 20 workers per model)
python run_strategy_switching.py --all
```

## Output

Results saved to `results/<model_id>.json`. Each entry:

```json
{
  "id": "cipher_3",
  "task_type": "cipher",
  "submitted": true,
  "correct": true,
  "switched": true,
  "switched_programmatic": true,
  "outcome": "SWITCHED_CORRECT",
  "score": 1.0,
  "n_prefill": 2,
  "messages": [...],
  "judge_response": "SUBMITTED: YES — ...\nSWITCHED: YES — ...\nCORRECT: YES — ..."
}
```

## Verification

Three layers of verification:

**1. Answer correctness.** Independently recomputes the correct answer for all 80 problems from raw task data: Bellman-Ford on Dijkstra graphs, Haversine on sphere coordinates, Vigenère round-trip on ciphertexts, BFS path validation on mazes, statement consistency on knights puzzles, coloring validity on graphs, distinguisher membership on wordle/mastermind candidate sets. All 80 verified correct.

**2. Judge grades gold-standard correct responses.** Gold-standard correct responses (with strategy-switch reasoning) were fed to the LLM judge; all 80 received SUBMITTED=YES, SWITCHED=YES, CORRECT=YES.

**3. Judge grades gold-standard flops.** Gold-standard perseveration responses (wrong strategy, wrong answer) were fed to the judge; all 80 received SUBMITTED=YES, SWITCHED=NO, CORRECT=NO.

## Structure

```
t4_strategy_switching/
├── task_spec.md                 # This file
├── run_strategy_switching.py    # Runner: load, dispatch, judge, aggregate
├── judge.py                     # LLM judge (SUBMITTED/SWITCHED/CORRECT)
├── tasks/                       # One module per game type
│   ├── wordle.py
│   ├── mastermind.py
│   ├── dijkstra.py
│   ├── maze.py
│   ├── sphere.py
│   ├── cipher.py
│   ├── knights.py
│   └── graph_coloring.py
├── tests/                       # Pytest suite
└── results/                     # Output from model runs
```

## API settings

All models (including Anthropic and OpenAI) are called via OpenRouter for parity with the other tasks in the project.

| Setting | Value |
|---------|-------|
| Router | OpenRouter (all models) |
| temperature | 0 |
| max_tokens | 16384 |
| system prompt | none |
