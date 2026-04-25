# DRIFT-Bench

**Decomposing Reasoning Into Failure Types.** A solver-instrumented benchmark for multi-turn constraint reasoning in large language models.

[![Paper](https://img.shields.io/badge/paper-ICLR%202026%20Workshop-b31b1b)](paper.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Z3](https://img.shields.io/badge/verifier-Z3-informational)](https://github.com/Z3Prover/z3)
[![Problems](https://img.shields.io/badge/problems-1%2C020-orange)](data/problems/)
[![Domains](https://img.shields.io/badge/domains-3-orange)](data/problems/README.md)

> Paper: *Residual Drift Dominates Contradiction in Multi-Turn Constraint Reasoning*. Sebastien Kawada, Kaons. Accepted to the ICLR 2026 Workshop on Reasoning and Planning for LLMs. [[PDF]](paper.pdf)

## Contents

1. [TL;DR](#tldr)
2. [Headline results](#headline-results)
3. [Install](#install)
4. [Quickstart](#quickstart)
5. [Repository layout](#repository-layout)
6. [Benchmark structure](#benchmark-structure)
7. [Methods](#methods)
8. [Reproducing the paper](#reproducing-the-paper)
9. [Citation](#citation)
10. [License](#license)

## TL;DR

Standard reasoning benchmarks collapse two fundamentally different failure modes into a single accuracy number.

* **Contradiction.** The maintained state becomes unsatisfiable. Formal methods detect this.
* **Satisfiable drift.** The state stays consistent, but the returned answer violates it. Most systems miss this.

DRIFT separates the two by checking *both* ledger satisfiability and assignment validity at every turn. The headline finding: after MUS-Repair feedback, **98 to 100 percent of residual errors are satisfiable drift**, while contradiction drops to near zero. Models stop contradicting themselves, yet keep forgetting their own commitments.

## Headline results

MUS-Repair across four open-weight models, turn-level accuracy on the 816-problem test split:

| Model        | Accuracy (%) | Drift (% of residual) | UNSAT (%) |
|:-------------|-------------:|----------------------:|----------:|
| Qwen3-8B     | 30.0         | **100.0**             | 0.0       |
| Qwen3-32B    | 38.2         | **98.1**              | 1.9       |
| gpt-oss-20b  | 68.7         | **99.9**              | 0.1       |
| gpt-oss-120b | 62.7         | **99.9**              | 0.1       |

Full per-method tables live in [`docs/paper_tables/`](docs/paper_tables/).

## Install

Python 3.10 or newer. Two runtime dependencies, both pure-Python wheels.

```bash
git clone https://github.com/kaons-research/drift-bench.git
cd drift-bench
pip install -r requirements.txt
```

No GPU, no model weights, no `torch` or `transformers`. The evaluation runner is a thin client against any OpenAI-compatible endpoint (vLLM, LM Studio, the OpenAI API, OpenRouter, llama.cpp's server, and so on).

## Quickstart

Smoke test with six dev problems against a local serving endpoint:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=<any-string-for-vllm>

python -m src.run_experiment \
  --model qwen3-8b \
  --split dev \
  --method mus_repair \
  --max-problems 6 \
  --db-path smoke.db
```

Then analyze:

```bash
python -m src.analyze --db-path smoke.db --split dev
```

Full test split, single method:

```bash
python -m src.run_experiment --model <name> --split test --method mus_repair --db-path results.db --max-tokens 2048
```

Full matrix (Direct, CoT, Ledger, MUS-Repair over all 816 test problems):

```bash
for method in direct cot ledger_only mus_repair; do
  python -m src.run_experiment --model <name> --method $method --split test --db-path results.db
done
```

## Repository layout

```
drift-bench/
├── paper.pdf                       # 18-page ICLR 2026 Workshop paper
├── data/problems/
│   ├── dev/                        # 204 problems, 68 per domain
│   └── test/                       # 816 problems, 272 per domain
├── src/                            # Evaluation code
│   ├── run_experiment.py           # 4-method runner, SQLite logging
│   ├── extraction.py               # Robust JSON and constraint extraction
│   ├── z3_checker.py               # Domain SAT and MUS computation
│   ├── prompts.py                  # System, turn, extraction, repair prompts
│   ├── repair_policy.py            # 7 repair trigger codes and policy logic
│   ├── analyze.py                  # Accuracy and diagnostic reducers
│   └── generate_problems.py        # Z3-validated problem generator
├── examples/transcripts/           # Illustrative per-domain pipeline traces
├── docs/
│   ├── prompts.md                  # Readable dump of every prompt template
│   └── paper_tables/               # Small CSVs reproduced from the paper
├── LICENSE                         # MIT
├── CITATION.cff                    # GitHub citation widget
└── requirements.txt
```

## Benchmark structure

Three domains, 340 problems each (272 test, 68 dev).

| Domain        | Entities | Turn count (min / mean / max) | Constraint vocabulary size |
|:--------------|:--------:|:-----------------------------:|:--------------------------:|
| Seating       | 6 to 8   | 4 / 6.97 / 10                 | 7                          |
| Scheduling    | 5 to 7   | 4 / 7.06 / 10                 | 6                          |
| Logic grid    | 4        | 4 / 6.89 / 10                 | 7                          |

At every turn, the user introduces one to three new constraints and the system must return an assignment consistent with the full cumulative constraint set. Every gold trajectory is Z3-validated to be satisfiable at every turn.

See [`data/problems/README.md`](data/problems/README.md) for the JSON schema and per-domain constraint vocabularies.

## Methods

All four methods share the same base prompt, extraction, and verification pipeline. They differ in what feedback, if any, is fed back to the generator between turns.

| Method         | State tracking | Solver check | Repair loop |
|:---------------|:--------------:|:------------:|:-----------:|
| Direct         | none           | no           | no          |
| Chain-of-Thought | scratchpad   | no           | no          |
| Ledger         | extracted      | no           | no          |
| MUS-Repair     | extracted      | Z3 SAT       | MUS-guided, up to 3 retries |

## Reproducing the paper

Four open-weight models were evaluated. Each ran all four methods over 816 test problems, for 5,672 total mus_repair turns per model.

```bash
for model in qwen3-8b qwen3-32b gpt-oss-20b gpt-oss-120b; do
  python -m src.run_experiment --model $model --split test --db-path results_${model}.db --max-tokens 2048
done

for model in qwen3-8b qwen3-32b gpt-oss-20b gpt-oss-120b; do
  python -m src.analyze --db-path results_${model}.db --split test
done
```

The original run's SQLite databases suffered filesystem corruption and are not redistributed. A fresh run against the same problem set is the recommended reproduction path. The paper's summary CSVs are included under `docs/paper_tables/` for number-to-number comparison.

## Citation

```bibtex
@inproceedings{kawada2026drift,
  title={Residual Drift Dominates Contradiction in Multi-Turn Constraint Reasoning},
  author={Kawada, Sebastien},
  booktitle={ICLR 2026 Workshop on Reasoning and Planning for LLMs},
  year={2026},
  url={https://github.com/kaons-research/drift-bench/blob/main/paper.pdf}
}
```

See [`CITATION.cff`](CITATION.cff) for the GitHub citation widget.

## License

MIT. See [`LICENSE`](LICENSE).

## Contact

Sebastien Kawada. sebastien@kaons.com. Kaons, Los Angeles, United States.
