# NPTEL Assignment Verification Agent

An agent that independently re-solves a submitted NPTEL assignment so a student can check their answers and see the reasoning behind them.

It runs only after submission and never retrieves the student's chosen answer — so the generated solution can't be biased toward or against it. It's a verification, not a confirmation.

## How it works

1. **Extraction** — assignment HTML and course transcript are parsed into a structured JSON (questions, options, passages), with passage-question association determined deterministically from document order, not inferred by an LLM.
2. **Submission gate** — checks a submitted flag before running; stops if the assignment hasn't been submitted yet.
3. **Classification** — a single batched LLM call tags each question as `general`, `numerical`, or `passage`.
4. **Routing** — each question goes to a specialized solver:
   - **General** — retrieves relevant course chunks (RAG), reasons over them.
   - **Numerical** — retrieves the relevant formula, has the LLM construct the expression, hands arithmetic to a Python tool.
   - **Passage** — loads the passage once, groups all its questions into one LLM call.
5. **Output** — each answer comes with its evidence and reasoning, not just a letter.

```json
{
  "question_id": "q1",
  "answer": ["q1_o2"],
  "confidence": 0.91,
  "evidence": [{ "source": "lecture_7", "chunk_id": "lecture7_chunk12", "text": "..." }],
  "reasoning": "..."
}
```

Numerical answers also include:

```json
{ "calculation": { "expression": "(0.12 - 0.04) / 0.08", "result": 1.0 } }
```

## Design principles

- Deterministic code owns document structure and state.
- The LLM owns classification and reasoning, not state.
- Calculations go through a Python tool, not model arithmetic.
- No evidence, no answer — the system flags insufficient grounding instead of guessing.


## Setup

```bash
git clone https://github.com/Nuclear990/NPTEL-Assignment-Verification-Agent.git
cd NPTEL-Assignment-Verification-Agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
