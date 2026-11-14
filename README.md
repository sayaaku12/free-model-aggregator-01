# Free Model Aggregator — make.fun bounty entry (free-model-aggregator-01)

One server (`main.py`, Python 3.13 standard library only) that serves the
OpenAI Chat Completions API and the Anthropic Messages API over ten
simulated free-tier providers, routing each request to the best free model
that meets its requirements and failing over when a provider refuses.

## Run

```sh
python3 main.py --list   # print the catalog document and exit 0
```

In a trial the sandbox sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE` and
`MAKEFUN_TRIAL_SEED`; the server reads the brief, probes each provider's
catalog route, registers its port and catalog with the env, serves the 20
requests, and exits 0 when the env says the workload is done.

## How it routes

- Candidates are the free models meeting the request's requirements
  (tools, vision, JSON mode, minimum context), ranked by the brief's
  quality table, with providers that already spent calls ranked lower on
  ties. JSON-mode OpenAI requests never go to an anthropic-wire provider
  (its wire cannot carry `response_format`, so the echo would lose it).
- On a 402/429 the server retries the best candidate once — but only when
  the next candidate is much worse (a provider call is the tie-break, so
  a 5-milli gap is not worth one) — then fails over to the next candidate.
  A provider that fails twice in a row is demoted to last-resort for the
  rest of the trial. It stops at the first successful provider call.
- All six client×provider wire combinations are translated faithfully
  (OpenAI, Anthropic, Gemini), preserving the echo fields the client
  checks: system prompt, user text, images, tool names, `max_tokens`,
  stop sequences, JSON mode — plus exact usage and stop-reason mapping,
  in JSON and in streaming.

## Measured (real env + real scorer, local)

| Trial | Score | Quality | Served | Wire | Failover | Catalog | Calls |
|---|---|---|---|---|---|---|---|
| practice 0 | 99 | 99 | 100 | 100 | 100 | 100 | 26 |
| practice 1 | 99 | 99 | 100 | 100 | 100 | 100 | 30 |
| practice 2 | 99 | 99 | 100 | 100 | 100 | 100 | 29 |
| fresh seed A | 99 | 99 | 100 | 100 | 100 | 100 | 26 |
| fresh seed B | 99 | 99 | 100 | 100 | 100 | 100 | 26 |
| fresh seed C | 99 | 99 | 100 | 100 | 100 | 100 | 28 |

Quality 99 is the ceiling: the best model (1475) lives on one provider
whose per-trial budget (10–12 calls) cannot cover all 17 requests that
want it. Replay check: running practice trial-0 twice leaves a
byte-identical state export. `--list` output equals the registered
catalog. Wall time per trial ~1–4 s of the 180 s budget.

See `NOTES.md` for the full feasibility notes and the local test rig.
