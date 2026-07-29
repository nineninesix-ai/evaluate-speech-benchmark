# speecheval

Evaluation of zero-shot TTS against the Multilingual Speech Benchmark v2.0 —
six language variants, 8,200 utterances, every row pairing a reference clip of
one speaker with a target text that speaker never read.

Two axes are measured the way the benchmark defines them, plus one it
deliberately leaves out:

| axis | what it answers | anchored? |
|---|---|---|
| **Intelligibility** — WER/CER, four recognisers | can a listener recover the words? | yes, against a human recording of the same text |
| **Speaker similarity** — SIM, three encoders | is it the same voice? | yes, against a same-speaker pair and an impostor floor |
| **Naturalness** — DNSMOS, UTMOSv2, NISQA | does it sound like a person? | **no** — reported separately, never mixed in |

## Why the numbers are comparable

The methodology and its reference implementation are
[nineninesix-ai/make-speech-benchmark](https://github.com/nineninesix-ai/make-speech-benchmark),
the pipeline that builds the benchmark and produced its published human anchors.

Everything that decides a value is delegated to that package (`msbench`), pinned
to a revision: preprocessing, text normalisation, error-rate aggregation, the
bootstraps, the recognisers and the speaker encoders. A reimplementation would
put our numbers on a ruler that only looks like the published one.

Verified in this repository:

- all three published SIM anchors for `ky` recompute exactly — `ecapa` 0.5593,
  `wavlm_ft` 0.6069, `wavlm_sv` 0.9214;
- all 8,200 benchmark texts normalise byte-for-byte to the shipped `text_norm`.

What lives here is everything `msbench` has no opinion about: our data layout,
reference voices that are not benchmark speakers, the naturalness axis, and the
reporting that puts our rows next to the published anchors.

## Setup

```bash
make setup          # uv creates .venv and installs everything
make login          # Hugging Face
echo 'ELEVENLABS_KEY=...' >> .env    # or disable the elevenlabs engine
```

`make setup VENV=path` reuses an existing environment. The measurement engine is
installed from a private repository over SSH, so the machine needs a key with
access to it.

## Use

```bash
make validate    # parse the config, resolve every model and path
make discover    # list synthesis subsets and verify they join onto the benchmark
make normalize   # prove our normalisation matches the benchmark's, text by text
make evaluate    # measure
make report      # rebuild every table from cached metrics, no GPU
```

Narrower runs go through the CLI directly:

```bash
speecheval run --subset en-US__cv_voice --stages asr sim
speecheval run --limit 32                  # smoke test, never a result
speecheval report                          # re-aggregate what is already measured
```

Every unit of work — one subset read by one recogniser, one subset embedded by
one encoder — caches its per-utterance output. A rerun recomputes only what the
configuration actually changed.

## Configuration

One file, [config/eval.yaml](config/eval.yaml). Nothing that determines a number
is hardcoded in the package, and the value used is written into every report.

The part worth understanding is `voices`, which says what each reference voice
is:

```yaml
voices:
  cv_voice:  { type: per_row, column: prompt_audio }   # a different benchmark speaker per row
  nurisa_en: { type: fixed_file, path: ref_audio_data/nurisa_en.wav }
```

`per_row` voices are benchmark speakers, so the published anchor and impostor
floor apply directly. `fixed_file` voices are not in the corpus, so both ends of
the SIM scale are measured here instead: the floor from this voice against every
prompt speaker of the language, and the anchor between chunks of the reference
recording — which shares a microphone and a session, and is therefore labelled
optimistic wherever it appears.

Synthesis directories are discovered automatically as `<language>__<voice>`.

## Output

```
results/<run>/
  results.md              every headline with its confidence interval
  csv/*.csv               the same tables, machine-readable
  per_utterance/*.parquet S/D/I, transcripts and similarities per utterance
  summary.json            every number and the configuration that produced it
```

The per-utterance artefacts are the point: any number can be re-sliced,
re-aggregated or re-tested without a GPU and without re-running a recogniser.

## Reading the results

**Compare paired, not side by side.** Two independent confidence intervals throw
away the fact that both systems read the same sentences. `delta vs anchor` in
`results.md` resamples the shared speakers once per replicate and applies that
resample to both sides.

**A bare cosine is unreadable.** Under `wavlm_sv` a SIM of 0.90 can still sit
below the impostor p95 — a stranger would score higher one time in twenty. Read
`sim_norm`, `floor p95` and `below floor p95` together, and prefer `wavlm_ft`
(the seed-tts-eval scale) and `ecapa` (outside the WavLM family, so it sees past
the benchmark's own selection bias).

**Cross-language comparison is not supported.** Anchor-normalised comparison is
valid within a language only; for `ky` the recogniser is a different model
entirely.

**Watch the failure table.** A mean WER hides looping and truncation. The
duration ratio and the catastrophic rate do not.

## Two findings from building this

**Batched speaker embedding is wrong.** Padding reaches the pooling layer: the
same 2 s clip scores cos 0.295 (`wavlm_ft`), 0.339 (`wavlm_sv`) and 0.997
(`ecapa`) against itself embedded alone. `sim.batch` is therefore pinned to 1 and
the config refuses anything else.

**Punctuation normalises to a space, not to nothing.** `pre-instalado` becomes
two words, not one, and that changes `N_ref` — the denominator of every
corpus-level rate. Deleting the character instead produces plausible numbers on a
different scale from the published anchor.
