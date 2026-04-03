# Metrics And `CodeWMScore`

`CodeWMBench` evaluates each watermarking method along five released dimensions:

- detection reliability
- robustness
- utility
- stealth
- generalization

The leaderboard metric is `CodeWMScore`. It is intentionally **not** a plain average: the benchmark first builds a weighted base score and then applies a fail-closed gate so that high false-positive rates or severe functional collapse cannot hide behind a single strong dimension.

## Notation

Let:

- \(x\) be the clean reference program
- \(x^{wm}\) be the watermarked program
- \(x^{atk}\) be an attacked program
- \(\mathrm{tok}(\cdot)\) be the benchmark tokenizer used for lexical overlap
- \(\mathrm{EditRatio}(\cdot,\cdot)\) be the normalized edit-distance similarity used by the implementation
- \(\mathrm{AUROC}_{neg,wm}\) be the AUROC separating negative controls from watermarked positives
- \(\mathrm{FPR}_{neg}\) be the negative-control false-positive rate
- \(\mathrm{SR}_{neg}\) be the negative-control support rate
- \(r_i\) be the per-row watermark-retention score
- \(d_i^{atk} \in \{0,1\}\) be whether the attacked sample is detected
- \(q_i\) be the row-level quality score for the candidate program attached to row \(i\)
- \(s_i\) be the row-level stealth score carried by row \(i\)
- \(v_i^{exec} \in \{0,1\}\) be whether executable semantic validation is available
- \(p_i^{sem} \in \{0,1\}\) be whether an attacked sample is semantically preserving
- \(p_{wm}\) be the watermarked pass-preservation ratio
- \(p_{atk}\) be the attacked pass-preservation ratio

All released scalar scores are clamped to \([0,1]\) before aggregation.

## Primitive Quality And Stealth Terms

The public scorecard exposes `quality_score` and `stealth_score` indirectly through `utility` and `stealth`. Their implementation is:

\[
\mathrm{LexPres}(x, x^{wm}) = \mathrm{Jaccard}\!\left(\mathrm{tok}(x), \mathrm{tok}(x^{wm})\right)
\]

\[
\mathrm{StructSim}(x, x^{wm}) = \mathrm{EditRatio}(x, x^{wm})
\]

\[
q(a,b) = 0.5 \cdot \mathrm{LexPres}(a, b) + 0.5 \cdot \mathrm{StructSim}(a, b)
\]

At runtime, the benchmark uses this quality primitive against the row-local candidate program chosen by the row-assembly and scorecard pipeline:

- for the watermark-time candidate pair attached to a row group, \(q_i = q(x, x^{wm})\)
- for an attacked row, \(q_i = q(x^{wm}, x^{atk})\)

For stealth, the implementation measures both overall edit footprint and line-count disruption:

\[
\mathrm{Footprint}(x, x^{wm}) = 1 - \mathrm{EditRatio}(x, x^{wm})
\]

\[
\mathrm{LineImpact}(x, x^{wm}) =
\frac{\left| \mathrm{Lines}(x^{wm}) - \mathrm{Lines}(x) \right|}{\max(\mathrm{Lines}(x), 1)}
\]

\[
s(x, x^{wm}) = \mathrm{clip}_{[0,1]}\!\left(1 - 0.7 \cdot \mathrm{Footprint}(x, x^{wm}) - 0.3 \cdot \mathrm{LineImpact}(x, x^{wm})\right)
\]

The benchmark treats stealth as a watermark-time property rather than an attack-specific score. It is computed once from the clean reference and the embedded watermark:

\[
s_i = s(x, x^{wm})
\]

and then reused across attacked rows derived from the same watermarked candidate.

The primitive formulas are implemented in:

- [`codewmbench/metrics/quality.py`](../codewmbench/metrics/quality.py)
- [`codewmbench/metrics/stealth.py`](../codewmbench/metrics/stealth.py)

The row-level choice of which candidate pair is compared, and when a watermark-time stealth value is reused across attacked rows, is determined by the benchmark row assembly and scorecard pipeline rather than by those two primitive-formula modules alone.

## Negative-Control Semantics

The benchmark evaluates two negative-control families when applicable:

- `human_reference`
- `clean_generation`

Coverage matters as much as the observed false-positive rate:

- `SR_neg` is the mean observed-coverage rate over all **applicable** negative-control families
- if a slice has applicable negative-control families but **no observed negative rows**, the implementation forces \(\mathrm{FPR}_{neg} = 1\) and leaves \(\mathrm{SR}_{neg} = 0\)
- if no negative-control family is applicable for a slice, the implementation reports \(\mathrm{FPR}_{neg} = 0\) and \(\mathrm{SR}_{neg} = 0\)

This makes unsupported slices fail closed through the gate instead of being treated as clean evidence.

Operationally, the applicability logic is asymmetric:

- `human_reference` is treated as applicable whenever a human reference exists for the slice
- `clean_generation` is only considered applicable for generation-backed settings, using the benchmark metadata heuristics implemented in `scorecard.py`

## Detection Reliability

The benchmark first computes a raw detection term:

\[
\mathrm{Det}_{raw} = 0.6 \cdot \mathrm{AUROC}_{neg,wm} + 0.4 \cdot (1 - \mathrm{FPR}_{neg})
\]

and then discounts it by negative-control support:

\[
\mathrm{Det} = \mathrm{SR}_{neg} \cdot \mathrm{Det}_{raw}
\]

## Robustness

Robustness combines semantic-preserving attack behavior, retention, and attacked-detection rate:

\[
\mathrm{Rob} =
0.4 \cdot \left(\overline{d^{atk}_{sem}} \cdot \overline{v^{exec}}\right)
+ 0.35 \cdot \overline{r}
+ 0.25 \cdot \overline{d^{atk}}
\]

where:

- \(\overline{d^{atk}_{sem}}\) is the mean attacked-detected rate over semantically preserving rows
- \(\overline{v^{exec}}\) is the executed semantic-validation rate
- \(\overline{r}\) is mean watermark retention
- \(\overline{d^{atk}}\) is the overall attacked-detected rate

## Utility

Utility measures whether watermarking preserves executable usefulness:

\[
\mathrm{Util} =
0.35 \cdot p_{wm}
+ 0.25 \cdot p_{atk}
+ 0.20 \cdot \overline{q}
+ 0.20 \cdot \left(\overline{p^{sem}} \cdot \overline{v^{exec}}\right)
\]

where:

- \(p_{wm}\) is watermarked pass-preservation relative to clean execution
- \(p_{atk}\) is attacked pass-preservation relative to clean execution
- \(\overline{q}\) is the mean row-level quality score
- \(\overline{p^{sem}}\) is the semantic-preservation rate

The pass-preservation ratios use a safe ratio:

\[
\mathrm{SafeRatio}(a,b)=
\begin{cases}
0 & \text{if } b \le 0 \text{ and } a \le 0 \\
1 & \text{if } b \le 0 \text{ and } a > 0 \\
\mathrm{clip}_{[0,1]}(a/b) & \text{otherwise}
\end{cases}
\]

This edge-case behavior matters because \(p_{wm}\) participates in both `utility` and the final gate.

For prompt-prefix tasks such as `HumanEval`, `HumanEval+`, `HumanEval-X (py/cpp/java slice)`, and `MBXP-5lang (py/cpp/java slice)`, detection evaluates the completion region while executable validation reconstructs `prompt + completion`.

## Stealth

The released stealth dimension is simply the mean stealth score:

\[
\mathrm{Stealth} = \overline{s}
\]

## Slice Core

For any single slice, the benchmark reports:

\[
\mathrm{SliceCore} =
0.25 \cdot \mathrm{Det}
+ 0.35 \cdot \mathrm{Rob}
+ 0.30 \cdot \mathrm{Util}
+ 0.10 \cdot \mathrm{Stealth}
\]

## Source-Balanced Suite Aggregation

Suite-level scorecards and exported leaderboards do **not** default to row-weighted aggregation. For paper-facing aggregate outputs, the benchmark uses source-balanced aggregation over atomic source groups.

If \(S\) is the set of atomic source groups in the active suite and \(C(s)\) is any slice-local component for source group \(s\), then the released suite component is:

\[
C_{\mathrm{balanced}} = \frac{1}{|S|}\sum_{s \in S} C(s)
\]

The implementation applies this balancing to:

- detection reliability
- robustness
- utility
- stealth
- slice core
- negative-control FPR / support terms
- pass-preservation terms

This prevents larger active source groups or compact-slice row counts from dominating the public suite score.

The implementation also source-balances the supporting fields that feed those published components, including `negative_vs_watermarked_auroc`, `semantic_validation_rate`, and `declared_semantic_validation_rate`.

For suite aggregates, `Base`, `Gate`, and `CodeWMScore` are computed from these already source-balanced component fields; the benchmark does **not** average per-source `Base`, `Gate`, or `CodeWMScore` values. `Generalization` is not source-averaged directly. Instead, the implementation first builds balanced slice-core maps for model/source/task axes and only then derives the available-axis stability terms from those balanced maps.

## Generalization

Generalization is computed from stability across the available slice axes:

- cross-model stability
- cross-source stability
- cross-task stability

For any axis with at least two valid slices:

\[
\mathrm{Stability}_{axis} =
\frac{\min_j \mathrm{SliceCore}_j}{\mathrm{mean}_j \mathrm{SliceCore}_j}
\]

and:

\[
\mathrm{Gen} = \mathrm{mean}\left(\mathrm{Stability}_{axis}\right)
\]

over the **available** axes only.

If no axis remains available after these validity checks, the released `generalization` value falls back to `0.0` rather than `null`.

The implementation uses the following edge-case rules:

- cross-task slices use `task_category` when present
- if `task_category` is missing, the task key falls back to `source_group:difficulty:reference_kind`
- task groups with fewer than `20` rows are folded into `other`
- any axis with fewer than two valid slices is dropped
- any axis whose mean `SliceCore` is non-positive is dropped

The released scorecard records this explicitly through:

- top-level fields:
  - `cross_model_stability`
  - `cross_source_stability`
  - `cross_task_stability`
  - `slice_core_by_model`
  - `slice_core_by_source`
  - `slice_core_by_task`
- `score_coverage` fields:
  - `generalization_axes_used`
  - `generalization_axes_missing`
  - `folded_sparse_task_rows`

## Base Score

The base multi-dimensional score is:

\[
\mathrm{Base} =
0.20 \cdot \mathrm{Det}
+ 0.25 \cdot \mathrm{Rob}
+ 0.25 \cdot \mathrm{Util}
+ 0.10 \cdot \mathrm{Stealth}
+ 0.20 \cdot \mathrm{Gen}
\]

## Gate

The gate suppresses methods whose watermarked execution collapses or whose negative-control behavior is not trustworthy:

\[
\mathrm{Gate} =
\min \left(
1,\;
p_{wm},\;
1 - \mathrm{FPR}_{neg},\;
\mathrm{SR}_{neg}
\right)
\]

At suite level, the gate acts only on the balanced `watermarked_pass_preservation`, `1 - negative_control_fpr`, and `negative_control_support_rate` terms. It does not directly gate on attacked execution or on the full `utility` aggregate.

## Final `CodeWMScore`

\[
\mathrm{CodeWMScore} = 100 \cdot \mathrm{Base} \cdot \mathrm{Gate}
\]

`Base`, `Gate`, and the released submetrics remain on `[0,1]`; `CodeWMScore` is the only public score scaled to `[0,100]`.

The released scorecard also records:

- `base_score`
- `gate`
- `score_version`
- `score_coverage`

so a reviewer can inspect not just the final score, but why it took that value.
