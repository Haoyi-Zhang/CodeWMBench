# Metrics And `CodeWMScore`

`CodeWMBench` evaluates each method along five dimensions:

- detection reliability
- robustness
- utility
- stealth
- generalization

The overall ranking metric is `CodeWMScore`. It is a gated weighted aggregate: the base score rewards strong multi-dimensional performance, while the gate suppresses methods with high false-positive rates or severe quality collapse.

## Notation

Let:

- \( \mathrm{AUROC}_{neg,wm} \) be the AUROC separating negative controls from watermarked positives
- \( \mathrm{FPR}_{neg} \) be the negative-control false-positive rate
- \( \mathrm{SR}_{neg} \) be the negative-control support rate
- \( r_i \) be the per-row watermark retention
- \( d_i^{atk} \in \{0,1\} \) be whether the attacked sample is detected
- \( q_i \) be the quality score
- \( s_i \) be the stealth score
- \( v_i^{exec} \in \{0,1\} \) be whether executable semantic validation is available
- \( p_i^{sem} \in \{0,1\} \) be whether the attacked sample is semantically preserving
- \( p_{wm} \) be the watermarked pass-preservation ratio
- \( p_{atk} \) be the attacked pass-preservation ratio

All scalar scores are clamped to \([0, 1]\) before aggregation.

## Detection Reliability

The benchmark first computes a raw detection term:

\[
\mathrm{Det}_{raw} = 0.6 \cdot \mathrm{AUROC}_{neg,wm} + 0.4 \cdot (1 - \mathrm{FPR}_{neg})
\]

and then multiplies it by negative-control support:

\[
\mathrm{Det} = \mathrm{SR}_{neg} \cdot \mathrm{Det}_{raw}
\]

This makes sparse or missing negative-control coverage visible in the final score instead of silently treating unsupported cases as valid evidence.

## Robustness

Robustness combines three attack-time signals:

\[
\mathrm{Rob} =
0.4 \cdot \left(\overline{d^{atk}_{sem}} \cdot \overline{v^{exec}}\right)
+ 0.35 \cdot \overline{r}
+ 0.25 \cdot \overline{d^{atk}}
\]

where:

- \( \overline{d^{atk}_{sem}} \) is the mean attacked-detected rate over semantically preserving rows
- \( \overline{v^{exec}} \) is the executed validation availability rate
- \( \overline{r} \) is mean watermark retention
- \( \overline{d^{atk}} \) is the overall attacked-detected rate

## Utility

Utility measures whether a watermarking method preserves executable usefulness:

\[
\mathrm{Util} =
0.35 \cdot p_{wm}
+ 0.25 \cdot p_{atk}
+ 0.20 \cdot \overline{q}
+ 0.20 \cdot \left(\overline{p^{sem}} \cdot \overline{v^{exec}}\right)
\]

where:

- \( p_{wm} \) is watermarked pass-preservation relative to clean execution
- \( p_{atk} \) is attacked pass-preservation relative to clean execution
- \( \overline{q} \) is mean quality score
- \( \overline{p^{sem}} \) is the semantic-preservation rate

For prompt-prefix tasks such as `HumanEval`, `HumanEval+`, `HumanEval-X`, and the aligned `MBXP` slice, detection evaluates the completion region while executable validation reconstructs `prompt + completion`.

## Stealth

Stealth is the mean stealth score:

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

## Generalization

Generalization is computed from stability across available axes:

- cross-model stability
- cross-source stability
- cross-task stability

For any axis with at least two slices:

\[
\mathrm{Stability}_{axis} =
\frac{\min_j \mathrm{SliceCore}_j}{\mathrm{mean}_j \mathrm{SliceCore}_j}
\]

and the final generalization score is:

\[
\mathrm{Gen} = \mathrm{mean}(\mathrm{Stability}_{axis})
\]

over the available axes only.

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

The gate is designed to fail-closed when a method is unusable or over-detects:

\[
\mathrm{Gate} =
\min \left(
1,\;
p_{wm},\;
1 - \mathrm{FPR}_{neg},\;
\mathrm{SR}_{neg}
\right)
\]

## Final `CodeWMScore`

The final leaderboard score is:

\[
\mathrm{CodeWMScore} = 100 \cdot \mathrm{Base} \cdot \mathrm{Gate}
\]

This means a method cannot achieve a strong overall score by excelling in a single dimension while failing badly on false positives or executable preservation.
