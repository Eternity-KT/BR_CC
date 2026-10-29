# GSI-MLC-PA v6.3.3

Multi-label classification with partial abstention. This repository contains
the final source, dataset manifests and saved experimental results for the
standard **v6.3.3 configuration with SAMG disabled**.

Use branch `paper/v6.3.3-final` for the setup and verification workflow below.
The benchmark source is frozen at tag `v6.3.3-final` (commit `0384c39`), and
the completed results at tag `v6.3.3-final-results` (commit `43d033d`). Later
setup scripts and documentation preserve those source files and result bytes.
Earlier v6.3.4/v6.3.5 work and paper drafts are preserved on
`archive/pre-final-v6_3_4-v6_3_5-20261028`.

If starting from the final ZIP, initialize its local Git history as described
in the packaging section before running tests or experiments.

## Set up the final environment

The tested environment is **Windows x64, Python 3.13.14 and PyTorch
2.6.0+cu124**. The full benchmark requires CUDA for its MLP learner; the
reference GPU is an NVIDIA GeForce RTX 4060 Laptop GPU. Logistic and SVM
experiments, artifact verification and the default smoke test can run without
a GPU. Git must be available for source provenance checks.

Run every command below from the repository root in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_final_env.ps1
```

If `python` points to a different version, select the Python 3.13.14 executable:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_final_env.ps1 -Python "C:\path\to\Python313\python.exe"
```

Setup creates `.venv-final`, installs pinned build tools and the 29 package
versions in [requirements-final-env.lock.txt](requirements-final-env.lock.txt),
and runs `pip check`. The dependency closure includes the immutable benchmark
runtime lock, [requirements-final.lock.txt](requirements-final.lock.txt).
Build temporary files and the package cache stay inside `.venv-final` on the
same volume. The CUDA wheel download is approximately 2.5 GB. Commands use
the environment's interpreter directly, so activation is optional.

## Validate the datasets

Raw dataset files must be present under `data/` at the paths listed in
[configs/datasets_v6_3_3.json](configs/datasets_v6_3_3.json). The complete final
ZIP includes them; they are excluded from Git. For a source checkout, obtain
the original bundles using the source and citation notes in
[data/README.md](data/README.md), and preserve their original bytes.

```powershell
.\.venv-final\Scripts\python.exe scripts/validate_v6_3_3_datasets.py
```

The validator checks file sizes, SHA256 hashes, loaded matrices and agreement
between the JSON manifest and [data/dataset_manifest.csv](data/dataset_manifest.csv).
Exit code `0` indicates success, including the documented dataset warnings.

| Dataset key | Samples | Features | Labels |
|---|---:|---:|---:|
| `emotions` | 593 | 72 | 6 |
| `scene` | 2,407 | 294 | 6 |
| `yeast` | 2,417 | 103 | 14 |
| `chd49` | 555 | 49 | 6 |
| `music` | 592 | 71 | 6 |
| `gpositivepseaac` | 519 | 440 | 4 |
| `genbase` | 662 | 1,185 | 27 |
| `viruspseaac` | 207 | 440 | 6 |
| `plantpseaac` | 978 | 440 | 12 |
| `humanpseaac` | 3,106 | 440 | 14 |

Loaders return finite `float32` features and binary `int32` labels. Genbase's
unique `protein` identifier is excluded. PseAAC train/test files are joined
before creating the benchmark folds. Exact duplicate rows are retained:
Scene has 7, Genbase has 455 after identifier removal, and VirusPseAAC has 1.
The folds do not group duplicates. Music and Emotions are related audio
benchmark variants. These choices matter when interpreting generalization
and statistical comparisons; details and upstream citation requirements are
in the dataset README.

## Run the final checks

```powershell
.\.venv-final\Scripts\python.exe scripts/check_final.py
```

This checks the isolated environment and dependency versions, runs the final
regression suite, fits all six models on VirusPseAAC with Logistic on fold 1,
checks checkpoint resume, compares predictions with the frozen reference and
verifies all 900 saved benchmark cells. Smoke outputs go to the ignored
`scratch/final_checks/smoke/` directory.

Include SVM and CUDA MLP in the smoke test for **18 fits**:

```powershell
.\.venv-final\Scripts\python.exe scripts/check_final.py --gpu-smoke
```

Individual stages are available through `--stage environment`,
`--stage tests`, `--stage smoke` and `--stage verify`. Each subprocess has a
600-second deadline, adjustable with `--timeout`. The controller disables
external pytest plugin autoloading and user-site packages for its children.
[pytest-final.ini](pytest-final.ini) selects the v6.3.3 path and its shared
dependencies, including dataset, metric, calibration, peeling and checkpoint
regressions. The legacy scikit-learn MLP test can emit convergence warnings;
the final MLP benchmark uses PyTorch.

## Reproduce experiments

[configs/v6_3_3_final.json](configs/v6_3_3_final.json) defines the frozen
protocol. Binary estimator parameters come from
[configs/experiment.json](configs/experiment.json).

| Setting | Final protocol |
|---|---|
| Models | `BR`, `CC`, `MLC_PA`, `GSI_v6_2_1`, `GSI_v6_3_2`, `GSI_v6_3_3` |
| Base learners | `logistic`, `svm_calibrated`, `mlp` |
| Outer evaluation | 5 shuffled multilabel-stratified folds, split seed 42 |
| Model seeds | `42 + fold - 1`, shared across model families |
| Preprocessing | MaxAbsScaler fitted only on the outer training fold |
| GSI inner folds | 5; peeling, calibration and threshold selection use training data only |
| Rejection cost | `0.30`, linear penalty, abstention encoded as `-1` |
| v6.3.3 variant | Standard; `separation_aware=false` (SAMG off) |
| Coverage target | `0.70` on OOF training predictions; test coverage can be lower |
| Execution | One Torch/BLAS thread, deterministic Torch, TF32 disabled |

All six families share the same binary estimator settings. MLP uses a
per-label 64-unit PyTorch network for every family. SVM uses nested sigmoid
calibration within the current training fold. `GSI_v6_2_1` names the
`GSIMLCPAv6_2Classifier` comparator with decaying peeling.

For a smaller CPU experiment, run all six models on Emotions with Logistic
across five folds (**30 fits**):

```powershell
.\.venv-final\Scripts\python.exe scripts/run_v6_3_3_final.py --datasets emotions --base-learners logistic --output-root scratch/final_checks/emotions
```

Run the complete matrix (**10 datasets × 3 learners × 5 folds × 6 models =
900 fits**) into a separate reproduction directory:

```powershell
.\.venv-final\Scripts\python.exe scripts/run_v6_3_3_final.py --output-root results/v6_3_3_reproductions
```

Keep the source and configuration committed before running. The runner
prints `RUN_DIR=...`; that directory contains the new run. The run hash
captures source bytes, configuration, data manifest, environment and selected
datasets/learners/folds. Repeating the identical command resumes completed
cells after checking prediction hashes. A changed identity creates a different
run directory; a corrupt prediction stops resume. LF/CRLF conversion or a
different platform/GPU can change the run hash.

Use the explicit output roots above to keep the frozen reference intact.
The original runner's default output root is `results/v6_3_3_final`; rerunning
there can refresh manifest timestamps and aggregate files even when fits are
reused.

## Use and verify the saved results

The complete reference is
[results/v6_3_3_final/run_eb0b08cd254f4f0e](results/v6_3_3_final/run_eb0b08cd254f4f0e).
Verification reads saved artifacts and recomputes metrics without refitting:

```powershell
.\.venv-final\Scripts\python.exe scripts/check_final.py --stage verify
```

To verify a reproduction, substitute its actual run directory:

```powershell
.\.venv-final\Scripts\python.exe scripts/check_final.py --stage verify --run-dir "results/v6_3_3_reproductions/run_<hash>"
```

The check covers source consistency, artifact hashes and inventory, fold
indices, prediction validity, recomputed metrics and dataset summaries.
LF/CRLF differences in source files are accepted only after comparison with
the recorded Git commit; other source changes fail. Artifact hashes remain
byte-exact. The original verifier remains available at
[scripts/verify_v6_3_3_results.py](scripts/verify_v6_3_3_results.py).

| Artifact | Contents in the complete reference |
|---|---|
| `detailed_folds.csv` | 900 metric rows, one per dataset/learner/model/fold |
| `dataset_summary.csv` | 180 rows with fold means and sample standard deviations (`ddof=1`) |
| `overall_summary.csv` | 18 learner/model rows, equally weighted dataset means |
| `per_label_folds.csv` | 9,090 rows with support, confusion counts, coverage and F1 |
| `thresholds_audit.csv` | Per-label thresholds and SAMG state |
| `partitions_audit.csv` | Per-label independent/dependent group membership |
| `predictions/*.npz` | 900 files: test indices, truth, probabilities, full/partial predictions and scaler state |
| `cells/*.json` | 900 resumable checkpoints with metrics and captured warnings |
| `fold_indices.json` | Original train/test row indices for every outer fold |
| `run_manifest.json`, `*_snapshot.json` | Source commit, environment, selection and captured input configuration |
| `checksums.sha256` | Artifact integrity inventory |

`Full_Macro_F1` evaluates binary predictions for every position.
`Selective_Macro_F1` evaluates decided positions per label, then averages
across labels; a label with no decided positions receives F1 zero.
`Coverage` is the fraction of decided sample-label positions. Read selective
F1 together with coverage. Fold means are computed within each dataset first;
overall summaries weight datasets equally. The 70% training coverage target
is not a test-set guarantee.

## Package the final project

Create the complete local package, including the 16 raw input files for the
10 final datasets and their upstream README files:

```powershell
.\.venv-final\Scripts\python.exe scripts/package_final.py
```

The output is an unpacked project at `dist/BR_CC-v6.3.3-final/`, a ZIP at
`dist/BR_CC-v6.3.3-final.zip` and its `.zip.sha256` sidecar. The package
contains only the frozen source/configuration, final workflow scripts and
tests, final dataset inputs/metadata, the 900-fit result set and this README.
It omits historical results, experimental reports, paper drafts, reference
PDFs, unrelated datasets, caches, `.git` and the virtual environment.

All 78 source/configuration files listed in the benchmark manifest are kept
byte-for-byte, including earlier modules/configurations captured by that
snapshot. This preserves the existing provenance and verification contract.
`PACKAGE_MANIFEST.json` records the packaging commit, original benchmark
commit, run ID and inventory counts. `PACKAGE_CHECKSUMS.sha256` covers every
other packaged file. ZIP entries use fixed timestamps, a stable order and
fixed compression settings; rebuilding the same committed inputs in the
same environment produces identical ZIP bytes.

Export requires a clean working tree and the original frozen source bytes.
An existing output is preserved; choose `--output-root dist/rebuild` for
another build. To omit raw datasets, use `--without-data` with a separate
output root. Dataset source/citation conditions remain those of the original
bundles. A root `LICENSE` is included if present; this repository currently
does not specify a code license.

Verify a ZIP or an untouched unpacked folder with the standard library only:

```powershell
python scripts/package_final.py --verify dist/BR_CC-v6.3.3-final.zip
```

After extracting the ZIP, open PowerShell inside `BR_CC-v6.3.3-final/`.
Initialize a local repository so the benchmark runner can record provenance
for new runs (Git needs your usual local author configuration):

```powershell
git init -b paper/v6.3.3-final
git add .
git commit -m "Initialize final artifact snapshot"
```

Then follow the environment setup and final check commands above. Existing
benchmark manifests retain their original source commit. New experiments
record the local snapshot commit. Verify the package inventory before setup
or execution, which creates extra local files such as caches and Git metadata.

## Repository entry points

| Path | Purpose |
|---|---|
| [src/models/gsi_v6_3_3.py](src/models/gsi_v6_3_3.py) | Final GSI model |
| [src/models](src/models), [src/selection](src/selection), [src/calibration](src/calibration) | Estimators, peeling/selection and calibration |
| [src/data](src/data) | Dataset loaders and validation |
| [src/evaluation](src/evaluation) | Metric implementations and contracts |
| [scripts/setup_final_env.ps1](scripts/setup_final_env.ps1) | Isolated pinned environment setup |
| [scripts/check_final.py](scripts/check_final.py) | Environment, tests, smoke/resume and artifact checks |
| [scripts/run_v6_3_3_final.py](scripts/run_v6_3_3_final.py) | Frozen benchmark runner |
| [scripts/package_final.py](scripts/package_final.py) | Final package export and inventory verification |
| [tests](tests) | Regression tests selected by `pytest-final.ini` |
| [data/README.md](data/README.md) | Dataset normalization, provenance and citations |

Historical runners and configurations serve earlier experiments. Use the
entry points above for the final v6.3.3 workflow.
