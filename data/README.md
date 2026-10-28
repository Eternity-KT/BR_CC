# Datasets used by GSI-MLC-PA v6.3.3

This directory contains the raw inputs for the frozen ten-dataset benchmark.
Raw files are never rewritten by the project. Their expected byte sizes and
SHA256 hashes are recorded in `configs/datasets_v6_3_3.json`.
`dataset_manifest.csv` is the compact summary; the validator also checks that
its shapes, statistics, sizes, and source-set checksums match the JSON.

The CSV `source_set_sha256` hashes the UTF-8 concatenation of
`path + "\t" + file_sha256 + "\n"` for every source file, in JSON file order.

Run the complete integrity and loader check from the repository root:

```powershell
python scripts/validate_v6_3_3_datasets.py
```

For a faster structural check without hashing every source file:

```powershell
python scripts/validate_v6_3_3_datasets.py --skip-checksums
```

To check a subset, use canonical names or aliases:

```powershell
python scripts/validate_v6_3_3_datasets.py --datasets CHD_49 genbase
```

Exit code 0 means all selected datasets pass (including documented warnings).
Exit code 1 means a file, checksum, matrix, or manifest check failed. The CSV
summary is checked for the whole collection even when a subset is selected.

## Canonical collection

| Canonical name | Loaded shape `(N, d, K)` | Raw format | Loader rule |
|---|---:|---|---|
| `emotions` | `(593, 72, 6)` | Mulan ARFF + XML | Labels are identified by XML. |
| `scene` | `(2407, 294, 6)` | MEKA ARFF | First six attributes are labels. |
| `yeast` | `(2417, 103, 14)` | MEKA ARFF | First fourteen attributes are labels. |
| `chd49` | `(555, 49, 6)` | MATLAB MAT | `data` is `X`; positive `targets` entries become 1. |
| `music` | `(592, 71, 6)` | MEKA ARFF | First six attributes are labels. |
| `gpositivepseaac` | `(519, 440, 4)` | Sparse text | Official train/test files are concatenated before outer CV. |
| `genbase` | `(662, 1185, 27)` | Mulan ARFF + XML | Labels come from XML; unique identifier `protein` is excluded. |
| `viruspseaac` | `(207, 440, 6)` | Sparse text | Official train/test files are concatenated before outer CV. |
| `plantpseaac` | `(978, 440, 12)` | Sparse text | Official train/test files are concatenated before outer CV. |
| `humanpseaac` | `(3106, 440, 14)` | Sparse text | Official train/test files are concatenated before outer CV. |

`X` is always returned as finite `float32`; `Y` is returned as binary
`int32`. Dataset keys are lowercase. Common aliases such as `CHD_49` and
`HumanPseAAC` are canonicalized by `src.data.loader`.

## Explicit normalization decisions

- No global scaling is performed by the dataset loader. Feature scaling must
  be fitted inside each outer training fold.
- The raw Genbase ARFF contains 1,186 non-label attributes, including a
  nominal `protein` field with one identifier per sample. The final loader
  excludes that identifier and exposes 1,185 predictive motif/profile
  attributes. This avoids converting sample identity into an ordinal numeric
  feature.
  Previous Genbase experiment results that included `protein` must be rerun
  before comparison with this normalized input.
- The PseAAC train/test files are retained byte-for-byte. The benchmark joins
  them and creates fresh multilabel-stratified folds; the original split is not
  used as an external test set.
- Exact feature-and-label duplicates are retained for compatibility. The
  frozen inputs contain seven duplicate rows in `scene`, 455 in `genbase`
  after its unique identifier is removed, and one in `viruspseaac`; the
  validator checks these known counts. A future grouped-split sensitivity
  analysis should measure the effect of the Genbase duplicates.
- `music` and `emotions` share the same six semantic labels and overlapping
  audio feature names. They are related benchmark variants and must not be
  described as independent application domains in statistical claims.

## Sources and citation notes

- `emotions`, `scene`, `yeast`, and `genbase` originate from the Mulan/MEKA
  multi-label benchmark distributions:
  <https://mulan.sourceforge.net/datasets-mlc.html> and
  <https://waikato.github.io/meka/datasets/>.
- `Music.arff` is the MEKA distribution available from
  <https://sourceforge.net/projects/meka/files/Datasets/Music.arff/download>.
- `CHD_49/CHD_49.mat` is accompanied by `CHD_49/readme.txt`, which specifies
  the original authors, citation, and download location.
- Each PseAAC directory includes an original README with its required citation
  and source homepage. The collection cites J. Xu, *Fast multi-label core
  vector machine*, Pattern Recognition 46(3), 2013.

The raw bundles do not provide one uniform redistribution license. Verify the
license and citation requirements of every upstream source before publishing
the dataset files. `dataset_manifest.csv` and the JSON manifest are project
metadata and may be distributed with the code independently of the raw data.
