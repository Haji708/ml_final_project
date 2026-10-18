# Contribution Report

Keep this file in the repo root. It is checked at the demo. Be honest and
specific — "helped with everything" is not acceptable. One row per member,
plus a short per-member paragraph.

## Team
| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Babayev, Mahammadali | `mahammadalibabayev11` | Data cleaning, preprocessing, Task A regression, shared pipeline and integration tests |
| Hajiyev, Ilham | `Haji708` | From-scratch decision tree, tree tuning, feature importance and tree experiments |
| Javadzade, Mahammad | `MahammadMaga` | From-scratch Pegasos SVM, RFF extension, SVM tests, diagnostics and convergence outputs |
| Isgandarli, Fariz | `Fariz-Isg` | EDA, classification baselines, evaluation, error analysis and project integration |

## Per-member summary
**Babayev, Mahammadali —** Implemented and maintained the shared data
preparation and regression workflow in `src/data_prep.py`,
`src/member1_regression.py`, and related integration tests. This included
cleaning, leakage controls, deterministic splitting, preprocessing, and
scratch/sklearn Ridge comparisons. The contribution is also represented by
the integration and validation work in commit `fc2b594`.

**Hajiyev, Ilham —** Implemented the NumPy decision tree in
`src/decision_tree.py`, including threshold search, Gini/entropy/MSE
criteria, recursive growth, prediction, and tree diagnostics. Added tree
experiments, feature-importance reporting, and presentation integration.
Representative commits include `c22c327`, `238dbb8`, `0c80b46`, and
`22301eb`.

**Javadzade, Mahammad —** Implemented and tested the from-scratch Pegasos
SVM in `src/svm.py`, including binary-label mapping, objective tracking,
margin width, support-vector extraction, feature scaling integration, and
the RFF extension. Added `tests/test_svm.py` and integrated validation
selection, diagnostics, convergence outputs, and runner fixes in
`src/run_all.py`; representative commits are `185a433`, `0a001ee`,
`e30747c`, `4725756`, and `b3cfe04`.

**Isgandarli, Fariz —** Contributed to exploratory analysis, classification
baselines, evaluation metrics, error-analysis figures, and overall
experiment integration. These contributions are reflected in the shared
runner, report figures, and presentation sections covering EDA, metrics,
feature importance, and model comparison.
## Rough effort split
These are approximate estimates based on the division of implementation,
experimentation, integration, and documentation work. The split is not
perfectly equal because the SVM and tree components required different
amounts of debugging and integration.

| Member | Approx. % |
|--------|-----------|
| Babayev, Mahhamadali | 25% |
| Hajiyev, Ilham | 25% |
| Javadzade, Mahammad | 25% |
| Isgandarli, Fariz | 25% |
