# Conditional Response Value (CRV)

Main-result reproduction for **Decision-Aware Active Learning for Repeated Annotation with Conditional Response Value**.

CRV selects additional annotations by their expected reduction in final decision loss. This repository includes the core algorithm and prepared numerical data for the five primary evaluations in the paper.

## Run

Use Python 3.12 and a CPU. No GPU or additional data download is needed.

```bash
git clone https://github.com/zjcerwin/CRV.git
cd CRV
python -m pip install -r requirements.txt
python reproduce.py
```

The script fits the response model from the included calibration annotations, calculates acquisition scores, selects responses under the original budgets, and updates the final predictions. It writes `outputs/main_results.csv` and checks the resulting error counts against `expected_results.csv`.

For just the main Wikimedia experiment:

```bash
python reproduce.py --dataset wikitox
```

## Expected terminal error (%)

| Policy | Wikimedia Toxicity | ImageNet-16H | DynaSent R1 | DynaSent R2 | CIFAR-10H |
|---|---:|---:|---:|---:|---:|
| CRV | 5.20 | 33.16 | 2.69 | 1.50 | 1.67 |
| Random | 5.84 | 37.40 | 8.06 | 8.07 | 3.05 |
| Disagreement | 5.67 | 36.34 | 3.45 | 3.00 | 1.67 |
| Current risk | 5.68 | 33.16 | 2.82 | 1.69 | 1.67 |
| Information gain | 5.68 | 33.42 | 3.49 | 2.06 | 1.67 |

Budgets are 5% for Wikimedia and 20% for the other evaluations. ImageNet-16H uses noise level 125. Lower error is better.

This focused release reproduces CRV and the four baselines that share its response model. Feature-aware baselines, synthetic studies, runtime benchmarks, and budget sweeps are outside this quick reproduction.

## Files

- `crv.py`: response-model calibration, CRV and baseline scoring, batch selection, and final prediction.
- `reproduce.py`: runs the five primary evaluations and checks their results.
- `data/*.npz`: calibration responses, evaluation histories, candidate labels, targets, model settings, budgets, and fixed tie orders.
- `expected_results.csv`: reference sample counts, budgets, and error counts.

The prepared arrays preserve the experimental splits and tie orders. They contain numerical annotations rather than the original text or image files. The script recalculates the model and predictions instead of loading saved final scores.

## Original datasets

- [Wikimedia Toxicity](https://doi.org/10.6084/m9.figshare.4563973)
- [ImageNet-16H](https://osf.io/2ntrf/)
- [DynaSent](https://github.com/cgpotts/dynasent)
- [CIFAR-10H](https://github.com/jcpeterson/cifar-10h)

Please cite the original datasets when using their annotations. The data remain subject to the original dataset terms.
