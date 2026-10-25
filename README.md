# SemCIDS
## Semantic Communication-Based Intrusion Detection System

Research code for semantic encoding, multi-device intrusion
detection, and reinforcement learning for offloading and
compression decisions.

## Requirements

Install the Python dependencies:

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

## Dataset

Place CICIoT2023 CSV files in a folder named `Data/` alongside
the Python scripts.

Dataset:
https://www.unb.ca/cic/datasets/iotdataset-2023.html

The loader assigns labels and threat priorities using CSV
filenames. Check `FILE_MAP` in `01_data_loader.py` before
using files with different naming conventions.

The dataset is not included.

## Quick Start

From the directory containing the Python scripts, run:

```bash
python run_all.py
```

When prompted, enter the number of IoT edge devices (2–10).

The runner creates output folders and copies
`04_drl_agent.py` to `drl_agent.py` for imports.

## Pipeline

| Step | Script | Purpose |
|---|---|---|
| 01 | `01_data_loader.py` | Load and preprocess data |
| 02 | `02_semantic_encoder.py` | Train the semantic encoder |
| 03 | `03_environment.py` | Construct the simulation environment |
| 04 | `04_drl_agent.py` | Define reinforcement learning agents |
| 05 | `05_train.py` | Train agents |
| 06 | `06_evaluate.py` | Evaluate methods |
| 07 | `07_plots_main.py` | Generate main plots |
| 08 | `08_plots_extended.py` | Generate additional plots |
| 09 | `09_fnr_analysis.py` | Analyze false negatives and false positives |
| 10 | `10_weight_reward_analysis.py` | Analyze reward weights |
| 11 | `11_bandwidth_experiment.py` | Run bandwidth experiments |
| 12 | `12_traffic_mix_experiment.py` | Run traffic mix experiments |
| 13 | `13_encoder_comparison.py` | Compare encoders |

## Run Options

```bash
# Restart from training
python run_all.py --from 05

# Run evaluation only
python run_all.py --only 06

# Run encoder comparison only
python run_all.py --only 13
```

Later steps require outputs from earlier steps.

## Outputs

| Folder | Contents |
|---|---|
| `results/` | Processed arrays, configurations, and CSV results |
| `checkpoints/` | Saved model weights |
| `Images/` | Generated plots |

## Validation Status

The Python files passed syntax checks.
The complete training and evaluation pipeline has not been
executed as part of preparing this README.
