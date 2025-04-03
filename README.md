# 🛰️ Satellite Image Segmentation – Training

## 🚀 Quickstart

### 1. Clone & Setup Environment

```bash
git clone https://github.com/InseeFrLab/satellite-images-train.git
cd satellite-images-train
uv sync
uv run pre-commit install
```

### 2. Run Locally

Set the trainning parameters and execute:

```bash
bash bash/mlflow-run.sh
```

It will log the entire run into MLflow.


### 3. Run with Argo Workflow ☁️

1. Update parameters in `argo-workflows/train-workflow.yaml`.
2. Submit via Argo CLI or UI:
```bash
argo submit argo-workflows/train-workflow.yaml
```

## 🧠 Model Configs

The training pipeline is built with PyTorch and designed to be flexible:

- **Architectures**: `deeplabv3`, `segformer-b[0–5]`, `single_class_deeplabv3`
- **Losses**: Cross-Entropy (with options), BCE, BCE with logits
- **Schedulers**: `reduce_on_plateau`, `one_cycle`, etc.
- **Labelers**: `BDTOPO`, `COSIA` (custom labelers can be added — see the [preprocessing repo](https://github.com/inseeFrLab/satellite-images-preprocessing))
- **Bands**: Default is 3, but can be customized


## 📈 MLflow Integration

This project relies on **MLflow** to keep track of everything that matters during training:

- Parameters, metrics, and artifacts are automatically logged thanks to PyTorch Lightning’s built-in support.
- Models are versioned and saved with their full training context (code, config, metrics, etc.).
- You can run experiments using the `MLproject` file so that everything is tracked.

⚠️ Don't forget to set the `MLFLOW_TRACKING_URI` environment variable !


## 📄 License

Distributed under the **MIT License**.
