# Uncertainty-aware multimodal hallucination detection in vision-language models

Code and results for the paper *Uncertainty-aware multimodal hallucination detection in vision-language models via logistic evidence fusion* (submitted to the Journal of Universal Computer Science).

Vision-language models (VLMs) sometimes deny objects that are in an image or affirm objects that are not. This repository contains a lightweight detector that flags such answers one at a time. For every yes/no answer, it reads four token-level uncertainty scores from the generator's logits (mean negative log-likelihood, mean entropy, maximum entropy, negated minimum log-probability) and one visual-risk score from a frozen SigLIP image-text model, oriented by the answer the VLM gave. A logistic regression fuses the five signals. Nothing is fine-tuned.

## Main results

All numbers come from the executed notebook in this repository. Folds and confidence intervals are grouped by image.

| Setting | AUROC (95% CI) |
| --- | --- |
| Proposed fusion, Qwen2.5-VL-3B, POPE-COCO (9,000 questions) | 0.815 (0.793–0.835) |
| Best uncertainty signal alone | 0.780 |
| Visual risk alone | 0.723 |
| Discrete semantic entropy | 0.731 |
| SelfCheck-style disagreement | 0.732 |
| P(True) | 0.602 |
| Proposed fusion, SmolVLM-Instruct | 0.768 |
| Proposed fusion, Qwen2.5-VL-7B (4-bit) | 0.818 |
| POPE-trained detector applied to HallusionBench (Qwen2.5-VL-3B) | 0.699 |

## Repository contents

| File | Contents |
| --- | --- |
| `JUCS_hallucination_detection_clean.ipynb` | The complete experiment notebook, with saved outputs |
| `raw_predictions_*.csv` | Answers and uncertainty features for every question, per generator and benchmark |
| `visual_scores_*.csv` | SigLIP scores for POPE-COCO and HallusionBench |
| `external_baselines_qwen.csv` | Semantic entropy, SelfCheck-style disagreement and P(True) scores |
| `pope_features_modelB.csv`, `pope_features_modelC.csv`, `hallusionbench_features_*.csv` | Feature tables used by the detectors |
| `oof_predictions_proposed.csv` | Out-of-fold detector probabilities |
| `ablation_table.csv`, `experiment_A_grouped_by_image.csv` | Detector comparison on POPE-COCO |
| `experiment_B_cross_benchmark.csv` | Transfer to HallusionBench |
| `experiment_C_cross_model.csv` | Transfer between Qwen2.5-VL-3B and SmolVLM-Instruct |
| `experiment_D_larger_generator.csv` | Qwen2.5-VL-7B experiments |
| `significance_*.csv`, `significance_*.json` | Paired bootstrap tests |
| `per_split_metrics.csv`, `error_cases.csv` | Results by POPE split; qualitative error cases |
| `consolidated_benchmark_metrics_summary.csv` | Summary of all experiments |
| `reliability_diagram_proposed.png` | Calibration plot of the proposed detector |
| `config.json`, `run_metadata.json`, `run_state.json` | Configuration, software versions, GPU and timing information |
| `image_manifest.json`, `hallusionbench_manifest.json` | SHA-256 hashes of every image used |

## How to run

1. Open `JUCS_hallucination_detection_clean.ipynb` in Google Colab.
2. Choose **Runtime → Change runtime type → T4 GPU** (or better).
3. Choose **Runtime → Run all** and allow access to Google Drive when asked.

All data, checkpoints and results are saved to `/content/drive/MyDrive/researh/ijucs`. If Colab disconnects, run all cells again: finished stages are read back from Drive, and unfinished ones resume from their last checkpoint. Once all GPU stages are complete, the analysis also runs on a CPU runtime.

To run a quick test instead of the full experiment, set `"PILOT": True` and `"SAMPLE_LIMIT": 300` in the configuration cell.

## Datasets

The notebook downloads everything it needs. No dataset files are stored in this repository.

- **POPE** (Li et al., EMNLP 2023): https://github.com/RUCAIBox/POPE
- **MSCOCO val2014 images**: http://images.cocodataset.org/val2014/
- **HallusionBench** (Guan et al., CVPR 2024): the `image` split of https://huggingface.co/datasets/lmms-lab/HallusionBench, or the authors' release at https://github.com/tianyi-lab/HallusionBench

## Models

- Generators: `Qwen/Qwen2.5-VL-3B-Instruct`, `HuggingFaceTB/SmolVLM-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit NF4 on GPUs with less than 24 GB)
- Visual model: `google/siglip-base-patch16-224`

## Requirements

Python 3.10 or later, `transformers>=4.49`, `torch`, `accelerate`, `bitsandbytes`, `scikit-learn`, `scipy`, `pandas`, `numpy`, `pillow`, `matplotlib`, `tqdm`, `requests`. The first code cell installs whatever is missing in Colab.

## Citation

If you use this code, please cite the paper (details will be added after publication).

## License

Code is released under the MIT License. The datasets and models remain under their own licenses.
