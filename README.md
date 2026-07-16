# [ECCV2026] AgentHOI


<!-- Replace the # targets below when the public resources are available. -->
[![Paper](https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red)](https://arxiv.org/pdf/2607.13881)
[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=githubpages&logoColor=blue)](https://ltttpku.github.io/AgentHOI-Page/)


This repository is the official PyTorch implementation of the paper "AgentHOI: Unleashing Multimodal Large Language Models for Training-free HOI Detection in the Wild".

AgentHOI is a training-free, agentic framework for human-object interaction (HOI) detection in the wild. It uses multimodal large language models for interaction reasoning and GroundingDINO for human-object grounding.


## Repository Layout

```text
.
├── hico_pipe/          # HICO-DET AgentHOI pipeline
├── swig_pipe/          # SWIG-HOI AgentHOI pipeline
├── datasets/           # HICO-DET and SWIG-HOI evaluation code
├── utils/              # Post-processing utilities
├── GroundingDINO/      # GroundingDINO source used by the box grounding stage
├── data/
│   ├── hico/           # HICO-DET test annotations
│   └── swig/           # SWIG-HOI test annotations
├── hoi_metirc.py       # HICO-DET evaluation entry
└── swig_metirc.py      # SWIG-HOI evaluation entry
```

## Installation

Create a Python environment, then install the project dependencies and GroundingDINO.

```bash
pip install -r requirements.txt
pip install -e GroundingDINO
```

Download the GroundingDINO Swin-B checkpoint and place it at:

```text
GroundingDINO/weights/groundingdino_swinb_cogcoor.pth
```

If your environment cannot download `bert-base-uncased` from Hugging Face at runtime, download it locally and set:

```bash
export BERT_BASE_UNCASED_PATH=/path/to/bert-base-uncased
```

## Data Preparation

The repository includes the HICO-DET and SWIG-HOI test annotation files used by the evaluation scripts. Dataset images are not included.

Place images at the default paths, or pass custom paths with environment variables:

```text
data/hico_20160224_det/images/test2015/
data/swig/images/
```

```bash
export HICO_IMAGE_DIR=/path/to/hico_20160224_det/images/test2015
export SWIG_IMAGE_DIR=/path/to/swig/images
```

## API Configuration

The pipeline reads OpenAI-compatible API settings from environment variables:

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_API_BASE_URL=https://api.openai.com/v1
export AGENTHOI_MODEL=gpt-4o-2024-11-20
export AGENTHOI_MODEL_NAME=4o
```

For OpenAI-compatible local or third-party endpoints, set `OPENAI_API_BASE_URL` and `AGENTHOI_MODEL` accordingly.

## Precomputed Predictions

We provide the released model predictions used for evaluation:

```text
hico_pipe/output/4o_logit/4o_box.json
swig_pipe/output/swig/4o_box.json
```

## Results

### HICO-DET

| Setting |  Unseen Split mAP |
| --- | ---: |
| Unseen Verb (UV) | 30.12 |
| Unseen Object (UO) | 33.97 |
| Rare First (RF) | 39.16 |
| Non-rare First (NF) | 28.47 |

### SWIG-HOI

| Split | mAP |
| --- | ---: |
| Zero-shot | 11.94 |
| Rare | 13.36 |
| Non-rare | 15.16 |
| Full | 13.43 |

## Run on HICO-DET

```bash
bash hico_pipe/pipe.sh
python hico_pipe/outbox.py
python hoi_metirc.py --dataset_file hico --input_file hico_pipe/output/4o_logit/4o_box.json
```

To use another output directory:

```bash
export HICO_OUTPUT_DIR=hico_pipe/output/my_run
python hoi_metirc.py --dataset_file hico --input_file hico_pipe/output/my_run/4o_box.json
```

## Run on SWIG-HOI

```bash
bash swig_pipe/pipe.sh
python swig_pipe/outbox.py
python swig_metirc.py --dataset_file swig --input_file swig_pipe/output/swig/4o_box.json
```

To use another output directory:

```bash
export SWIG_OUTPUT_DIR=swig_pipe/output/my_run
python swig_metirc.py --dataset_file swig --input_file swig_pipe/output/my_run/4o_box.json
```


## Citation

If you find this project useful, please cite:

```bibtex
@inproceedings{lei2026unleashing,
  title={Unleashing Multimodal Large Language Models for Training-free HOI Detection in the Wild},
  author={Lei, Ting and Liu, Jialin and Xu, Zhu and Peng, Yuxin and Liu, Yang},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```
