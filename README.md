# RoBERTa Testing

Original Code at: https://github.com/AmirAbaskohi/SemEval2022-Task6-Sarcasm-Detection.git

## Requirements

1. Anaconda
2. Python 3.7

## Setup

```bash
conda create -n sarc python=3.7
conda activate sarc
pip install -r requirements.txt
```

## How to run

```bash
cd roberta
rm -rf ./cardiffnlp # remove previous configuration
python3 roberta.py
```

## Potential Issues

### 1. Mismatched library versions

If you're getting the error below, `pytorch` might have a mismatching CUDA toolkit version in the environment.

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
```

1. source (https://yjs-program.tistory.com/206) try:

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3.1 -c pytorch
```

2. try this requirements.txt instead

```bash
pip install -r requirements-icons.txt
```

### 2. GPU out of memory

In `roberta/roberta.py` lines `88` and `89`, try changing batch sizes in the parameters to a lower number like below.

```python
per_device_train_batch_size=4,
per_device_eval_batch_size=4
```

If you have more GPU ram, you can increase batch size to `16`, `32`, `64` or higher.

