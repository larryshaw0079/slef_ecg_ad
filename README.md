# Arrhythmia Detection with Self-supervised Pre-training

## Requirements

- numpy
- scipy
- matplotlib
- tqdm
- pytorch >= 1.2

## Basic Usage

Pre-process the dataset:
```bash
>> python scripts/preprocess_ecg.py --data <data dir> --dest <dest dir>
```

Start training:
```bash
>> python train.py
```

Start evaluating:
```bash
>> python evaluate.py
```
