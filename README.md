# Self-supervised Pre-training with Various Down-stream Tasks for Bio-signals

## Introduction



| Task / Method                               | Relative Position | Temporal Shuffling | Transformation Distinguishing | CPC  | SimCLR |
| ------------------------------------------- | ----------------- | ------------------ | ----------------------------- | ---- | ------ |
| Motor Imagery (BCICIV 2a, BCICIII 3a, HaLT) |                   |                    |                               |      |        |
| Sleep Staging (MASS-SS3)                    |                   |                    |                               |      |        |
| Pathology Detection                         |                   |                    |                               |      |        |
| Arrhythmia Detection                        |                   |                    |                               |      |        |



## Framework

- Data Pre-processing
- Dataset Preparing: The augmented/transformed dataset. Some of them can be processed during dataset construction phase, but others need to be finished during training phase.
- Model Construction
- Training (Objective Function)
- Fine-tuning: Encoder with MLP classifier
- Evaluation



## Implementation Details

### Relative Position

#### Dataset Construction

Including augmentation, transformation, returning types...

#### Training Procedure

Including in-training data processing, loss function...

### Temporal Shuffling



### Transformation Distinguishing



### Contrastive Predictive Coding (CPC)



### SimCLR



## Experimental Results

