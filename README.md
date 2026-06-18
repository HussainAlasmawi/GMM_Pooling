# From Point Estimates to Distributions: GMM Pooling for MIL in Preterm Birth Prediction

Official implementation of **GMM Pooling**, a multiple instance learning pooling method that represents each bag using a Gaussian mixture distribution rather than collapsing instance features into a single vector.

This repository contains the code used for the paper:
> **From Point Estimates to Distributions: GMM Pooling for MIL in Preterm Birth Prediction**  
> Hussain Alasmawi, Numan Saeed, Soha Said, Mohammad Yaqub

## Overview

Multiple Instance Learning is commonly used when image-level labels are available but instance-level annotations are missing. Standard pooling methods such as mean, max, or attention pooling summarize a bag into a single representation, which may discard useful intra-bag heterogeneity.

GMM Pooling instead models the distribution of instance-level features using multiple Gaussian components. This allows the model to capture richer bag-level structure, including feature variability and relationships between instances.

## Repository Contents

- `positive_negative_classifcation/` — classification experiments
- `regression/` — regression experiments
- `README.md` — setup, training, evaluation, and data/model instructions

Given a bag of instance features, GMM Pooling estimates a mixture-based representation of the feature distribution. The resulting parameters are used as the bag-level representation for downstream prediction.

<p align="center">
  <img src="assets/motivation_figure.png" width="700">
</p>

## Main Results

### Preterm Birth Prediction

| Method | PR-AUC ↑ | ROC-AUC ↑ |
|----------|----------|----------|
| Instance-based | 0.44 ± 0.01 | 0.64 ± 0.02 |
| Max Pooling | **0.57 ± 0.05** | **0.71 ± 0.03** |
| Attention Pooling | 0.56 ± 0.02 | 0.67 ± 0.03 |
| Density Pooling | 0.54 ± 0.03 | 0.68 ± 0.02 |
| **GMM Pooling (Ours)** | **0.56 ± 0.03** | **0.69 ± 0.03** |

### Lymph Node Metastasis Benchmark

| Method | F1 ↑ | ROC-AUC ↑ | MAE ↓ |
|----------|----------|----------|----------|
| Max Pooling | 0.84 ± 0.03 | 0.83 ± 0.05 | 0.24 ± 0.01 |
| Mean Pooling | 0.86 ± 0.04 | 0.86 ± 0.03 | 0.23 ± 0.02 |
| Attention Pooling | 0.87 ± 0.02 | 0.88 ± 0.02 | 0.18 ± 0.04 |
| Density Pooling | 0.87 ± 0.01 | 0.87 ± 0.01 | **0.17 ± 0.01** |
| **GMM Pooling (Ours)** | **0.91 ± 0.01** | **0.89 ± 0.01** | 0.18 ± 0.02 |

Key observations:
- MIL substantially improves over instance-based PTB prediction.
- GMM Pooling achieves competitive PTB performance while modeling intra-bag feature distributions.
- GMM Pooling achieves state-of-the-art classification performance on the lymph node benchmark.


## Requirements

All the experiments were run in a virtual environment created with pip on a Linux machine. You can create a virtual environment and install the required packages.

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train a model with a specific MIL pooling filter (e.g. 'distribution' pooling filter), run:

```train
python train.py --mil_pooling_filter "gmm"
```

> Some other hyper-parameters can also be passed to 'train.py' as '--key value' pairs. Please check 'train.py' for the full list of hyper-parameters. Also note that all hyper-parameters are set to the values used in the paper by default.

To run a full set of experiments with all 4 pooling filters in a task, run:

```shell
$ train.sh
```

## Evaluation

To test a model on the bags created from test set images, run:

```test
python test.py --mil_pooling_filter "distribution" --init_model_file "saved_models/checkpoint.pth"
```
> This will generate and test multiple bags for each image and save the results in 'test_metrics/checkpoint/test' folder.

To obtain image level statistics, run:

```
python collect_statistics_over_bag_predictions.py --data_folder_path "test_metrics/checkpoint/test"
```

To test full set of models with all 5 pooling filters and collect image level statistics in a task, run:

```shell
$ test.sh
```
> Note that this will test the models used in the paper. If you want to test your own models, go an update the script accordingly. Note that this script will run the statistical tests to compare the trained models as well.

## Dataset

The lymph node metastases dataset used in the paper can be downloaded from [here](https://bit.ly/mil_pooling_filters).


## Citation

If you use this code, please cite:

```bibtex
@inproceedings{alasmawi2026gmmpooling,
  title={From Point Estimates to Distributions: GMM Pooling for MIL in Preterm Birth Prediction},
  author={Alasmawi, Hussain and others},
  booktitle={},
  year={2026}
}
```
## Contact
If you have any questions, please create an issue on this repository or contact at hussain.alasmawi@mbzuai.ac.ae.

## Acknowledgements

Parts of this codebase are adapted from the **mil_pooling_filters** repository by Mustafa Ümit Öner:

https://github.com/onermustafaumit/mil_pooling_filters
