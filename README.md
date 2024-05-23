# GRACE

The official PyTorch implementation of Perfect Alignment May be Poisonous to Graph Contrastive Learning.

## Dependencies

- torch 1.4.0
- torch-geometric 1.5.0
- sklearn 0.21.3
- numpy 1.18.1
- pyyaml 5.3.1

Install all dependencies using
```
pip install -r requirements.txt
```

## Usage

Train and evaluate the model by executing
```
python train.py --dataset Cora --method I #information augmentation
python train.py --dataset Cora --method S #spectrum augmentation
```

## Citation

If you use our code in your own research, please cite the following article:

```
@inproceedings{
anonymous2024perfect,
title={Perfect Alignment May be Poisonous to Graph Contrastive Learning},
author={Anonymous},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=wdezvnc9EG}
}
```
