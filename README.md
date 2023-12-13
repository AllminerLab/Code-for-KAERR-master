# KAERR
This is the official PyTorch implementation for the paper:

> Kai-Huang Lai, Zhe-Rui Yang, Pei-Yuan Lai, Chang-Dong Wang, Mohsen Guizani, Min Chen. "Knowledge-Aware Explainable Reciprocal Recommendation". AAAI 2024.

## Requirements
```
torch==1.12.1
dgl==1.1.1+cu113
recbole==1.1.1
```

## Dataset
- Zhaopin

Raw data can be download from [TIANCHI](https://tianchi.aliyun.com/dataset/dataDetail?dataId=31623) data contest.

Put ```*.txt``` files into ```dataset/zhaopin_kg/```, and run scripts in the folder to preprocess the dataset.

## Usage

```bash
python main.py --model KAERR --dataset zhaopin_kg --device 0
```

## Acknowledgement

The implementation is based on the open-source recommendation library [RecBole-PJF](https://github.com/RUCAIBox/RecBole-PJF).
