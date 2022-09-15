# From Easy to Hard: A Dual Curriculum Learning Framework for Context-Aware Document Ranking

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)

This repository contains the source code and datasets for the CIKM 2022 paper [From Easy to Hard: A Dual Curriculum Learning Framework for Context-Aware Document Ranking](https://arxiv.org/pdf/2208.10226.pdf) by Zhu et al. <br>

## Abstract

Contextual information in search sessions is important for capturing users' search intents. Various approaches have been proposed to model user behavior sequences to improve document ranking in a session. Typically, training samples of (search context, document) pairs are sampled randomly in each training epoch. In reality, the difficulty to understand user's search intent and to judge document's relevance varies greatly from one search context to another. Mixing up training samples of  different difficulties may confuse the model's optimization process. In this work, we propose a curriculum learning framework for context-aware document ranking, in which the ranking model learns matching signals between the search context and the candidate document in an easy-to-hard manner. In so doing, we aim to guide the model gradually toward a global optimum. To leverage both positive and negative examples, two curricula are designed. Experiments on two real query log datasets show that our proposed framework can improve the performance of several existing methods significantly, demonstrating the effectiveness of curriculum learning for context-aware document ranking.

Authors: Yutao Zhu, Jian-Yun Nie, Yixuan Su, Haonan Chen, Xinyu Zhang, and Zhicheng Dou

## Requirements
- Python 3.8.5 <br>
- Pytorch 1.8.1 (with GPU support) <br>
- Transformers 4.5.1 <br>
- [pytrec-eval](https://pypi.org/project/pytrec-eval/) 0.5  

## Usage
- Obtain the data (some data samples are provided in the data directory)
  - For AOL dataset, please contact the author of [CARS](https://arxiv.org/pdf/1906.02329.pdf)
  - For Tiangong dataset, you can download it from the [link](http://www.thuir.cn/tiangong-st/)
  - Decompress all data to the "data" directory
- Prepare the pretrained BERT model
  - [BertModel](https://huggingface.co/bert-base-uncased)
  - [BertChinese](https://huggingface.co/bert-base-chinese)
  - Save these models to the "pretrained_model" directory 
- Prepare the pretrained COCA model
  - Download the contrastive pretrained model from the [link](https://github.com/DaoD/COCA)
  - Save the checkpoint to the "pretrained_model" directory
- Train the model (on AOL)
```
python3 runModelCL.py --task aol --is_training --bert_model_path ./pretrained_model/BERT/ --pretrain_model_path ./pretrained_model/coca.aol
```
- Test the model (on AOL)
```
python3 runModelCL.py --task aol
```

## Citations
If you use the code and datasets, please cite the following paper:  
```
@inproceedings{ZhuNSCZD22,
  author    = {Yutao Zhu and
               Jian{-}Yun Nie and
               Yixuan Su and
               Haonan Chen and
               Xinyu Zhang and
               Zhicheng Dou},
  title     = {From Easy to Hard: A Dual Curriculum Learning Framework for Context-Aware Document Ranking},
  booktitle = {{CIKM} '22: The 31st {ACM} International Conference on Information
               and Knowledge Management, Atlanta, GA, USA, October
               17 - 21, 2022},
  publisher = {{ACM}},
  year      = {2022},
  url       = {https://doi.org/10.1145/3511808.3557328},
  doi       = {10.1145/3511808.3557328}
}
```
