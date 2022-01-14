# DuCLEF
This anonymous repository contains the source code for the SIGIR 2022 submission "From Easy to Hard: A Dual Curriculum Learning Framework for Context-Aware Document Ranking".

## Requirements
- Python 3.6
- PyTorch 1.8.0
- Transformers 4.2.0
- pytrec-eval 0.5  

## Usage
- Obtain the data
  - For AOL dataset, please contact the author of [CARS](https://arxiv.org/pdf/1906.02329.pdf)
  - For Tiangong dataset, download from the [link](http://www.thuir.cn/tiangong-st/)
  - Move all data to the "data" directory
- Prepare pretrained BERT
  - [BertModel](https://huggingface.co/bert-base-uncased)
  - [BertChinese](https://huggingface.co/bert-base-chinese)
  - Save these models to the "pretrained_model" directory 
- Training model
```
python3 runModelCL.py
```
