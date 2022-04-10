# DCL
This anonymous repository contains the source code for the SIGIR 2022 submission "From Easy to Hard: A Dual Curriculum Learning Framework for Context-Aware Document Ranking".

## Requirements
- Python 3.6
- PyTorch 1.8.0
- Transformers 4.2.0
- pytrec-eval 0.5  

## Usage
- Obtain the data
  - According to the rule of anonymity, we will provide the link of the preprocessed dataset later
  - Decompress all data to the "data" directory
- Prepare the pretrained BERT model
  - [BertModel](https://huggingface.co/bert-base-uncased)
  - [BertChinese](https://huggingface.co/bert-base-chinese)
  - Save these models to the "pretrained_model" directory 
- Train the model
```
python3 runModelCL.py --task aol/tiangong
```
- Test the model
```
python3 runModelCL.py --is_training False --task aol/tiangong
```
