[**中文说明**](https://github.com/ymcui/Chinese-PreTrained-XLNet/) | [**English**](https://github.com/ymcui/Chinese-PreTrained-XLNet/blob/master/README_EN.md)

## Chinese Pre-Trained XLNet
This project provides a XLNet pre-training model for Chinese, which aims to enrich Chinese natural language processing resources and provide a variety of Chinese pre-training model selection.
We welcome all experts and scholars to download and use this model.

This project is based on CMU/Google official XLNet: https://github.com/zihangdai/xlnet

## News
**2019/9/5 `XLNet-base` has been released. **  
2019/8/19 We provide pre-trained Chinese `XLNet-mid` model, which was trained on large-scale data. Check [Download](#Download)


## Guide
| Section | Description |
|-|-|
| [Download](#Download) | Download links for Chinese XLNet |
| [Baselines](#Baselines) | Baseline results for several Chinese NLP datasets (partial) |
| [Pre-training Details](#Pre-training-Details) | Details for pre-training |
| [Fine-tuning Details](#Fine-tuning-Details) | Details for fine-tuning |
| [FAQ](#faq) | Frequently Asked Questions |

## Download
* **`XLNet-mid`**：24-layer, 768-hidden, 12-heads, 209M parameters  
* **`XLNet-base`**：12-layer, 768-hidden, 12-heads, 117M parameters  

| Model | Data | Google Drive | iFLYTEK Cloud |
| :------- | :--------- | :---------: | :---------: |
| **`XLNet-mid, Chinese`** | **Wikipedia+Extended data<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1342uBc7ZmQwV6Hm6eUIN_OnBSz1LcvfA)** <br/>**[PyTorch](https://drive.google.com/open?id=1u-UmsJGy5wkXgbNK4w9uRnC0RxHLXhxy)** | **[TensorFlow (pw: f5ux)](https://pan.iflytek.com:443/link/AE46DD1269A4D253447488ACF050E7DD)** <br/>**[PyTorch (pw: vnnt)](https://pan.iflytek.com:443/link/92F000AE7BA874BCA00051E12B3EC1DE)** |
| **`XLNet-base, Chinese`** | **Wikipedia+Extended data<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1m9t-a4gKimbkP5rqGXXsEAEPhJSZ8tvx)** <br/> | **TensorFlow** <br/> |

> [1] Extended data includes: baike, news, QA data, with 5.4B words in total, which is exactly the same with [BERT-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm).

The whole zip package roughly takes ~800M for `XLNet-mid` model.
ZIP package includes the following files:
```
chinese_xlnet_mid_L-24_H-768_A-12.zip
    |- xlnet_model.ckpt      # Model Weights
    |- xlnet_model.meta      # Meta info
    |- xlnet_model.index     # Index info
    |- xlnet_config.json     # Config file
    |- spiece.model          # Vocabulary
```


## Baselines
We conduct experiments on several Chinese NLP data, and compare the performance among BERT, BERT-wwm, BERT-wwm-ext, XLNet-base, and XLNet-mid.
The results of BERT/BERT-wwm/BERT-wwm-ext were extracted from [Chinese BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm).

**Note: To ensure the stability of the results, we run 10 times for each experiment and report maximum and average scores.**


### [CMRC 2018](https://github.com/ymcui/cmrc2018)
CMRC 2018 dataset is released by Joint Laboratory of HIT and iFLYTEK Research.
The model should answer the questions based on the given passage, which is identical to SQuAD.

| Model | Development | Test | Challenge |
| :------- | :---------: | :---------: | :---------: |
| BERT | 65.5 (64.4) / 84.5 (84.0) | 70.0 (68.7) / 87.0 (86.3) | 18.6 (17.0) / 43.3 (41.3) | 
| BERT-wwm | 66.3 (65.0) / 85.6 (84.7) | 70.5 (69.1) / 87.4 (86.7) | 21.0 (19.3) / 47.0 (43.9) | 
| BERT-wwm-ext | **67.1** (65.6) / 85.7 (85.0) | **71.4 (70.0)** / 87.7 (87.0) | 24.0 (20.0) / 47.3 (44.6) |
| **XLNet-base** | 65.2 (63.0) / 86.9  (85.9) | 67.0 (65.8) / 87.2 (86.8) | 25.0 (22.7) / 51.3 (49.5) | 
| **XLNet-mid** | 66.8 **(66.3) / 88.4 (88.1)** | 69.3 (68.5) / **89.2 (88.8)** | **29.1 (27.1) / 55.8 (54.9)** |


### [DRCD](https://github.com/DRCKnowledgeTeam/DRCD)
DRCD is also a span-extraction machine reading comprehension dataset, released by Delta Research Center. The text is written in Traditional Chinese.

| Model | Development | Test |
| :------- | :---------: | :---------: |
| BERT | 83.1 (82.7) / 89.9 (89.6) | 82.2 (81.6) / 89.2 (88.8) | 
| BERT-wwm | 84.3 (83.4) / 90.5 (90.2) | 82.8 (81.8) / 89.7 (89.0) | 
| BERT-wwm-ext | 85.0 (84.5) / 91.2 (90.9) | 83.6 (83.0) / 90.4 (89.9) |
| **XLNet-base** | 83.8 (83.2) / 92.3 (92.0) | 83.5 (82.8) / 92.2 (91.8) |
| **XLNet-mid** | **85.3 (84.9) / 93.5 (93.3)** | **85.5 (84.8) / 93.6 (93.2)** |


### Sentiment Classification: ChnSentiCorp
We use ChnSentiCorp data for sentiment classification, which is a binary classification task.

| Model | Development | Test |
| :------- | :---------: | :---------: |
| BERT | 94.7 (94.3) | 95.0 (94.7) |  
| BERT-wwm | 95.1 (94.5) | **95.4 (95.0)** |
| **XLNet-base** | | |
| **XLNet-mid** | **95.8 (95.2)** | **95.4** (94.9) |


## Pre-training Details
We take `XLNet-mid` for example to demonstrate the pre-training details.

### Generate Vocabulary
Following official tutorial of XLNet, we need to generate vocabulary using [Sentence Piece](https://github.com/google/sentencepiece).
In this project, we use a vocabulary of 32000 words.
The rest of the parameters are identical to the default settings.

```
spm_train \
	--input=wiki.zh.txt \
	--model_prefix=sp10m.cased.v3 \
	--vocab_size=32000 \
	--character_coverage=0.99995 \
	--model_type=unigram \
	--control_symbols=\<cls\>,\<sep\>,\<pad\>,\<mask\>,\<eod\> \
	--user_defined_symbols=\<eop\>,.,\(,\),\",-,–,£,€ \
	--shuffle_input_sentence \
	--input_sentence_size=10000000
```

### Generate tf_records
We use raw text files to generate tf_records.
```
SAVE_DIR=./output_b32
INPUT=./data/*.proc.txt

python data_utils.py \
	--bsz_per_host=32 \
	--num_core_per_host=8 \
	--seq_len=512 \
	--reuse_len=256 \
	--input_glob=${INPUT} \
	--save_dir=${SAVE_DIR} \
	--num_passes=20 \
	--bi_data=True \
	--sp_path=spiece.model \
	--mask_alpha=6 \
	--mask_beta=1 \
	--num_predict=85 \
	--uncased=False \
	--num_task=10 \
	--task=1
```

### Pre-training
Now we can pre-train our Chinese XLNet.
Note that, `XLNet-mid` is named because of it only increases the number of Transformers (from 12 to 24).

```
DATA=YOUR_GS_BUCKET_PATH_TO_TFRECORDS
MODEL_DIR=YOUR_OUTPUT_MODEL_PATH
TPU_NAME=v3-xlnet
TPU_ZONE=us-central1-b

python train.py \
	--record_info_dir=$DATA \
	--model_dir=$MODEL_DIR \
	--train_batch_size=32 \
	--seq_len=512 \
	--reuse_len=256 \
	--mem_len=384 \
	--perm_size=256 \
	--n_layer=24 \
	--d_model=768 \
	--d_embed=768 \
	--n_head=12 \
	--d_head=64 \
	--d_inner=3072 \
	--untie_r=True \
	--mask_alpha=6 \
	--mask_beta=1 \
	--num_predict=85 \
	--uncased=False \
	--train_steps=2000000 \
	--save_steps=20000 \
	--warmup_steps=20000 \
	--max_save=20 \
	--weight_decay=0.01 \
	--adam_epsilon=1e-6 \
	--learning_rate=1e-4 \
	--dropout=0.1 \
	--dropatt=0.1 \
	--tpu=$TPU_NAME \
	--tpu_zone=$TPU_ZONE \
	--use_tpu=True
```

## Fine-tuning Details
We use Google Cloud TPU v2 (64G HBM) for fine-tuning.

### CMRC 2018
For reading comprehension tasks, we first need to generate tf_records data.
Please infer official tutorial of XLNet: [SQuAD 2.0](https://github.com/zihangdai/xlnet#squad20).

```
XLNET_DIR=YOUR_GS_BUCKET_PATH_TO_XLNET
MODEL_DIR=YOUR_OUTPUT_MODEL_PATH
DATA_DIR=YOUR_DATA_DIR_TO_TFRECORDS
RAW_DIR=YOUR_RAW_DATA_DIR
TPU_NAME=v2-xlnet
TPU_ZONE=us-central1-b

python -u run_cmrc_drcd.py \
	--spiece_model_file=./spiece.model \
	--model_config_path=${XLNET_DIR}/xlnet_config.json \
	--init_checkpoint=${XLNET_DIR}/xlnet_model.ckpt \
	--tpu_zone=${TPU_ZONE} \
	--use_tpu=True \
	--tpu=${TPU_NAME} \
	--num_hosts=1 \
	--num_core_per_host=8 \
	--output_dir=${DATA_DIR} \
	--model_dir=${MODEL_DIR} \
	--predict_dir=${MODEL_DIR}/eval \
	--train_file=${DATA_DIR}/cmrc2018_train.json \
	--predict_file=${DATA_DIR}/cmrc2018_dev.json \
	--uncased=False \
	--max_answer_length=40 \
	--max_seq_length=512 \
	--do_train=True \
	--train_batch_size=16 \
	--do_predict=True \
	--predict_batch_size=16 \
	--learning_rate=3e-5 \
	--adam_epsilon=1e-6 \
	--iterations=1000 \
	--save_steps=2000 \
	--train_steps=2400 \
	--warmup_steps=240
```

### DRCD

```
XLNET_DIR=YOUR_GS_BUCKET_PATH_TO_XLNET
MODEL_DIR=YOUR_OUTPUT_MODEL_PATH
DATA_DIR=YOUR_DATA_DIR_TO_TFRECORDS
RAW_DIR=YOUR_RAW_DATA_DIR
TPU_NAME=v2-xlnet
TPU_ZONE=us-central1-b

python -u run_cmrc_drcd.py \
	--spiece_model_file=./spiece.model \
	--model_config_path=${XLNET_DIR}/xlnet_config.json \
	--init_checkpoint=${XLNET_DIR}/xlnet_model.ckpt \
	--tpu_zone=${TPU_ZONE} \
	--use_tpu=True \
	--tpu=${TPU_NAME} \
	--num_hosts=1 \
	--num_core_per_host=8 \
	--output_dir=${DATA_DIR} \
	--model_dir=${MODEL_DIR} \
	--predict_dir=${MODEL_DIR}/eval \
	--train_file=${DATA_DIR}/DRCD_training.json \
	--predict_file=${DATA_DIR}/DRCD_dev.json \
	--uncased=False \
	--max_answer_length=30 \
	--max_seq_length=512 \
	--do_train=True \
	--train_batch_size=16 \
	--do_predict=True \
	--predict_batch_size=16 \
	--learning_rate=3e-5 \
	--adam_epsilon=1e-6 \
	--iterations=1000 \
	--save_steps=2000 \
	--train_steps=3600 \
	--warmup_steps=360
```

### ChnSentiCorp
Different from reading comprehension task, we do not need to generate tf_records in advance.

```
XLNET_DIR=YOUR_GS_BUCKET_PATH_TO_XLNET
MODEL_DIR=YOUR_OUTPUT_MODEL_PATH
DATA_DIR=YOUR_DATA_DIR_TO_TFRECORDS
RAW_DIR=YOUR_RAW_DATA_DIR
TPU_NAME=v2-xlnet
TPU_ZONE=us-central1-b

python -u run_classifier.py \
	--spiece_model_file=./spiece.model \
	--model_config_path=${XLNET_DIR}/xlnet_config.json \
	--init_checkpoint=${XLNET_DIR}/xlnet_model.ckpt \
	--task_name=csc \
	--do_train=True \
	--do_eval=True \
	--eval_all_ckpt=False \
	--uncased=False \
	--data_dir=${RAW_DIR} \
	--output_dir=${DATA_DIR} \
	--model_dir=${MODEL_DIR} \
	--train_batch_size=48 \
	--eval_batch_size=48 \
	--num_hosts=1 \
	--num_core_per_host=8 \
	--num_train_epochs=3 \
	--max_seq_length=256 \
	--learning_rate=3e-5 \
	--save_steps=5000 \
	--use_tpu=True \
	--tpu=${TPU_NAME} \
	--tpu_zone=${TPU_ZONE}
```

## FAQ
**Q: Will you release larger data?**  
A: It depends.

**Q: Bad results on some datasets?**  
A: Please use other pre-trained model or continue to do pre-training on your own data.

**Q: Will you publish the data used in pre-training?**  
A: Nope, copyright is the biggest concern.

**Q: How long did you take to train XLNet-mid?**  
A: We use Cloud TPU v3 (128G HBM) to train 2M steps with batch size of 32, which takes roughly three weeks.

**Q: Does XLNet perform better than BERT in most of the times?**  
A: Seems to be right. At least the tasks we tried above are substantially better than BERTs.


## Acknowledgement
Authors: Yiming Cui (Joint Laboratory of HIT and iFLYTEK Research, HFL), Wanxiang Che (Harbin Institute of Technology), Ting Liu (Harbin Institute of Technology), Shijin Wang (iFLYTEK), Guoping Hu (iFLYTEK)

This project is supported by Google [TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc) Program。

We also refered to the following repository:
- XLNet: https://github.com/zihangdai/xlnet
- Malaya: https://github.com/huseinzol05/Malaya/tree/master/xlnet
- Korean XLNet: https://github.com/yeontaek/XLNET-Korean-Model


## Disclaimer
**This is NOT a project by [XLNet official](https://github.com/zihangdai/xlnet). Also, this is NOT an official product by HIT and iFLYTEK.**

The experiments only represent the empirical results in certain conditions and should not be regarded as the nature of the respective models. The results may vary using different random seeds, computing devices, etc. 

**The contents in this repository are for academic research purpose, and we do not provide any conclusive remarks. Users are free to use anythings in this repository within the scope of Apache-2.0 licence. However, we are not responsible for direct or indirect losses that was caused by using the content in this project.**

## Issues
If there is any problem, please submit a GitHub Issue.
