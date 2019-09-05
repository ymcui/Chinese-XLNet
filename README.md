[**中文说明**](https://github.com/ymcui/Chinese-PreTrained-XLNet/) | [**English**](https://github.com/ymcui/Chinese-PreTrained-XLNet/blob/master/README_EN.md)

## 中文XLNet预训练模型（Chinese Pre-Trained XLNet）
本项目提供了面向中文的XLNet预训练模型，旨在丰富中文自然语言处理资源，提供多元化的中文预训练模型选择。
我们欢迎各位专家学者下载使用，并共同促进和发展中文资源建设。

本项目基于CMU/谷歌官方的XLNet：https://github.com/zihangdai/xlnet

## 新闻
**2019/9/5 `XLNet-base`已可下载**  
2019/8/19 提供了在大规模通用语料（5.4B词数）上训练的中文`XLNet-mid`模型，查看[模型下载](#模型下载)


## TODO List
- ~~上传TensorFlow版本@Google Drive以及讯飞云下载点~~
- ~~上传PyTorch版本~~
- ~~上传Fine-tuning脚本~~
- ~~完善fine-tuning使用说明~~
- ~~英文版README~~
- ~~上传XLNet-base~~


## 内容导引
| 章节 | 描述 |
|-|-|
| [模型下载](#模型下载) | 提供了中文预训练XLNet下载地址 |
| [基线系统效果](#基线系统效果) | 列举了部分基线系统效果 |
| [预训练细节](#预训练细节) | 预训练细节的相关描述 |
| [下游任务微调细节](#下游任务微调细节) | 下游任务微调细节的相关描述 |
| [FAQ](#faq) | 常见问题答疑 |


## 模型下载
* **`XLNet-mid`**：24-layer, 768-hidden, 12-heads, 209M parameters
* **`XLNet-base`**：12-layer, 768-hidden, 12-heads, 117M parameters  

| 模型简称 | 语料 | Google下载 | 讯飞云下载 |
| :------- | :--------- | :---------: | :---------: |
| **`XLNet-mid, Chinese`** | **中文维基+<br/>通用数据<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1342uBc7ZmQwV6Hm6eUIN_OnBSz1LcvfA)** <br/>**[PyTorch](https://drive.google.com/open?id=1u-UmsJGy5wkXgbNK4w9uRnC0RxHLXhxy)** | **[TensorFlow（密码f5ux）](https://pan.iflytek.com:443/link/AE46DD1269A4D253447488ACF050E7DD)** <br/>**[PyTorch（密码vnnt）](https://pan.iflytek.com:443/link/92F000AE7BA874BCA00051E12B3EC1DE)** |
| **`XLNet-base, Chinese`** | **中文维基+<br/>通用数据<sup>[1]</sup>** | **[TensorFlow](https://drive.google.com/open?id=1m9t-a4gKimbkP5rqGXXsEAEPhJSZ8tvx)** <br/> | **TensorFlow** <br/> |

> [1] 通用数据包括：百科、新闻、问答等数据，总词数达5.4B，与我们发布的[BERT-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm)训练语料相同。

以上预训练模型以TensorFlow版本的权重为准。
对于PyTorch版本，我们使用的是由Huggingface出品的[PyTorch-Transformers 1.0](https://github.com/huggingface/pytorch-transformers)提供的转换脚本。
如果使用的是其他版本，请自行进行权重转换。
中国大陆境内建议使用讯飞云下载点，境外用户建议使用谷歌下载点，`XLNet-mid`模型文件大小约**800M**。 
以TensorFlow版`XLNet-mid, Chinese`为例，下载完毕后对zip文件进行解压得到：
```
chinese_xlnet_mid_L-24_H-768_A-12.zip
    |- xlnet_model.ckpt      # 模型权重
    |- xlnet_model.meta      # 模型meta信息
    |- xlnet_model.index     # 模型index信息
    |- xlnet_config.json     # 模型参数
    |- spiece.model          # 词表
```


## 基线系统效果
为了对比基线效果，我们在以下几个中文数据集上进行了测试。对比了中文BERT、BERT-wwm、BERT-wwm-ext以及XLNet-base、XLNet-mid。
其中中文BERT、BERT-wwm、BERT-wwm-ext结果取自[中文BERT-wwm项目](https://github.com/ymcui/Chinese-BERT-wwm)。
时间及精力有限，并未能覆盖更多类别的任务，请大家自行尝试。

**注意：为了保证结果的可靠性，对于同一模型，我们运行10遍（不同随机种子），汇报模型性能的最大值和平均值。不出意外，你运行的结果应该很大概率落在这个区间内。**


### 简体中文阅读理解：CMRC 2018
**[CMRC 2018数据集](https://github.com/ymcui/cmrc2018)**是哈工大讯飞联合实验室发布的中文机器阅读理解数据。
根据给定问题，系统需要从篇章中抽取出片段作为答案，形式与SQuAD相同。

| 模型 | 开发集 | 测试集 | 挑战集 |
| :------- | :---------: | :---------: | :---------: |
| BERT | 65.5 (64.4) / 84.5 (84.0) | 70.0 (68.7) / 87.0 (86.3) | 18.6 (17.0) / 43.3 (41.3) | 
| BERT-wwm | 66.3 (65.0) / 85.6 (84.7) | 70.5 (69.1) / 87.4 (86.7) | 21.0 (19.3) / 47.0 (43.9) | 
| BERT-wwm-ext | **67.1** (65.6) / 85.7 (85.0) | **71.4 (70.0)** / 87.7 (87.0) | 24.0 (20.0) / 47.3 (44.6) |
| **XLNet-base** | 65.2 (63.0) / 86.9  (85.9) | 67.0 (65.8) / 87.2 (86.8) | 25.0 (22.7) / 51.3 (49.5) | 
| **XLNet-mid** | 66.8 **(66.3) / 88.4 (88.1)** | 69.3 (68.5) / **89.2 (88.8)** | **29.1 (27.1) / 55.8 (54.9)** |


### 繁体中文阅读理解：DRCD
**[DRCD数据集](https://github.com/DRCKnowledgeTeam/DRCD)**由中国台湾台达研究院发布，其形式与SQuAD相同，是基于繁体中文的抽取式阅读理解数据集。

| 模型 | 开发集 | 测试集 |
| :------- | :---------: | :---------: |
| BERT | 83.1 (82.7) / 89.9 (89.6) | 82.2 (81.6) / 89.2 (88.8) | 
| BERT-wwm | 84.3 (83.4) / 90.5 (90.2) | 82.8 (81.8) / 89.7 (89.0) | 
| BERT-wwm-ext | 85.0 (84.5) / 91.2 (90.9) | 83.6 (83.0) / 90.4 (89.9) |
| **XLNet-base** | 83.8 (83.2) / 92.3 (92.0) | 83.5 (82.8) / 92.2 (91.8) |
| **XLNet-mid** | **85.3 (84.9) / 93.5 (93.3)** | **85.5 (84.8) / 93.6 (93.2)** |

### 情感分类：ChnSentiCorp
在情感分类任务中，我们使用的是ChnSentiCorp数据集。模型需要将文本分成`积极`, `消极`两个类别。

| 模型 | 开发集 | 测试集 |
| :------- | :---------: | :---------: |
| BERT | 94.7 (94.3) | 95.0 (94.7) |  
| BERT-wwm | 95.1 (94.5) | **95.4 (95.0)** |
| **XLNet-base** | | |
| **XLNet-mid** | **95.8 (95.2)** | **95.4** (94.9) |

## 预训练细节
以下以`XLNet-mid`模型为例，对预训练细节进行说明。

### 生成词表
按照XLNet官方教程步骤，首先需要使用[Sentence Piece](https://github.com/google/sentencepiece)生成词表。
在本项目中，我们使用的词表大小为32000，其余参数采用官方示例中的默认配置。

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

### 生成tf_records
生成词表后，开始利用原始文本语料生成训练用的tf_records文件。
原始文本的构造方式与原教程相同：
- 每行都是一个句子
- 空行代表文档末尾

以下是生成数据时的命令（`num_task`与`task`请根据实际切片数量进行设置）：
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

### 预训练
获得以上数据后，正式开始预训练XLNet。
之所以叫`XLNet-mid`是因为仅相比`XLNet-base`增加了层数（12层增加到24层），其余参数没有变动，主要因为计算设备受限。
使用的命令如下：
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

## 下游任务微调细节
下游任务微调使用的设备是谷歌Cloud TPU v2（64G HBM），以下简要说明各任务精调时的配置。
如果你使用GPU进行精调，请更改相应参数以适配，尤其是`batch_size`, `learning_rate`等参数。

### CMRC 2018
对于阅读理解任务，首先需要生成tf_records数据。
请参考XLNet官方教程之[SQuAD 2.0处理方法](https://github.com/zihangdai/xlnet#squad20)，在这里不再赘述。
以下是CMRC 2018中文机器阅读理解任务中使用的脚本参数：
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
以下是DRCD繁体中文机器阅读理解任务中使用的脚本参数：
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
与阅读理解任务不同，分类任务无需提前生成tf_records。
以下是ChnSentiCorp情感分类任务中使用的脚本参数：
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
	--learning_rate=2e-5 \
	--save_steps=5000 \
	--use_tpu=True \
	--tpu=${TPU_NAME} \
	--tpu_zone=${TPU_ZONE}
```

## FAQ
**Q: 会发布更大的模型吗？**  
A: 不一定，不保证。如果我们获得了显著性能提升，会考虑发布出来。

**Q: 在某些数据集上效果不好？**  
A: 选用其他模型或者在这个checkpoint上继续用你的数据做预训练。

**Q: 预训练数据会发布吗？**  
A: 抱歉，因为版权问题无法发布。

**Q: 训练XLNet花了多长时间？**  
A: `XLNet-mid`使用了Cloud TPU v3 (128G HBM)训练了2M steps（batch=32），大约需要3周时间。`XLNet-base`则是训练了4M steps。

**Q: 为什么XLNet官方没有发布Multilingual或者Chinese XLNet？**  
A: 
（以下是个人看法）不得而知，很多人留言表示希望有，戳[XLNet-issue-#3](https://github.com/zihangdai/xlnet/issues/3)。
以XLNet官方的技术和算力来说，训练一个这样的模型并非难事（multilingual版可能比较复杂，需要考虑各语种之间的平衡，也可以参考[multilingual-bert](https://github.com/google-research/bert/blob/master/multilingual.md)中的描述。）。 
**不过反过来想一下，作者们也并没有义务一定要这么做。** 
作为学者来说，他们的technical contribution已经足够，不发布出来也不应受到指责，呼吁大家理性对待别人的工作。

**Q: XLNet多数情况下比BERT要好吗？**  
A: 目前看来至少上述几个任务效果都还不错，使用的数据和我们发布的[BERT-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm)是一样的。

**Q: ？**  
A: 。


## 致谢
项目作者： 崔一鸣（哈工大讯飞联合实验室）、车万翔（哈工大）、刘挺（哈工大）、王士进（科大讯飞）、胡国平（科大讯飞）  

本项目受到谷歌[TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc)计划资助。

建设该项目过程中参考了如下仓库，在这里表示感谢：
- XLNet: https://github.com/zihangdai/xlnet
- Malaya: https://github.com/huseinzol05/Malaya/tree/master/xlnet
- Korean XLNet（韩文描述，无翻译）: https://github.com/yeontaek/XLNET-Korean-Model


## 免责声明
本项目并非[XLNet官方](https://github.com/zihangdai/xlnet)发布的Chinese XLNet模型。
同时，本项目不是哈工大或科大讯飞的官方产品。
该项目中的内容仅供技术研究参考，不作为任何结论性依据。
使用者可以在许可证范围内任意使用该模型，但我们不对因使用该项目内容造成的直接或间接损失负责。


## 关注我们
欢迎关注哈工大讯飞联合实验室官方微信公众号。

![qrcode.png](https://github.com/ymcui/cmrc2019/raw/master/qrcode.jpg)


## 问题反馈 & 贡献
如有问题，请在GitHub Issue中提交。  
我们没有运营，鼓励网友互相帮助解决问题。  
如果发现实现上的问题或愿意共同建设该项目，请提交Pull Request。  

