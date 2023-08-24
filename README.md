<div align= "center">
    <h1> 🦙Llama2-Chinese🦙 </h1>
</div>

这个项目旨在构建**开源、大规模、高质量**的预训练/指令调整SFT/领域垂直/RLHF数据，进一步增强Llama2大模型的中文能力以及打造多种垂直领域模型。我们提供数据集、相应的训练和评估脚本，以及经过微调的强大模型Llama2-Chinese。

这个工作的独特之处
- **中文化只是开始**：Llama2中文化只是第一步！我们的目标是在Llama2-Chinese基础上持续加强数学能力&插件调用能力。
- **开源与透明**：采用开源的预训练数据、指令微调数据、以及RLHF数据，致力于构建可复现的、透明的研究生态。
- **一站式训练方法**：实现二次预训练，词表扩充，LoRA / QLoRA微调，全参数指令微调，奖励建模、强化学习训练。

## 🚀 更新日志

- [2023.08.18] [Llama2-13b-Chinese-chat](https://huggingface.co/carlAIwarts/Llama2-13b-Chinese-chat)发布🎉🎉🎉🎉；采用中英文开源指令微调数据
⏳ **Llama2-7b-Chinese-chat**: Llama2-7b-Chinese-chat正在飞奔而来！
⏳ **Llama2-Chinese**: 使用中英文语料对Llama 2进行增量预训练
同时，我们将会围绕 Llama2-Chinese 打造各种垂直领域模型

## 效果演示

## 🔧 模型微调

本仓库中提供了基于QLoRA的微调代码

我们的训练代码基于[FastChat](https://github.com/lm-sys/FastChat)开发.您可以使用以下命令用两张A100（80G）训练ToolLLaMA-7b, 训练数据是我们已经处理好的[数据]
```bash
export PYTHONPATH=./
deepspeed --master_port=20001 toolbench/train/train_long_seq_lora.py \
    --model_name_or_path huggyllama/llama-7b  \
    --data_path  data/toolllama_G123_dfs_train.json \
    --eval_data_path  data/toolllama_G123_dfs_eval.json \
    --conv_template tool-llama-single-round \
    --bf16 True \
    --output_dir toolllama_lora \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "epoch" \
    --prediction_loss_only \
    --save_strategy "epoch" \
    --save_total_limit 8 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --lazy_preprocess True \    
    --deepspeed ds_configs/stage2.json \
    --report_to none
```

## 数据清洗
* 仅中英文：使用fasttext的语言识别模型来按语言标记内容，仅留下中英文数据
* 网络数据：大多数开放语言模型（如Llama 1/2、Falcon、T5、MPT）都是基于预处理的网络文本的一个重要部分进行训练的。
* 质量过滤：网络爬取的数据中有很大一部分不适合语言模型训练（例如，格式不正确的文本、自动生成的网站文本）。这些通常通过“质量”过滤方法来移除，这里选择使用简单的启发式方法和正则表达式来过滤段落。这些过滤器的效果是去除从HTML转换为纯文本时产生的错误。
* 去重：最近的研究表明，数据去重可以使语言模型更有效地训练。遵循这一原则，我选择在每个来源中去重数据。
* 多样化的来源：像GPT-Neo或Pythia这样的模型（两者都在The Pile上训练）已经显示了在多样化的文档集上训练的重要性，如技术文档或生物医学文章。对于Dolma，AI2利用Semantic Scholar的语料库，包括来自peS2o的论文，这是一个包含38M可许可的科学手稿的子集。实践中：关于如何处理peS2o的更多细节可以在其主页上找到。对于维基百科，Dolma使用英文和简化英文子集。对于古腾堡计划中的书籍，筛选主要是英文的书籍。
* 去污染：因此，在准备Dolma时，AI2删除了训练文档中包含评估数据的内容。实践中：再次使用Bloom过滤器检查评估数据集中是否有任何超过13个令牌的段落出现在训练数据中。我们的去污染步骤通过字符删除了少于0.001%的训练数据，并影响了少于0.02%的文档。

## 数据发布

使用下面链接下载我们的数据集
- 


## 增量预训练数据

| 类型                                                       | 描述                                                         |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| [CLUECorpus2020](https://github.com/CLUEbenchmark/CLUECorpus2020) | 通过对Common Crawl的中文部分进行语料清洗，最终得到100GB的高质量中文预训练语料，可直接用于预训练、语言模型或语言生成任务以及专用于简体中文NLP任务的小词表。 |
| [Wikipedia](https://github.com/goldsmith/Wikipedia)        | 中文Wikipedia的数据                                          |
| [MNBVC（part）](https://github.com/esbatmop/MNBVC)                 | 超大规模中文语料集，不但包括主流文化，也包括各个小众文化甚至火星文的数据。MNBVC数据集包括新闻、作文、小说、书籍、杂志、论文、台词、帖子、wiki、古诗、歌词、商品介绍、笑话、糗事、聊天记录等一切形式的纯文本中文数据。数据均来源于互联网收集|

### SFT数据

| 类型                                                       | 描述                                                         |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| [ShareChat](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k) | 中英文平行双语优质人机问答数据集，覆盖真实复杂场景下的用户提问。 |
| [alpaca-gpt4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4)        | Alpaca-GPT-4 是一个使用 self-instruct 技术，基于 175 条中文种子任务和 GPT-4 接口生成的 50K 的指令微调数据集。                                          |
| [BELLE-data-1.5M](https://github.com/LianjiaTech/BELLE/tree/main/data/1.5M)        | 通过self-instruct生成，使用了中文种子任务，以及openai的text-davinci-003接口,涉及175个种子任务|
| [InstructionWild](https://github.com/XueFuzhao/InstructionWild)        | InstructionWild 是一个从网络上收集自然指令并过滤之后使用自然指令结合 ChatGPT 接口生成指令微调数据集的项目。主要的指令来源：Twitter、CookUp.AI、Github 和 Discard。|
| [COIG(part)](https://huggingface.co/datasets/BAAI/COIG)| 一套无害、有用且多样化的中文指令语料库，包括一个人工验证翻译的通用指令语料库、一个人工标注的考试指令语料库、一个人类价值对齐指令语料库、一个多轮反事实修正聊天语料库和一个 leetcode 指令语料库。|

### 垂直领域数据

| 类型                                                       | 描述                                                         |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| [Chinese medical dialogue data](https://github.com/Toyhom/Chinese-medical-dialogue-data) | 中文医疗对话数据集，包括：<Andriatria_男科> 94596个问答对 <IM_内科> 220606个问答对 <OAGD_妇产科> 183751个问答对 <Oncology_肿瘤科> 75553个问答对 <Pediatric_儿科> 101602个问答对 <Surgical_外科> 115991个问答对 总计 792099个问答对。 |
| [Huatuo-26M](https://github.com/FreedomIntelligence/Huatuo-26M)        | Huatuo-26M 是一个中文医疗问答数据集，此数据集包含了超过2600万个高质量的医疗问答对，涵盖了各种疾病、症状、治疗方式、药品信息等多个方面。 |
| [ToolBench](https://github.com/OpenBMB/ToolBench)        | ToolBench 包括单工具和多工具场景，从 RapidAPI 中获取 16,000 多个真实世界的API，并整理出涉及这些API的真实世界人类指令。多工具场景可进一步分为类别内多工具和集合内多工具。 |
| [moss-003-sft-plugin-data](https://github.com/OpenLMLab/MOSS/tree/main/SFT_data/conversations/conversation_with_plugins)        | 使用的插件增强的多轮对话数据，包含支持搜索引擎、文生图、计算器、解方程等四个插件在内的约30万条多轮对话数据。 |

### RLHF数据

| 类型                                                       | 描述                                                         |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) | 该项目开源了由GPT4生成的多种数据集，包括通过GPT4生成的中英PPO数据，可以用于奖励模型的训练。 |


## 模型下载

Meta在🤗Hugging Face上提供了所有模型的下载链接：https://huggingface.co/meta-llama

### Llama2 模型

Llama2 预训练模型包含7B、13B和70B三个版本；Llama2-Chat模型基于预训练模型进行了监督微调，具备更强的对话能力。

| 模型名称   | 🤗模型加载名称             | 下载地址                                                     |
| ---------- | ------------------------- | ------------------------------------------------------------ |
| Llama2-7B  | meta-llama/Llama-2-7b-hf  | [模型下载](https://huggingface.co/meta-llama/Llama-2-7b-hf)  |
| Llama2-13B | meta-llama/Llama-2-13b-hf | [模型下载](https://huggingface.co/meta-llama/Llama-2-13b-hf) |
| Llama2-70B | meta-llama/Llama-2-70b-hf | [模型下载](https://huggingface.co/meta-llama/Llama-2-70b-hf) |
| Llama2-7B-Chat  | meta-llama/Llama-2-7b-chat-hf  | [模型下载](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| Llama2-13B-Chat | meta-llama/Llama-2-13b-chat-hf | [模型下载](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) |
| Llama2-70B-Chat | meta-llama/Llama-2-70b-chat-hf | [模型下载](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) |
| Llama2-13b-Chinese-chat | carlAIwarts/Llama2-13b-Chinese-chat | [模型下载](https://huggingface.co/carlAIwarts/Llama2-13b-Chinese-chat) |

### 模型推理

**常用生成**
```python
from transformers import LlamaForCausalLM, LlamaTokenizer

model_path = 'carlAIwarts/Llama2-13b-Chinese-chat'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path)
prompt = "用中文介绍一下你自己"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.6,
    max_length=512,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)
```

**流式生成**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

model_path = 'carlAIwarts/Llama2-13b-Chinese-chat'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
prompt = "用中文介绍一下你自己"
generate_ids = model.generate(tokenizer(prompt, return_tensors='pt').input_ids.cuda(), max_new_tokens=4096, streamer=streamer)
```

### Gradio快速搭建问答平台

- [ ] 基于gradio搭建的问答界面，实现流式的输出

## 🏆 模型评测
为了能够更加清晰地了解Llama2模型的中文问答能力，我们实现了对应的评估数据和代码

| 类型                                                       | 描述                                                         |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| [C-Eval](https://github.com/SJTU-LIT/ceval) | 构造了一个覆盖人文，社科，理工，其他专业四个大方向，52 个学科（微积分，线代 …），从中学到大学研究生以及职业考试，一共 13948 道题目的中文知识和推理型测试集。此外还给出了当前主流中文LLM的评测结果。 |
| [MMCU](https://github.com/Felixgithub2017/MMCU) | 该项目提供对中文大模型语义理解能力的测试，评测方式、评测数据集、评测记录都公开，确保可以复现。该项目旨在帮助各位研究者们评测自己的模型性能，并验证训练策略是否有效。 |
| [MMLU](https://github.com/hendrycks/test) | 包含 57 个多选任务的英文评测数据集，涵盖了初等数学、美国历史、计算机科学、法律等，难度覆盖高中水平到专家水平 |

## 📚 Llama论文
* [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
* [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)

## 🤔 入群交流

加入微信群讨论；图片过期的话也欢迎加我的微信：carl_like_travel

<p align="center" width="60%">
<img src="./assets/wechat.jpeg" alt="Wechat" style="width: 60%; display: block; margin: auto;">
</p>
