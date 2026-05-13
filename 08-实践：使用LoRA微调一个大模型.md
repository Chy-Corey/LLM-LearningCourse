## LoRA 微调实践：基于Qwen-7B的LoRA微调

[实战｜基于LoRA的开源大模型微调全流程：从环境搭建到效果验证 - AI设计栈 - 博客园](https://www.cnblogs.com/aitoolhub/p/19473148)

> 做不出来，没有足够的显卡

LoRA（Low-Rank Adaptation，低秩自适应）是目前大模型微调领域最主流、最实用的技术之一。简单来说，它的核心目标是**用极低的成本（显存和时间），让一个通用的大模型快速学会某个特定领域的知识或技能**。

如果把大模型比作一个拥有千亿参数的“学霸”，全量微调（Fine-tuning）相当于把这个学霸拉回学校，把他脑子里所有的知识重新学一遍，这不仅耗时，而且需要极其昂贵的算力（显存）。

LoRA 的做法则完全不同：**它不改动学霸原有的任何知识（冻结原始模型权重），而是给他发了一本小小的“专用笔记本”（低秩矩阵）**。

- 在训练时，我们只在这个“笔记本”上记录新任务的知识（比如医疗问诊的逻辑、法律条文的引用格式）。
- 在推理（使用）时，模型会同时参考“大脑”里的通用知识和“笔记本”上的专用知识来回答问题。

从数学角度看，假设模型某层的原始权重是 $W_0$ ，LoRA 并不直接修改它，而是增加两个非常小的矩阵 $A$ 和 $B$ 。训练时 $W_0$ 保持冻结，只更新 $A$ 和 $B$ ，最终的权重表现为 $W=W_0+BA$ 。这样，比起训练一整个 $W_0$，训练两个矩阵B和A所需要的参数量少了非常多。

LoRA之所以受欢迎，是因为：

1. **极低的硬件门槛**：LoRA 将需要训练的参数减少了数千倍（通常只占原始模型参数的不到 1%）。这意味着，原本需要 8 张 A100 显卡才能微调的模型，现在可能只需要 1 张消费级显卡（如 RTX 3090/4090）甚至笔记本显卡就能跑起来。
2. **无推理延迟**：训练完成后，“笔记本”上的内容可以和原始模型权重完美合并。在实际使用时，模型的大小和速度与原版完全一样，不会增加任何额外的等待时间。
3. **模块化与快速切换**：训练出来的 LoRA 文件非常小（通常只有几十 MB 到几百 MB）。你可以为同一个基座模型准备几十个不同的 LoRA（比如一个用于写代码，一个用于写小说），在使用时随时热切换，非常灵活。

### 一、环境配置

#### 1. 依赖介绍

核心依赖库：PyTorch（深度学习框架）、Transformers（模型加载）、Peft（LoRA实现）、Dataset（数据处理）、 accelerate（分布式训练支持）。

PyTorch 是由 Meta（Facebook）推出的开源深度学习框架，是整个技术栈的**底层计算引擎**。

- **核心作用**：它提供了基础的张量（Tensor）运算和自动微分功能。简单来说，神经网络里的所有矩阵计算、梯度反向传播，都是由 PyTorch 在幕后完成的。
- **为什么用它**：PyTorch 采用**动态计算图**机制，代码写起来非常符合 Python 的直觉，调试起来就像调试普通 Python 代码一样方便。目前绝大多数的开源大模型（如 LLaMA、Qwen 等）和微调工具，底层都是基于 PyTorch 构建的。

Transformers 是由 Hugging Face 公司开发的库，它是目前 NLP（自然语言处理）领域**事实上的工业标准**。

- **核心作用**：它提供了一个统一的接口（如 `AutoModel`），让你能用**一行代码**加载成千上万种预训练大模型（BERT, GPT, LLaMA 等）以及配套的分词器（Tokenizer）。
- **为什么用它**：如果没有它，你需要为每种模型编写复杂的加载代码。有了它，你只需指定模型名称（如 `"qwen/Qwen2.5-7B"`），它就能自动从 Hugging Face Hub 下载模型架构和权重，并组装成可以直接运行的 PyTorch 模型。

Peft (Parameter-Efficient Fine-Tuning) 同样出自 Hugging Face，是专门用来做**高效微调**的工具库。

- **核心作用**：它封装了 LoRA、QLoRA 等微调算法。在微调时，你只需要通过 Peft 提供的 `LoraConfig` 配置好参数（比如秩 `r` 的大小、要微调的目标模块），然后调用 `get_peft_model`，它就会自动把你的基础模型“包裹”起来，**冻结原始模型权重，只插入并训练微小的 LoRA 适配器矩阵**。
- **为什么用它**：它让你无需手动去修改模型的底层代码或冻结参数，极大地简化了 LoRA 的实现流程。

Datasets 也是 Hugging Face 生态的一员，专门负责**数据的加载与预处理**。

- **核心作用**：无论是本地的 CSV、JSON 文件，还是云端的大型开源数据集，它都能快速加载。它支持对数据进行高效的映射（map）、过滤（filter）、分词处理以及分批（batch）。
- **为什么用它**：大模型微调需要处理海量文本，Datasets 库底层做了大量内存优化（如内存映射），即使你的数据集有几百 GB，它也能在不撑爆内存的情况下进行快速读取和处理。

Accelerate 依然是 Hugging Face 推出的库，它的目标是**让 PyTorch 代码在任何设备上都能轻松并行运行**。

- **核心作用**：在微调大模型时，单张显卡的显存往往不够。Accelerate 能够自动处理底层的分布式逻辑（如数据并行、模型并行、DeepSpeed 集成等）。你只需要在代码中加入几行 `accelerator.prepare()`，原本只能在单卡上运行的 PyTorch 训练脚本，就可以无缝切换到多卡、多机甚至 TPU 上运行。
- **为什么用它**：它屏蔽了复杂的分布式训练配置，让你不用重写大量代码就能利用所有显卡的性能，同时它还内置了混合精度训练（FP16/BF16）和梯度累积等显存优化技术。

**总结一下它们的工作流：**
你用 **Datasets** 准备好训练语料，通过 **Transformers** 加载预训练好的大模型，利用 **Peft** 给模型加上 LoRA 适配器，最后依靠 **PyTorch** 进行底层的张量计算，并由 **Accelerate** 调度多张显卡来加速整个训练过程。

#### 2. 安装依赖

在终端中输入以下命令安装依赖：

```bash
# 创建虚拟环境
conda create -n lora-finetune python=3.10
conda activate lora-finetune

# 安装PyTorch（根据显卡型号选择，此处以CUDA 13.0为例）
# nvidia-smi 可查看cuda版本
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 安装核心依赖
pip install transformers==4.38.2 peft==0.8.2 datasets==2.18.0 accelerate==0.30.0
pip install sentencepiece==0.1.99 tokenizers==0.15.2 pandas numpy
```

测试依赖环境：

```python
import torch
from peft import LoraConfig

# 验证CUDA是否可用
print(f"CUDA可用: {torch.cuda.is_available()}")
# 验证Peft是否正常导入
print(f"LoRA配置类加载成功: {LoraConfig.__name__}")
```

若输出`“CUDA可用: True”`和`“LoRA配置类加载成功: LoraConfig”`，则环境搭建完成。

### 二、数据加载

使用**Magpie-Qwen2**数据集，Magpie-Qwen2 是一个专门为指令微调设计的高质量合成数据集，它的核心亮点在于“由强大的 72B 模型生成并经过严格过滤”，这保证了数据的纯净度和逻辑深度。使用该数据集可以做到：

- **提升模型的“智商”与逻辑性**：由于是由 72B 级别的大模型生成的，数据中蕴含了非常严密的思维链（CoT）和推理过程。用它微调小模型（如 Qwen2-7B 或 Llama3-8B），可以显著提升小模型在复杂问答、逻辑推理和事实性回复上的表现。
- **低成本复刻大模型能力**：对于无法直接使用 72B 超大模型的团队，通过 Magpie-Qwen2 进行指令微调，可以让你的 7B 或 14B 模型在对话能力上大幅逼近超大模型的水准。

加载方式和其他 Hugging Face 数据集完全一致。可以直接通过以下代码加载它的不同版本：

```python
from datasets import load_dataset

# 加载 Magpie-Qwen2 的 200K 高质量版本
dataset_200k = load_dataset(
    "Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese", 
    cache_dir="./model/huggingface_cache",
    split="train")

# 或者加载 1M 的大规模版本（如果存在）
# dataset_1m = load_dataset("Magpie-Align/Magpie-Qwen2-Pro-1M", split="train")

# 查看第一条数据的结构
print(dataset_200k[0])
```

> 待定

### 三、模型加载 & LoRA配置

加载Qwen-7B模型（采用INT4量化版本，降低显存占用），并配置LoRA参数。

#### 1. 加载量化模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# 模型路径
local_cache_path = "./model/qwen-7b" 
# 量化配置：采用4-bit量化，降低显存占用
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载Qwen-7B模型和Tokenizer（需提前在Hugging Face注册并同意协议）
model_name = "Qwen/Qwen-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    cache_dir=local_cache_path,
    trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=local_cache_path,
    quantization_config=bnb_config,
    device_map="auto",  # 自动分配设备（GPU/CPU）
    trust_remote_code=True
)
model.config.use_cache = False  # 禁用缓存，避免训练时出错
model.config.pretraining_tp = 1
```

#### 2. 配置LoRA参数

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,  # LoRA秩，越小显存占用越少，推荐8-32
    lora_alpha=32,  # 缩放系数，通常为r的2-4倍
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen模型的Attention层关键模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"  # 因果语言建模（生成任务）
)

# 将LoRA配置注入模型
model = get_peft_model(model, lora_config)
# 查看模型参数总量和可训练参数量
model.print_trainable_parameters()
```

输出示例：`trainable params: 1,179,648 || all params: 7,249,483,776 || trainable%: 0.0163`，可见仅训练0.016%的参数，显存压力极大降低。
