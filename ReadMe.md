# 中文古诗语言模型 - 基于 LSTM 和 GRU

该项目使用 PyTorch 构建了一个字符级别的语言模型，旨在训练和评估基于《唐诗》文本数据的语言模型。模型支持 LSTM 和 GRU 两种结构，并通过困惑度（Perplexity）来评估模型性能。

## 功能

- 加载并处理《唐诗》文本数据
- 构建字符级别的 LSTM 和 GRU 语言模型
- 训练模型并计算训练损失
- 评估模型性能并计算困惑度

## 环境要求

- Python >= 3.7
- PyTorch >= 1.7
- CUDA（如果需要 GPU 加速）

### 依赖包

- torch
- numpy

安装依赖包：

```bash
pip install torch numpy

数据
使用《唐诗》文本数据进行训练。文件名为 poetryFromTang.txt，需放置在项目的根目录中。

## 项目结构

项目根目录/
│
├── poetryFromTang.txt     // 训练数据文件
├── main.py                // 主程序文件，包含模型定义、训练和评估代码
├── DEPLOYMENT.md          // 部署文档
├── requirements.txt       // 环境依赖
└── README.md              // 项目说明文档

## 安装和运行

安装步骤

克隆项目代码：

git clone https://github.com/ljhawsome666/finalproject-nlp-hust.git
进入项目目录：

cd yourproject
安装依赖包：

pip install -r requirements.txt

## 运行模型训练

python main.py

## 使用说明

调整 main.py 文件中的参数：

batch_size
seq_length
n_hidden
n_layers
epochs
lr
model_type

训练过程中将输出训练和验证损失以及困惑度。