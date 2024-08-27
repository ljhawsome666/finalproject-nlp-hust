# 部署文档

本文件描述了如何在本地或服务器上部署和运行中文古诗语言模型项目。

## 前提条件

- 确保已安装 Python 3.7 或更高版本
- 确保已安装 `pip` 包管理器

## 部署步骤

### 1. 克隆代码仓库

首先，使用以下命令克隆项目代码到本地：

```bash
git clone https://github.com/ljhawsome666/finalproject-nlp-hust.git
cd yourproject

### 2. 准备数据

将 poetryFromTang.txt 文件放置在项目的根目录中。该文件包含了训练所需的《唐诗》文本数据。

### 3. 运行训练脚本

使用以下命令运行模型的训练脚本：
python main.py
