# 餐厅客流量预测项目

本项目基于日本餐厅数据，利用迁移学习技术预测餐厅客流量。

## 项目结构

```
.
├── data/                    # 原始数据目录
├── src/                     # 源代码目录
│   └── data_processing/     # 数据处理模块
│       ├── data_loader.py   # 数据加载模块
│       └── feature_engineering.py  # 特征工程模块
├── outputs/                 # 输出数据目录
├── main.py                  # 主程序入口（移至项目根目录）
├── requirements.txt         # 项目依赖
└── 项目实施步骤.md          # 项目实施步骤说明
```

## 数据说明

数据来源于两个平台：
- AIR平台：包含真实的客流量数据
- HPG平台：仅包含预订数据

我们的目标是利用AIR平台丰富的客流量真实数据，通过迁移学习技术帮助HPG平台提升客流预测准确性。

## 运行环境

推荐使用Python 3.8+版本。

安装依赖包：
```bash
pip install -r requirements.txt
```

## 运行方法

在项目根目录下执行：
```bash
python main.py
```

程序将自动完成以下步骤：
1. 加载并预处理数据
2. 进行特征工程
3. 按时间划分训练集和测试集
4. 生成标准化的中间产物文件

输出文件位于[outputs/](file://d:/Homework%20Collection/Fudan%20Grad/S1-%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E4%B8%8E%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E5%A4%A7%E4%BD%9C%E4%B8%9A/outputs/)目录下：
- air_train.csv: AIR平台训练集
- air_test.csv: AIR平台测试集
- hpg_train.csv: HPG平台训练集（基于预订数据构建）

## 迁移学习策略

我们将采用从AIR到HPG的迁移学习策略：
1. AIR平台拥有真实的客流量数据，数据质量更高
2. AIR平台数据时间跨度更长，特征更丰富
3. 部分餐厅仅在AIR平台存在，具有独特性价值

通过迁移学习，我们可以将在AIR平台学到的知识迁移到HPG平台，提升HPG平台的客流预测能力。