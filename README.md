# Transfer Learning Demo

本项目旨在演示如何在表格数据上应用各种迁移学习技术，特别关注从AIR平台到HPG平台的客流预测任务。

## 项目结构

```
.
├── data/                          # 原始数据目录
│   └── outputs/                   # 处理后的特征数据
├── lgbm_weights/                  # 预训练模型权重
├── src/                           # 数据处理模块
│   ├── data_transformation/       # 数据转换模块
│   │   ├── add_features/          # 特征工程模块
│   │   ├── data_cleansing.py      # 数据清洗
│   │   └── load_data.py           # 数据加载
│   └── __init__.py
├── transfer_learning/             # 迁移学习核心算法模块
│   ├── base.py                    # 基础模型类
│   ├── baseline.py                # 基线方法
│   ├── transfer.py                # 基础迁移学习方法
│   ├── ensemble_transfer.py       # 集成迁移学习方法
│   ├── advanced_transfer.py       # 高级迁移学习方法
│   ├── super_ensemble.py          # 超级集成迁移学习方法
│   └── __init__.py
├── experiments/                   # 实验脚本和可视化
│   ├── run_all_experiments_with_visualization.py  # 运行所有实验并可视化
│   ├── run_failed_experiments_only.py             # 仅运行失败的实验
│   ├── run_finetune_experiments.py               # 运行微调实验
│   └── __init__.py
├── RUN ME.ipynb                   # 快速入门Jupyter Notebook
└── requirements.txt               # 项目依赖
```

## 迁移学习方法

1. **伪标签迁移学习** - 使用源域模型为目标域数据生成伪标签
2. **微调迁移学习** - 加载预训练模型并在目标域上进行微调
3. **对抗域适应** - 通过对抗训练减少源域和目标域之间的差异
4. **超级集成方法** - 结合多种技术的综合迁移方法

## 快速开始

1. 安装依赖:
   ```
   pip install -r requirements.txt
   ```

2. 运行数据处理脚本:
   ```
   python main.py
   ```

3. 运行迁移学习实验:
   ```
   python experiments/run_all_experiments_with_visualization.py
   ```

## 可视化

运行实验脚本后，将在项目根目录生成多种可视化图表:
- 预测值与实际值对比图
- 特征重要性图
- 方法性能对比图

# Transfer Learning Demo for Restaurant Visitor Forecasting

## 项目背景

本项目基于日本餐厅的客流量数据，利用迁移学习技术解决HPG平台因缺乏真实客流标签而难以准确预测的问题。通过从拥有丰富真实客流数据的AIR平台向HPG平台迁移知识，提升目标平台的预测性能。

Recruit Holdings公司运营两个主要平台：
- **Hot Pepper Gourmet (HPG)**: 类似于Yelp的餐厅点评服务，用户可以搜索餐厅并在线预订
- **AirREGI / Restaurant Board (AIR)**: 类似于Square的餐厅预订控制系统和收银系统

HPG平台仅有预订数据，缺少真实的客流量标签，难以直接训练高质量预测模型；而AIR平台具有完整的历史客流记录，可用于构建强特征表示和预训练模型。本项目旨在探索如何有效将AIR平台学到的数据分布和特征模式迁移到HPG平台。

## 安装和依赖

### 环境要求
- Python 3.8+

### 安装依赖
```bash
pip install -r requirements.txt
```

### 项目依赖
- polars==1.32.2
- numpy==1.26.4
- scikit-learn==1.3.0
- lightgbm==4.0.0
- pandas==1.5.3

## 数据集描述

本项目使用来自AIR和HPG两个平台的数据，包括：

### AIR平台数据
- `air_visit_data.parquet`: AIR餐厅的历史访问数据
- `air_reserve.parquet`: AIR系统的预订数据
- `air_store_info.parquet`: AIR餐厅信息

### HPG平台数据
- `hpg_reserve.parquet`: HPG系统的预订数据
- `hpg_store_info.parquet`: HPG餐厅信息

### 关联数据
- `store_id_relation.parquet`: AIR和HPG平台餐厅ID映射关系
- `date_info.parquet`: 日期相关信息（节假日等）
- `sample_submission.csv`: 提交样例文件

## 项目详细介绍

### 数据预处理流程

1. **数据加载**: 从Parquet文件中加载原始数据
2. **数据清洗**: 处理缺失值和异常值
3. **特征工程**: 构造时间序列特征、统计特征等
4. **数据分割**: 按时间划分训练集和测试集

### 基线模型

使用LightGBM构建基线模型，仅使用HPG平台自身数据进行训练。

### 迁移学习方法

实现了多种迁移学习方法：

1. **微调(Fine-tuning)**: 加载AIR平台预训练模型，在HPG数据上继续训练
2. **伪标签(Pseudo-labeling)**: 使用AIR模型为HPG数据生成伪标签，混合训练
3. **样本选择迁移(Direct Transfer with Sample Selection)**: 从AIR数据中选择与HPG相似的样本进行迁移
4. **对抗域适应(Adversarial Domain Adaptation)**: 使用对抗训练减少域间差异
5. **超级集成(Super Ensemble)**: 结合多种技术的综合迁移方法

### 实验结果

各种方法在测试集上的表现(RMSLE)：

| 方法 | RMSLE |
|------|-------|
| 基线模型 | ~0.55 |
| 微调 | ~0.54 |
| 伪标签 | ~0.55 |
| 样本选择迁移 | ~0.53 |
| 对抗域适应 | ~0.54 |
| 超级集成 | ~0.52 |

## 运行说明

### 数据预处理
```bash
python main.py
```

### 运行所有实验
```bash
python run_all_experiments.py
```

### 运行特定实验
```bash
# 运行失败的实验
python run_failed_experiments_only.py

# 运行超参数调优实验
python run_hyperparameter_tuning.py

# 运行微调实验
python run_finetune_experiments.py
```

## 项目结构

```
├── data/
│   ├── inputs/              # 原始数据文件
│   └── outputs/             # 处理后的数据文件
├── lgbm_weights/           # LightGBM模型权重文件
├── src/
│   ├── data_transformation/ # 数据预处理模块
│   └── model/              # 模型实现模块
├── run_*.py                # 各种实验脚本
└── README.md
```

## 总结

通过本项目实践发现：
1. 迁移学习确实能够提升HPG平台的预测性能
2. 简单的微调方法就能取得不错的效果
3. 结合多种技术的超级集成方法效果最佳
4. 样本选择是一种有效的迁移策略
