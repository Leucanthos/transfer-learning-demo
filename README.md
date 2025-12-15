# Recruit餐厅访客预测项目

## 1. 项目概述

### 1.1 项目背景

本项目基于日本餐厅数据，利用迁移学习技术预测餐厅客流量。数据来源于Recruit Holdings公司旗下的两个平台：

- **Hot Pepper Gourmet (HPG)**: 类似于Yelp的餐厅点评服务，用户可以搜索餐厅并在线预订
- **AirREGI / Restaurant Board (AIR)**: 类似于Square的餐厅预订控制系统和收银系统

该项目旨在使用预订、访问和其他信息来预测未来特定日期的餐厅访客总数，这有助于餐厅更有效地采购食材和安排员工，提高运营效率。

### 11.2 核心预测问题

基于餐厅的历史访客记录、预订数据、日期特征（节假日/周末）、店铺属性（类型/区域）等信息，预测AIR系列餐厅未来的日访客数量，属于时间序列回归任务。

### 1.3 评估指标

提交结果使用均方根对数误差(RMSLE)进行评估：

$$\text{RMSLE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\log(p_i + 1) - \log(a_i + 1))^2}$$

其中：
- n 是总观察数
- p_i 是对访客的预测
- a_i 是实际访客数
- log(x) 是 x 的自然对数

## 2. 数据集描述

### 2.1 数据集构成

这是一个来自两个系统的关联数据集。每个文件都以来源前缀（air_ 或 hpg_）标明其来源。每个餐厅都有唯一的air_store_id和hpg_store_id。需要注意的是，并非所有餐厅都被两个系统覆盖，而且提供给你的数据超出了你必须预测的餐厅范围。为了防止餐厅去标识化，纬度和经度并不完全精确。

### 2.2 文件详细说明

#### air_reserve.csv
此文件包含在AIR系统中进行的预订。

- air_store_id - AIR系统中的餐厅ID
- visit_datetime - 预订时间
- reserve_datetime - 创建预订的时间
- reserve_visitors - 该预订的访客数量

#### hpg_reserve.csv
此文件包含在HPG系统中进行的预订。

- hpg_store_id - HPG系统中的餐厅ID
- visit_datetime - 预订时间
- reserve_datetime - 创建预订的时间
- reserve_visitors - 该预订的访客数量

#### air_store_info.csv
此文件包含选定AIR餐厅的信息。

- air_store_id
- air_genre_name
- air_area_name
- latitude
- longitude
> 注意：纬度和经度是该餐厅所属区域的纬度和经度

#### hpg_store_info.csv
此文件包含选定HPG餐厅的信息。

- hpg_store_id
- hpg_genre_name
- hpg_area_name
- latitude
- longitude
> 注意：纬度和经度是该餐厅所属区域的纬度和经度

#### store_id_relation.csv
此文件允许你连接同时拥有AIR和HPG系统的选定餐厅。

- hpg_store_id
- air_store_id

#### air_visit_data.csv
此文件包含AIR餐厅的历史访问数据。

- air_store_id
- visit_date - 日期
- visitors - 该日期餐厅的访客数量

#### sample_submission.csv
此文件显示了正确格式的提交，包括必须预测的日期。

- id - ID由air_store_id和visit_date用下划线连接而成
- visitors - 预测的商店和日期组合的访客数量

#### date_info.csv
此文件提供了数据集中日历日期的基本信息。

- calendar_date
- day_of_week
- holiday_flg - 该日在日本是否为假日

## 3. 项目架构与实施步骤

### 3.1 项目结构

```
.
├── data/
│   ├── inputs/              # 原始数据目录
│   ├── intermediates/       # 中间处理结果目录
│   └── outputs/             # 最终输出数据目录
├── src/
│   ├── data_transformation/ # 数据转换模块
│   └── model/               # 模型训练模块
├── lgbm_weights/            # LightGBM模型权重文件
├── RUN ME.ipynb             # 主执行脚本
├── solution.py              # 原始解决方案（已移至archive目录）
├── requirements.txt         # 项目依赖
└── README.md               # 项目说明文档
```

### 3.2 数据处理流程

#### 第一步：数据加载与预处理

通过 `src.data_transformation.load_inputs` 加载所有原始数据文件，包括AIR和HPG的预订数据、店铺信息、访问数据等。

#### 第二步：数据合并

通过 `src.data_transformation.merge_reservation` 将AIR和HPG的预订数据合并，形成统一的预订数据表。

#### 第三步：特征工程

通过 `src.data_transformation.add_features_pipeline` 执行完整的特征工程流程，包括：
- 日期特征提取（星期几、是否节假日等）
- 店铺属性特征（类型、区域等）
- 预订相关统计特征
- 历史访问统计特征（多种时间窗口）

#### 第四步：数据集划分

根据店铺是否在两个平台都有数据，将数据划分为两组：
- AIR-only数据：仅在AIR平台有数据的店铺
- HPG-AIR关联数据：在两个平台都有数据的店铺

#### 第五步：生成标准化输出

生成四个标准化的数据集文件（Parquet格式）：
1. **air_train.parquet**: AIR-only训练集
2. **air_test.parquet**: AIR-only测试集
3. **hpg_train.parquet**: HPG-AIR关联训练集
4. **hpg_test.parquet**: HPG-AIR关联测试集

## 4. 模型训练

### 4.1 模型架构

项目使用LightGBM作为基础模型，通过以下模块进行训练：

- `src.model.prepare_lgbm_dataset_with_weights`: 数据预处理，包含时间加权机制
- `src.model.train_or_load_lgbm`: 模型训练或加载
- `src.model.evaluate_model`: 模型评估

### 4.2 训练策略

1. **时间加权**: 使用day_gap作为权重依据，距离预测日期越近的样本权重越高
2. **早停机制**: 使用验证集进行早停，防止过拟合
3. **模型持久化**: 训练好的模型保存在lgbm_weights目录中

### 4.3 运行方法

在项目根目录下执行RUN ME.ipynb notebook，程序将自动完成以下步骤：
1. 数据加载与预处理
2. 特征工程
3. 模型训练
4. 模型评估

## 5. 技术实现细节

### 5.1 环境依赖

项目主要依赖以下Python库：
- polars: 高性能数据处理库
- numpy: 数值计算库
- scikit-learn: 机器学习库
- lightgbm: 梯度提升框架
- pandas: 数据分析库

### 5.2 核心代码组件

#### 数据转换模块 (src/data_transformation)
包含完整的数据处理流水线：
- load_data.py: 数据加载
- merge_reservation.py: 预订数据合并
- add_features目录: 特征工程实现

#### 模型模块 (src/model)
包含模型训练相关功能：
- baseline.py: 模型训练基础功能

## 6. 总结

本项目展示了如何使用迁移学习技术解决餐厅访客预测问题。通过精心设计的特征工程和时间加权训练策略，提升了模型在目标领域的预测性能。
