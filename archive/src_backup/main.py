import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datetime import date
from data_processing.data_loader import load_data, preprocess_data
from data_processing.feature_engineering import create_features, encode_count

def prepare_air_info(air_info):
    """处理AIR餐厅信息"""
    air_info = encode_count(air_info, "air_genre_name")
    air_info = encode_count(air_info, "air_area_name")
    return air_info

def get_PrEp():
    """获取核函数矩阵"""
    PrOriginalEp = np.zeros((2000, 2000))
    PrOriginalEp[1, 0] = 1
    PrOriginalEp[2, range(2)] = [0.5, 0.5]
    for i in range(3, 2000):
        scale = (i - 1) / 2.
        x = np.arange(-(i + 1) / 2. + 1, (i + 1) / 2., step=1) / scale
        y = 3. / 4. * (1 - x ** 2)
        y = y / np.sum(y)
        PrOriginalEp[i, range(i)] = y
    PrEp = PrOriginalEp.copy()
    for i in range(3, 2000):
        PrEp[i, :i] = (PrEp[i, :i] * i + 1) / (i + 1)
    return PrEp

def create_air_datasets(data, air_info, date_info, PrEp):
    """创建AIR相关的训练集和测试集"""
    # AIR访问数据
    df_train = data['air_visit'].copy()
    
    # AIR预订数据
    air_reserve = data['air_reserve'].copy()
    
    # HPG预订数据（通过关联关系转换为AIR视角）
    hpg_reserve = data['hpg_reserve'].copy()
    
    # 查看数据的时间范围
    print(f"AIR visit data date range: {df_train['visit_date'].min()} to {df_train['visit_date'].max()}")
    
    # 根据实际情况调整时间节点，确保有测试集数据
    # 根据dataset description，测试数据覆盖4月最后一周和5月
    t_begin = date(2017, 4, 15)  # 提前一周作为分割点
    print(f"Split date: {t_begin}")
    
    # 计算时间间隔
    df_train["day_gap"] = df_train["visit_date"].apply(lambda x: date_gap(x, t_begin))
    air_reserve["day_gap"] = air_reserve["reserve_date"].apply(lambda x: date_gap(x, t_begin))
    hpg_reserve["day_gap"] = hpg_reserve["reserve_date"].apply(lambda x: date_gap(x, t_begin))
    
    # 划分训练集和测试集
    air_train = df_train[df_train.day_gap < 0].copy()
    air_test = df_train[df_train.day_gap >= 0].copy()
    
    print(f"Training samples: {len(air_train)}, Test samples: {len(air_test)}")
    
    air_reserve_train = air_reserve[air_reserve.day_gap < 0].copy()
    air_reserve_test = air_reserve[air_reserve.day_gap >= 0].copy()
    
    hpg_reserve_train = hpg_reserve[hpg_reserve.day_gap < 0].copy()
    hpg_reserve_test = hpg_reserve[hpg_reserve.day_gap >= 0].copy()
    
    # 为训练集创建特征
    print("Creating features for AIR training set...")
    air_train_features = create_features(
        air_train, air_train, air_reserve_train, hpg_reserve_train,
        air_info, date_info, PrEp
    )
    
    # 为测试集创建特征
    print("Creating features for AIR testing set...")
    air_test_features = create_features(
        air_test, air_train, air_reserve_train, hpg_reserve_train,
        air_info, date_info, PrEp
    )
    
    return air_train_features, air_test_features

def create_hpg_dataset(data, air_info, date_info, PrEp):
    """创建HPG训练集"""
    # 处理HPG访问数据
    hpg_visit = data['hpg_visit'].copy() if 'hpg_visit' in data else None
    if hpg_visit is None:
        # 如果没有单独的HPG访问数据，则使用HPG预订数据中的相关信息
        # 这里我们假设可以从预订数据中推断一些访问模式
        hpg_visit = pd.DataFrame()
    
    # HPG预订数据
    hpg_reserve = data['hpg_reserve'].copy()
    
    # AIR预订数据（用于对比）
    air_reserve = data['air_reserve'].copy()
    
    # 使用全部HPG数据作为训练集
    t_begin = date(2017, 4, 15)
    hpg_reserve["day_gap"] = hpg_reserve["reserve_date"].apply(lambda x: date_gap(x, t_begin))
    air_reserve["day_gap"] = air_reserve["reserve_date"].apply(lambda x: date_gap(x, t_begin))
    
    # 选择时间范围内的数据
    hpg_reserve_train = hpg_reserve[hpg_reserve.day_gap < 0].copy()
    air_reserve_train = air_reserve[air_reserve.day_gap < 0].copy()
    
    # 构造HPG的训练数据（这里简化处理，实际应该有HPG的真实访问数据）
    # 我们使用一个模拟方法来构造训练数据
    hpg_train = construct_hpg_training_data(hpg_reserve_train, data['store_relation'])
    
    # 为HPG训练集创建特征
    print("Creating features for HPG training set...")
    hpg_train_features = create_features(
        hpg_train, hpg_train, air_reserve_train, hpg_reserve_train,
        air_info, date_info, PrEp
    )
    
    return hpg_train_features

def construct_hpg_training_data(hpg_reserve, store_relation):
    """根据HPG预订数据构造HPG训练数据"""
    # 合并store_relation以获得air_store_id
    hpg_with_air_id = hpg_reserve.merge(store_relation, on='hpg_store_id', how='inner')
    
    # 通过聚合预订数据来模拟访问数据，这里应该按hpg_store_id和visit_date分组
    hpg_visit_data = hpg_with_air_id.groupby(['hpg_store_id', 'visit_date'])['reserve_visitors'].sum().reset_index()
    hpg_visit_data.columns = ['hpg_store_id', 'visit_date', 'visitors']
    # 对访客数量进行对数变换
    hpg_visit_data['visitors'] = hpg_visit_data['visitors'].apply(
        lambda x: np.log1p(float(x)) if float(x) > 0 else 0
    )
    
    # 添加air_store_id等信息：通过再次合并store_relation表
    hpg_visit_data = hpg_visit_data.merge(store_relation, on='hpg_store_id', how='left')
    hpg_visit_data['day_gap'] = 0  # 占位符，实际会在特征工程中重新计算
    
    return hpg_visit_data

def date_gap(x, y):
    """计算日期间隔天数"""
    a, b, c = x.split("-")
    return (date(int(a), int(b), int(c)) - y).days

def main():
    """主函数"""
    print("Loading data...")
    data = load_data('data')
    data = preprocess_data(data)
    
    print("Preparing restaurant info...")
    air_info = prepare_air_info(data['air_info'])
    
    print("Getting PrEp matrix...")
    PrEp = get_PrEp()
    
    print("Creating datasets...")
    air_train, air_test = create_air_datasets(
        data, air_info, data['date_info'], PrEp
    )
    
    # 注意：由于原始数据中没有HPG的访问数据，我们通过预订数据模拟
    hpg_train = create_hpg_dataset(
        data, air_info, data['date_info'], PrEp
    )
    
    print("Saving datasets...")
    # 确保输出目录存在
    os.makedirs('outputs', exist_ok=True)
    
    # 保存数据集
    air_train.to_csv('outputs/air_train.csv', index=False)
    air_test.to_csv('outputs/air_test.csv', index=False)
    hpg_train.to_csv('outputs/hpg_train.csv', index=False)
    
    print("Done! Datasets saved to outputs/")
    print(f"AIR train shape: {air_train.shape}")
    print(f"AIR test shape: {air_test.shape}")
    print(f"HPG train shape: {hpg_train.shape}")

if __name__ == "__main__":
    main()