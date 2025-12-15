import polars as pl
import numpy as np
import os
import sys
from datetime import date
from src.data_processing.data_loader import load_data, preprocess_data, merge_reservation_data, split_data_by_time
from src.data_processing.feature_engineering import create_features, encode_count, get_PrEp

def prepare_air_info(air_info):
    """处理AIR餐厅信息"""
    air_info = encode_count(air_info, "air_genre_name")
    air_info = encode_count(air_info, "air_area_name")
    return air_info

def create_air_datasets(data, air_info, date_info):
    """创建AIR相关的训练集和测试集"""
    # AIR访问数据
    df_train = data['air_visit'].clone()
    
    # 查看数据的时间范围
    min_date = df_train.select(pl.col('visit_date')).min().item()
    max_date = df_train.select(pl.col('visit_date')).max().item()
    print(f"AIR visit data date range: {min_date} to {max_date}")
    
    # 根据实际情况调整时间节点，确保有测试集数据
    # 根据dataset description，测试数据覆盖4月最后一周和5月
    t_begin = date(2017, 4, 15)  # 提前一周作为分割点
    print(f"Split date: {t_begin}")
    
    # 划分训练集和测试集
    air_train, air_test = split_data_by_time(df_train, t_begin)
    
    print(f"Training samples: {air_train.height}, Test samples: {air_test.height}")
    
    return air_train, air_test

def create_hpg_datasets(data, air_info, date_info):
    """创建HPG训练集和测试集"""
    # HPG预订数据
    hpg_reserve = data['hpg_reserve'].clone()
    
    # AIR访问数据（用于时间划分参考）
    air_visit = data['air_visit'].clone()
    
    # 使用AIR访问数据的时间划分原则对HPG数据进行划分
    t_begin = date(2017, 4, 15)
    
    # 构造HPG的训练和测试数据（通过聚合预订数据来模拟访问数据）
    hpg_full_data = construct_hpg_full_data(hpg_reserve, data['store_relation'], air_visit)
    
    # 添加时间差列
    hpg_full_data = hpg_full_data.with_columns(
        (pl.col('visit_date').str.strptime(pl.Date, "%Y-%m-%d") - pl.lit(t_begin)).dt.total_days().alias('day_gap')
    )
    
    # 划分训练集和测试集
    hpg_train = hpg_full_data.filter(pl.col('day_gap') < 0)
    hpg_test = hpg_full_data.filter(pl.col('day_gap') >= 0)
    
    # 添加标记
    hpg_train = hpg_train.with_columns([
        pl.lit(0).alias('is_air_only'),  # HPG数据不是纯AIR数据
        pl.lit(1).alias('is_train')
    ])
    
    hpg_test = hpg_test.with_columns([
        pl.lit(0).alias('is_air_only'),  # HPG数据不是纯AIR数据
        pl.lit(0).alias('is_train')
    ])
    
    return hpg_train, hpg_test

def construct_hpg_full_data(hpg_reserve, store_relation, air_visit):
    """根据HPG预订数据构造HPG完整数据集，并使用AIR访问数据作为标签"""
    # 合并store_relation以获得air_store_id
    hpg_with_air_id = hpg_reserve.join(store_relation, on='hpg_store_id', how='inner')
    
    # 获取AIR访问数据中的唯一商店和日期组合
    air_visit_unique = air_visit.select(['air_store_id', 'visit_date']).unique()
    
    # 只保留在AIR访问数据中存在的商店和日期组合
    hpg_filtered = hpg_with_air_id.join(
        air_visit_unique, 
        on=['air_store_id', 'visit_date'], 
        how='inner'
    )
    
    # 通过聚合预订数据来模拟访问数据
    hpg_visit_data = hpg_filtered.group_by(['air_store_id', 'visit_date']).agg([
        pl.col('reserve_visitors').sum().alias('reserve_visitors_sum')
    ])
    
    # 将AIR访问数据中的真实访客数作为标签
    hpg_visit_data = hpg_visit_data.join(
        air_visit.select(['air_store_id', 'visit_date', 'visitors']),
        on=['air_store_id', 'visit_date'],
        how='left'
    )
    
    # 对访客数量进行对数变换
    hpg_visit_data = hpg_visit_data.with_columns(
        pl.col('visitors').map_elements(
            lambda x: np.log1p(float(x)) if float(x) > 0 else 0
        ).alias('visitors')
    )
    
    return hpg_visit_data

def add_day_gap_to_reservations(data):
    """为预订数据添加day_gap列"""
    t_begin = date(2017, 4, 15)
    
    # 为AIR预订数据添加day_gap列
    air_reserve = data['air_reserve'].clone()
    air_reserve = air_reserve.with_columns(
        (pl.col('reserve_date').str.strptime(pl.Date, "%Y-%m-%d") - pl.lit(t_begin)).dt.total_days().alias('day_gap')
    )
    
    # 为HPG预订数据添加day_gap列
    hpg_reserve = data['hpg_reserve'].clone()
    hpg_reserve = hpg_reserve.with_columns(
        (pl.col('reserve_date').str.strptime(pl.Date, "%Y-%m-%d") - pl.lit(t_begin)).dt.total_days().alias('day_gap')
    )
    
    data['air_reserve'] = air_reserve
    data['hpg_reserve'] = hpg_reserve
    
    return data

def main():
    """主函数"""
    print("Loading data...")
    # 修复数据目录路径
    data = load_data('data')
    data = preprocess_data(data)
    
    print("Preparing restaurant info...")
    air_info = prepare_air_info(data['air_info'])
    
    print("Creating datasets...")
    air_train, air_test = create_air_datasets(
        data, air_info, data['date_info']
    )
    
    # 创建HPG训练和测试数据集
    hpg_train, hpg_test = create_hpg_datasets(
        data, air_info, data['date_info']
    )
    
    print("Saving intermediate datasets...")
    # 确保输出目录存在
    os.makedirs('outputs', exist_ok=True)
    
    # 保存数据集
    air_train.write_csv('outputs/air_train_intermediate.csv')
    air_test.write_csv('outputs/air_test_intermediate.csv')
    hpg_train.write_csv('outputs/hpg_train_intermediate.csv')
    hpg_test.write_csv('outputs/hpg_test_intermediate.csv')
    
    print("Done! Intermediate datasets saved to outputs/")
    print(f"AIR train shape: ({air_train.height}, {len(air_train.columns)})")
    print(f"AIR test shape: ({air_test.height}, {len(air_test.columns)})")
    print(f"HPG train shape: ({hpg_train.height}, {len(hpg_train.columns)})")
    print(f"HPG test shape: ({hpg_test.height}, {len(hpg_test.columns)})")

def extract_features():
    """提取特征的主函数"""
    print("Loading intermediate data...")
    air_train = pl.read_csv('outputs/air_train_intermediate.csv')
    air_test = pl.read_csv('outputs/air_test_intermediate.csv')
    hpg_train = pl.read_csv('outputs/hpg_train_intermediate.csv')
    hpg_test = pl.read_csv('outputs/hpg_test_intermediate.csv')
    
    # 加载原始数据
    data = load_data('data')
    data = preprocess_data(data)
    data = add_day_gap_to_reservations(data)  # 添加day_gap列到预订数据
    air_info = prepare_air_info(data['air_info'])
    date_info = data['date_info']
    PrEp = get_PrEp()
    
    print("Extracting features for AIR train set...")
    air_train_features = create_features_for_dataset(air_train, data, air_info, date_info, PrEp)
    air_train_features = air_train_features.with_columns([
        pl.lit(1).alias('is_air_only'),
        pl.lit(1).alias('is_train')
    ])
    
    print("Extracting features for AIR test set...")
    air_test_features = create_features_for_dataset(air_test, data, air_info, date_info, PrEp)
    air_test_features = air_test_features.with_columns([
        pl.lit(1).alias('is_air_only'),
        pl.lit(0).alias('is_train')
    ])
    
    print("Extracting features for HPG train set...")
    hpg_train_features = create_features_for_dataset(hpg_train, data, air_info, date_info, PrEp)
    # HPG数据的标记已经在main函数中设置过了，这里不再重复设置
    
    print("Extracting features for HPG test set...")
    hpg_test_features = create_features_for_dataset(hpg_test, data, air_info, date_info, PrEp)
    # HPG数据的标记已经在main函数中设置过了，这里不再重复设置
    
    print("Saving final feature datasets...")
    # 保存最终的特征数据集
    air_train_features.write_csv('outputs/air_train.csv')
    air_test_features.write_csv('outputs/air_test.csv')
    hpg_train_features.write_csv('outputs/hpg_train.csv')
    hpg_test_features.write_csv('outputs/hpg_test.csv')
    
    print("Done! Final feature datasets saved to outputs/")
    print(f"AIR train shape: ({air_train_features.height}, {len(air_train_features.columns)})")
    print(f"AIR test shape: ({air_test_features.height}, {len(air_test_features.columns)})")
    print(f"HPG train shape: ({hpg_train_features.height}, {len(hpg_train_features.columns)})")
    print(f"HPG test shape: ({hpg_test_features.height}, {len(hpg_test_features.columns)})")

def create_features_for_dataset(dataset, data, air_info, date_info, PrEp):
    """为数据集创建特征"""
    # 为整个数据集创建特征
    features = create_features(
        dataset, 
        data['air_visit'], 
        data['air_reserve'], 
        data['hpg_reserve'], 
        air_info, 
        date_info, 
        PrEp
    )
    return features

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "extract_features":
        extract_features()
    else:
        main()