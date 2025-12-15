import pandas as pd
import numpy as np
from datetime import date, timedelta
import os

def load_data(data_dir='../data'):
    """
    加载所有原始数据文件
    
    Returns:
        dict: 包含所有数据表的字典
    """
    # 获取当前脚本所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建数据目录的绝对路径
    abs_data_dir = os.path.normpath(os.path.join(current_dir, '..', '..', data_dir))
    
    data = {}
    
    # 加载AIR相关数据
    data['air_visit'] = pd.read_csv(os.path.join(abs_data_dir, 'air_visit_data.csv'))
    data['air_reserve'] = pd.read_csv(os.path.join(abs_data_dir, 'air_reserve.csv'))
    data['air_info'] = pd.read_csv(os.path.join(abs_data_dir, 'air_store_info.csv'))
    
    # 加载HPG相关数据
    data['hpg_reserve'] = pd.read_csv(os.path.join(abs_data_dir, 'hpg_reserve.csv'))
    data['hpg_info'] = pd.read_csv(os.path.join(abs_data_dir, 'hpg_store_info.csv'))
    
    # 加载其他数据
    data['date_info'] = pd.read_csv(os.path.join(abs_data_dir, 'date_info.csv'))
    data['store_relation'] = pd.read_csv(os.path.join(abs_data_dir, 'store_id_relation.csv'))
    data['sample_submission'] = pd.read_csv(os.path.join(abs_data_dir, 'sample_submission.csv'))
    
    return data

def preprocess_data(data):
    """
    对原始数据进行预处理
    
    Args:
        data (dict): 原始数据字典
        
    Returns:
        dict: 预处理后的数据字典
    """
    # 对访客数量进行对数变换
    data['air_visit']['visitors'] = data['air_visit']['visitors'].apply(
        lambda x: np.log1p(float(x)) if float(x) > 0 else 0
    )
    
    # 处理sample_submission，提取air_store_id和visit_date
    data['sample_submission']['air_store_id'] = data['sample_submission']['id'].apply(
        lambda x: '_'.join(x.split('_')[:2])
    )
    data['sample_submission']['visit_date'] = data['sample_submission']['id'].apply(
        lambda x: x.split('_')[2]
    )
    
    # 处理预订数据的时间信息
    # AIR预订数据
    data['air_reserve']['reserve_date'] = data['air_reserve']['reserve_datetime'].apply(
        lambda x: x.split(' ')[0]
    )
    data['air_reserve']['visit_date'] = data['air_reserve']['visit_datetime'].apply(
        lambda x: x.split(' ')[0]
    )
    data['air_reserve']['reserve_datetime_diff'] = (
        pd.to_datetime(data['air_reserve']['visit_date']) - 
        pd.to_datetime(data['air_reserve']['reserve_date'])
    ).dt.days
    
    # HPG预订数据
    data['hpg_reserve']['reserve_date'] = data['hpg_reserve']['reserve_datetime'].apply(
        lambda x: x.split(' ')[0]
    )
    data['hpg_reserve']['visit_date'] = data['hpg_reserve']['visit_datetime'].apply(
        lambda x: x.split(' ')[0]
    )
    data['hpg_reserve']['reserve_datetime_diff'] = (
        pd.to_datetime(data['hpg_reserve']['visit_date']) - 
        pd.to_datetime(data['hpg_reserve']['reserve_date'])
    ).dt.days
    
    # 合并store_relation到air_visit和hpg_reserve
    data['air_visit'] = data['air_visit'].merge(
        data['store_relation'], on='air_store_id', how='left'
    )
    data['hpg_reserve'] = data['hpg_reserve'].merge(
        data['store_relation'], on='hpg_store_id', how='inner'
    )
    
    # 处理日期信息
    data['date_info'].columns = ['visit_date', 'day_of_week', 'holiday_flg']
    data['date_info']['day_of_week'] = data['date_info']['day_of_week'].replace({
        'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 
        'Friday': 5, 'Saturday': 6, 'Sunday': 7
    })
    data['date_info']['holiday'] = (
        (data['date_info']['day_of_week'] >= 6) | 
        (data['date_info']['holiday_flg'] == 1)
    ).astype(int)
    
    return data