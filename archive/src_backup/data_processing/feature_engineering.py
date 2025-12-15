import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn import preprocessing

def encode_count(df, column_name):
    """
    对类别特征进行Label Encoding
    
    Args:
        df (DataFrame): 输入数据
        column_name (str): 需要编码的列名
        
    Returns:
        DataFrame: 编码后的数据
    """
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df[column_name].values))
    df[column_name] = lbl.transform(list(df[column_name].values))
    return df

def feat_sum(df, df_feature, fe, value, name=""):
    """
    计算特征的总和
    """
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].sum()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_sum" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_mean(df, df_feature, fe, value, name=""):
    """
    计算特征的均值
    """
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].mean()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_count(df, df_feature, fe, value, name=""):
    """
    计算特征的计数
    """
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].count()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_count" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_median(df, df_feature, fe, value, name=""):
    """
    计算特征的中位数
    """
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].median()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_median" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_max(df, df_feature, fe, value, name=""):
    """
    计算特征的最大值
    """
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].max()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_max" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_min(df, df_feature, fe, value, name=""):
    """
    计算特征的最小值
    """
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].min()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_min" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_std(df, df_feature, fe, value, name=""):
    """
    计算特征的标准差
    """
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].std()).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_std" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def date_gap(x, y):
    """
    计算日期间隔天数
    """
    a, b, c = x.split("-")
    return (date(int(a), int(b), int(c)) - y).days

def date_handle(df, air_info, date_info):
    """
    处理日期相关特征
    """
    df_visit_date = pd.to_datetime(df["visit_date"])
    df["weekday"] = df_visit_date.dt.weekday
    df["day"] = df_visit_date.dt.day
    days_of_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    df["days_to_side"] = df_visit_date.apply(
        lambda x: min(x.day, days_of_months[x.month-1]-x.day)
    )
    df["day"] = df["day"].apply(lambda x: 0 if x <= 7 else 2 if x >= 24 else 1)
    df = df.merge(air_info, on="air_store_id", how="left").fillna(-1)
    df = df.merge(date_info, on="visit_date", how="left").fillna(-1)
    return df

def create_features(df_label, df_train, df_air_reserve, df_hpg_reserve, 
                   air_info, date_info, PrEp):
    """
    创建特征，与solution.py中方法一致
    """
    df_train = date_handle(df_train, air_info, date_info)
    df_label = date_handle(df_label, air_info, date_info)

    # 预定信息特征
    df_label = feat_sum(df_label, df_air_reserve, ["air_store_id", "visit_date"], 
                       "reserve_datetime_diff", "air_reserve_datetime_diff_sum")
    df_label = feat_mean(df_label, df_air_reserve, ["air_store_id", "visit_date"], 
                        "reserve_datetime_diff", "air_reserve_datetime_diff_mean")
    df_label = feat_sum(df_label, df_air_reserve, ["air_store_id", "visit_date"], 
                       "reserve_visitors", "air_reserve_visitors_sum")
    df_label = feat_mean(df_label, df_air_reserve, ["air_store_id", "visit_date"], 
                        "reserve_visitors", "air_reserve_visitors_mean")
    df_label = feat_sum(df_label, df_air_reserve, ["visit_date"], 
                       "reserve_visitors", "air_date_reserve_visitors_sum")
    df_label = feat_mean(df_label, df_air_reserve, ["visit_date"], 
                        "reserve_visitors", "air_date_reserve_visitors_mean")

    df_label = feat_sum(df_label, df_hpg_reserve, ["air_store_id", "visit_date"], 
                       "reserve_datetime_diff", "hpg_reserve_datetime_diff_sum")
    df_label = feat_mean(df_label, df_hpg_reserve, ["air_store_id", "visit_date"], 
                        "reserve_datetime_diff", "hpg_reserve_datetime_diff_mean")
    df_label = feat_sum(df_label, df_hpg_reserve, ["air_store_id", "visit_date"], 
                       "reserve_visitors", "hpg_reserve_visitors_sum")
    df_label = feat_mean(df_label, df_hpg_reserve, ["air_store_id", "visit_date"], 
                        "reserve_visitors", "hpg_reserve_visitors_mean")
    df_label = feat_sum(df_label, df_hpg_reserve, ["visit_date"], 
                       "reserve_visitors", "hpg_date_reserve_visitors_sum")
    df_label = feat_mean(df_label, df_hpg_reserve, ["visit_date"], 
                        "reserve_visitors", "hpg_date_reserve_visitors_mean")

    for i in [35, 63, 140]:
        df_air_reserve_select = df_air_reserve[df_air_reserve.day_gap >= -i].copy()
        df_hpg_reserve_select = df_hpg_reserve[df_hpg_reserve.day_gap >= -i].copy()

        date_air_reserve = pd.DataFrame(
            df_air_reserve_select.groupby(["air_store_id", "visit_date"]).reserve_visitors.sum()
        ).reset_index()
        date_air_reserve.columns = ["air_store_id", "visit_date", "reserve_visitors_sum"]
        date_air_reserve = feat_count(date_air_reserve, df_air_reserve_select, 
                                     ["air_store_id", "visit_date"], "reserve_visitors", 
                                     "reserve_visitors_count")
        date_air_reserve = feat_mean(date_air_reserve, df_air_reserve_select, 
                                    ["air_store_id", "visit_date"], "reserve_visitors", 
                                    "reserve_visitors_mean")

        date_hpg_reserve = pd.DataFrame(
            df_hpg_reserve_select.groupby(["air_store_id", "visit_date"]).reserve_visitors.sum()
        ).reset_index()
        date_hpg_reserve.columns = ["air_store_id", "visit_date", "reserve_visitors_sum"]
        date_hpg_reserve = feat_count(date_hpg_reserve, df_hpg_reserve_select, 
                                     ["air_store_id", "visit_date"], "reserve_visitors", 
                                     "reserve_visitors_count")
        date_hpg_reserve = feat_mean(date_hpg_reserve, df_hpg_reserve_select, 
                                    ["air_store_id", "visit_date"], "reserve_visitors", 
                                    "reserve_visitors_mean")

        date_air_reserve = date_handle(date_air_reserve, air_info, date_info)
        date_hpg_reserve = date_handle(date_hpg_reserve, air_info, date_info)
        date_air_reserve["holiday"] = (
            (date_air_reserve["weekday"] >= 5) | (date_air_reserve["holiday_flg"] == 1)
        ).astype(int)
        date_hpg_reserve["holiday"] = (
            (date_hpg_reserve["weekday"] >= 5) | (date_hpg_reserve["holiday_flg"] == 1)
        ).astype(int)

        df_label = feat_mean(df_label, date_air_reserve, ["air_store_id", "weekday"], 
                            "reserve_visitors_sum", "air_reserve_visitors_sum_weekday_mean_%s" % i)
        df_label = feat_mean(df_label, date_hpg_reserve, ["air_store_id", "weekday"], 
                            "reserve_visitors_sum", "hpg_reserve_visitors_sum_weekday_mean_%s" % i)
        df_label = feat_mean(df_label, date_air_reserve, ["air_store_id", "weekday"], 
                            "reserve_visitors_mean", "air_reserve_visitors_mean_weekday_mean_%s" % i)
        df_label = feat_mean(df_label, date_hpg_reserve, ["air_store_id", "weekday"], 
                            "reserve_visitors_mean", "hpg_reserve_visitors_mean_weekday_mean_%s" % i)
        df_label = feat_mean(df_label, date_air_reserve, ["air_store_id", "weekday"], 
                            "reserve_visitors_count", "air_reserve_visitors_count_weekday_mean_%s" % i)
        df_label = feat_mean(df_label, date_hpg_reserve, ["air_store_id", "weekday"], 
                            "reserve_visitors_count", "hpg_reserve_visitors_count_weekday_mean_%s" % i)

        df_label = feat_mean(df_label, date_air_reserve, ["air_store_id", "holiday"], 
                            "reserve_visitors_sum", "air_reserve_visitors_sum_holiday_mean_%s" % i)
        df_label = feat_mean(df_label, date_hpg_reserve, ["air_store_id", "holiday"], 
                            "reserve_visitors_sum", "hpg_reserve_visitors_sum_holiday_mean_%s" % i)
        df_label = feat_mean(df_label, date_air_reserve, ["air_store_id", "holiday"], 
                            "reserve_visitors_mean", "air_reserve_visitors_mean_holiday_mean_%s" % i)
        df_label = feat_mean(df_label, date_hpg_reserve, ["air_store_id", "holiday"], 
                            "reserve_visitors_mean", "hpg_reserve_visitors_mean_holiday_mean_%s" % i)
        df_label = feat_mean(df_label, date_air_reserve, ["air_store_id", "holiday"], 
                            "reserve_visitors_count", "air_reserve_visitors_count_holiday_mean_%s" % i)
        df_label = feat_mean(df_label, date_hpg_reserve, ["air_store_id", "holiday"], 
                            "reserve_visitors_count", "hpg_reserve_visitors_count_holiday_mean_%s" % i)

    # 历史访问特征
    for i in [21, 35, 63, 140, 280, 350, 420]:
        df_select = df_train[df_train.day_gap >= -i].copy()

        df_label = feat_median(df_label, df_select, ["air_store_id"], "visitors", "air_median_%s" % i)
        df_label = feat_mean(df_label, df_select, ["air_store_id"], "visitors", "air_mean_%s" % i)
        df_label = feat_max(df_label, df_select, ["air_store_id"], "visitors", "air_max_%s" % i)
        df_label = feat_min(df_label, df_select, ["air_store_id"], "visitors", "air_min_%s" % i)
        df_label = feat_std(df_label, df_select, ["air_store_id"], "visitors", "air_std_%s" % i)
        df_label = feat_count(df_label, df_select, ["air_store_id"], "visitors", "air_count_%s" % i)

        df_label = feat_mean(df_label, df_select, ["air_store_id", "weekday"], "visitors", "air_week_mean_%s" % i)
        df_label = feat_max(df_label, df_select, ["air_store_id", "weekday"], "visitors", "air_week_max_%s" % i)
        df_label = feat_min(df_label, df_select, ["air_store_id", "weekday"], "visitors", "air_week_min_%s" % i)
        df_label = feat_std(df_label, df_select, ["air_store_id", "weekday"], "visitors", "air_week_std_%s" % i)
        df_label = feat_count(df_label, df_select, ["air_store_id", "weekday"], "visitors", "air_week_count_%s" % i)

        df_label = feat_mean(df_label, df_select, ["air_store_id", "holiday"], "visitors", "air_holiday_mean_%s" % i)
        df_label = feat_max(df_label, df_select, ["air_store_id", "holiday"], "visitors", "air_holiday_max_%s" % i)
        df_label = feat_min(df_label, df_select, ["air_store_id", "holiday"], "visitors", "air_holiday_min_%s" % i)
        df_label = feat_count(df_label, df_select, ["air_store_id", "holiday"], "visitors", "air_holiday_count_%s" % i)

        df_label = feat_mean(df_label, df_select, ["air_genre_name", "holiday"], "visitors", "air_genre_name_holiday_mean_%s" % i)
        df_label = feat_max(df_label, df_select, ["air_genre_name", "holiday"], "visitors", "air_genre_name_holiday_max_%s" % i)
        df_label = feat_min(df_label, df_select, ["air_genre_name", "holiday"], "visitors", "air_genre_name_holiday_min_%s" % i)
        df_label = feat_count(df_label, df_select, ["air_genre_name", "holiday"], "visitors", "air_genre_name_holiday_count_%s" % i)

        df_label = feat_mean(df_label, df_select, ["air_genre_name", "weekday"], "visitors", "air_genre_name_weekday_mean_%s" % i)
        df_label = feat_max(df_label, df_select, ["air_genre_name", "weekday"], "visitors", "air_genre_name_weekday_max_%s" % i)
        df_label = feat_min(df_label, df_select, ["air_genre_name", "weekday"], "visitors", "air_genre_name_weekday_min_%s" % i)
        df_label = feat_count(df_label, df_select, ["air_genre_name", "weekday"], "visitors", "air_genre_name_weekday_count_%s" % i)

        df_label = feat_mean(df_label, df_select, ["air_area_name", "holiday"], "visitors", "air_area_name_holiday_mean_%s" % i)
        df_label = feat_max(df_label, df_select, ["air_area_name", "holiday"], "visitors", "air_area_name_holiday_max_%s" % i)
        df_label = feat_min(df_label, df_select, ["air_area_name", "holiday"], "visitors", "air_area_name_holiday_min_%s" % i)
        df_label = feat_count(df_label, df_select, ["air_area_name", "holiday"], "visitors", "air_area_name_holiday_count_%s" % i)

        df_label = feat_mean(df_label, df_select, ["air_area_name", "air_genre_name", "holiday"], "visitors", "air_area_genre_name_holiday_mean_%s" % i)
        df_label = feat_max(df_label, df_select, ["air_area_name", "air_genre_name", "holiday"], "visitors", "air_area_genre_name_holiday_max_%s" % i)
        df_label = feat_min(df_label, df_select, ["air_area_name", "air_genre_name", "holiday"], "visitors", "air_area_genre_name_holiday_min_%s" % i)
        df_label = feat_count(df_label, df_select, ["air_area_name", "air_genre_name", "holiday"], "visitors", "air_area_genre_name_holiday_count_%s" % i)

    return df_label