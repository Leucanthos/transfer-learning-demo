#encoding=utf8
from datetime import date, timedelta

import pandas as pd
import numpy as np

from . import _utils
from .features import create_features, date_gap


def add_features_pipeline(
    train_visit_data_pth:str,    
    test_visit_data_pth:str,
    reserve_data_pth:str,
    data_info_pth:str,
    store_info_pth:str,
    test_begin:date = date(2017, 3, 11),
    nday:int = 35
    ):

    df_train = pd.read_csv(train_visit_data_pth)
    df_train["visitors"] = df_train["visitors"].apply(lambda x: np.log1p(float(x)) if float(x) > 0 else 0)
    
    df_test = pd.read_csv(test_visit_data_pth)
    df_test["visitors"] = df_test["visitors"].apply(lambda x: np.log1p(float(x)) if float(x) > 0 else 0)

    air_reserve = pd.read_csv(reserve_data_pth)
    air_reserve["reserve_date"]=air_reserve["reserve_datetime"].apply(lambda x:x.split(" ")[0])
    air_reserve["visit_date"]=air_reserve["visit_datetime"].apply(lambda x:x.split(" ")[0])
    air_reserve['reserve_datetime_diff'] = (pd.to_datetime(air_reserve['visit_date']) - pd.to_datetime(air_reserve['reserve_date'])).dt.days



    df_date_info= pd.read_csv(data_info_pth)
    df_date_info.columns=["visit_date","day_of_week","holiday_flg"]
    df_date_info["day_of_week"]=df_date_info["day_of_week"].map({"Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6,"Sunday":7})
    df_date_info["holiday"] = ((df_date_info["day_of_week"]>=6) | (df_date_info["holiday_flg"]==1)).astype(int)
    #df_date_info["holiday"]= map(lambda a, b: 1 if a in [6, 7] or b == 1 else 0, df_date_info["day_of_week"], df_date_info["holiday_flg"])
    del df_date_info["day_of_week"]

    air_info=pd.read_csv(store_info_pth)
    air_info=_utils.encode_count(air_info,"air_genre_name")
    air_info=_utils.encode_count(air_info,"air_area_name")
    #构造数据

    # Feiyang: 1. 获得核函数 PrEp
    PrOriginalEp = np.zeros((2000,2000))
    PrOriginalEp[1,0] = 1
    PrOriginalEp[2,range(2)] = [0.5,0.5]
    for i in range(3,2000):
        scale = (i-1)/2.
        x = np.arange(-(i+1)/2.+1, (i+1)/2., step=1)/scale
        y = 3./4.*(1-x**2)
        y = y/np.sum(y)
        PrOriginalEp[i, range(i)] = y
    PrEp = PrOriginalEp.copy()
    for i in range(3, 2000):
        PrEp[i,:i] = (PrEp[i,:i]*i+1)/(i+1)


    #构造训练集
    all_data=[]
    for i in range(nday*1,nday*(420//nday+1),nday):  #windowsize==step
        delta = timedelta(days=i)
        t_begin=test_begin - delta
        print(t_begin)
        df_train["day_gap"]=df_train["visit_date"].apply(lambda x:date_gap(x,t_begin))
        air_reserve["day_gap"]=air_reserve["reserve_date"].apply(lambda x:date_gap(x,t_begin))

        df_feature=df_train[df_train.day_gap<0].copy()
        df_air_reserve=air_reserve[air_reserve.day_gap<0].copy()

        df_label=df_train[(df_train.day_gap>=0)&(df_train.day_gap<nday)][["air_store_id","visit_date","day_gap","visitors"]].copy()
        train_data_tmp=create_features(df_label,df_feature,df_air_reserve,air_info,df_date_info,PrEp)
        all_data.append(train_data_tmp)

    train=pd.concat(all_data)


    #构造线上测试集
    t_begin=test_begin
    print(t_begin)
    df_label=df_test
    df_label["day_gap"]=df_label["visit_date"].apply(lambda x:date_gap(x,t_begin))
    df_train["day_gap"]=df_train["visit_date"].apply(lambda x:date_gap(x,t_begin))
    air_reserve["day_gap"] = air_reserve["reserve_date"].apply(lambda x: date_gap(x, t_begin))

    df_label=df_label[["air_store_id","visit_date","day_gap","visitors"]].copy()
    test=create_features(df_label,df_train,air_reserve,air_info,df_date_info,PrEp)

    return train,test

        #save features data for stacking
        #train.to_csv("../stacking/train.csv",index=None)
        #test.to_csv("../stacking/test.csv",index=None)
