#encoding=utf8
from datetime import date, timedelta
import pandas as pd
from . import _utils

def date_gap(x,y):
    a,b,c=x.split("-")
    return (date(int(a),int(b),int(c))-y).days
def date_handle(df,air_info, date_info):
    df_visit_date=pd.to_datetime(df["visit_date"])
    df["weekday"]=df_visit_date.dt.weekday
    df["day"]=df_visit_date.dt.day
    days_of_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # Feiyang: 2. 新的特征 "days_to_side" 表示当天距离月初月末的最短距离
    df["days_to_side"] = df_visit_date.apply(
        lambda x: min(x.day, days_of_months[x.month-1]-x.day))
    
    # Feiyang: 3. 把月初月末的定义改成了7天
    df["day"]=df["day"].apply(lambda x:0 if x<=7 else 2 if x>=24 else 1)
    df = df.merge(air_info, on="air_store_id", how="left").fillna(-1)
    df = df.merge(date_info, on="visit_date", how="left").fillna(-1)
    return df

def create_features(df_label,df_train,df_air_reserve,air_info,date_info, PrEp):
    df_train=date_handle(df_train, air_info=air_info, date_info=date_info)
    df_label=date_handle(df_label,air_info=air_info,date_info=date_info)


    #预定信息
    # Feiyang: 4. 把这两段的 mean 改成了 kernelMedian
    df_label=_utils.feat_sum(df_label,df_air_reserve,["air_store_id","visit_date"],"reserve_datetime_diff","air_reserve_datetime_diff_sum")
    df_label=_utils.feat_kernelMedian(df_label,df_air_reserve,["air_store_id","visit_date"],"reserve_datetime_diff",PrEp,"air_reserve_datetime_diff_mean")
    df_label=_utils.feat_sum(df_label,df_air_reserve,["air_store_id","visit_date"],"reserve_visitors","air_reserve_visitors_sum")
    df_label=_utils.feat_kernelMedian(df_label,df_air_reserve,["air_store_id","visit_date"],"reserve_visitors",PrEp,"air_reserve_visitors_mean")
    df_label=_utils.feat_sum(df_label,df_air_reserve,["visit_date"],"reserve_visitors","air_date_reserve_visitors_sum")
    df_label=_utils.feat_kernelMedian(df_label,df_air_reserve,["visit_date"],"reserve_visitors",PrEp,"air_date_reserve_visitors_mean")


    for i in [35,63,140]:
        df_air_reserve_select=df_air_reserve[df_air_reserve.day_gap>=-i].copy()

        # Feiyang: 5. 把这两段的 mean 改成了 kernelMedian
        date_air_reserve=pd.DataFrame(df_air_reserve_select.groupby(["air_store_id","visit_date"]).reserve_visitors.sum()).reset_index()
        date_air_reserve.columns=["air_store_id","visit_date","reserve_visitors_sum"]
        date_air_reserve=_utils.feat_count(date_air_reserve,df_air_reserve_select,["air_store_id","visit_date"],"reserve_visitors","reserve_visitors_count")
        date_air_reserve=_utils.feat_kernelMedian(date_air_reserve,df_air_reserve_select,["air_store_id","visit_date"],"reserve_visitors",PrEp,"reserve_visitors_mean")

        date_air_reserve=date_handle(date_air_reserve, air_info=air_info, date_info=date_info)
        date_air_reserve["holiday"] = ((date_air_reserve["weekday"]>=5) | (date_air_reserve["holiday_flg"]==1)).astype(int)

        df_label=_utils.feat_mean(df_label,date_air_reserve,["air_store_id","weekday"],"reserve_visitors_sum", "air_reserve_visitors_sum_weekday_mean_%s"%i)
        df_label=_utils.feat_mean(df_label,date_air_reserve,["air_store_id","weekday"],"reserve_visitors_mean", "air_reserve_visitors_mean_weekday_mean_%s"%i)
        df_label=_utils.feat_mean(df_label,date_air_reserve,["air_store_id","weekday"],"reserve_visitors_count", "air_reserve_visitors_count_weekday_mean_%s"%i)

        df_label=_utils.feat_mean(df_label,date_air_reserve,["air_store_id","holiday"],"reserve_visitors_sum", "air_reserve_visitors_sum_holiday_mean_%s"%i)
        df_label=_utils.feat_mean(df_label,date_air_reserve,["air_store_id","holiday"],"reserve_visitors_mean", "air_reserve_visitors_mean_holiday_mean_%s"%i)
        df_label=_utils.feat_mean(df_label,date_air_reserve,["air_store_id","holiday"],"reserve_visitors_count", "air_reserve_visitors_count_holiday_mean_%s"%i)


    #月初月中月末
    # Feiyang: 6. 把这两段的 mean 改成了 kernelMedian
    df_label = _utils.feat_kernelMedian(df_label, df_train, ["air_store_id","day","weekday"], "visitors",PrEp, "air_day_mean")
    df_label = _utils.feat_kernelMedian(df_label, df_train, ["air_store_id","day","holiday"], "visitors",PrEp, "air_holiday_mean")
    for i in [21,35,63,140,280,350,420]:
        df_select=df_train[df_train.day_gap>=-i].copy()

        # Feiyang: 7. 给最重要的 visitors 这一列加上了新的特征: kernelMedian, median
        df_label=_utils.feat_median(df_label, df_select, ["air_store_id"], "visitors", "air_median_%s"%i)
        df_label=_utils.feat_mean(df_label,df_select,["air_store_id"],"visitors", "air_mean_%s"%i)
        df_label=_utils.feat_kernelMedian(df_label,df_select,["air_store_id"],"visitors",PrEp,"air_kermed_%s"%i)
        df_label=_utils.feat_max(df_label,df_select,["air_store_id"],"visitors","air_max_%s"%i)
        df_label=_utils.feat_min(df_label,df_select,["air_store_id"],"visitors","air_min_%s"%i)
        df_label=_utils.feat_std(df_label,df_select,["air_store_id"],"visitors","air_std_%s"%i)
        df_label=_utils.feat_count(df_label,df_select,["air_store_id"],"visitors","air_count_%s"%i)

        # Feiyang: 8. 把这几段的 mean 改成了 kernelMedian
        #df_label=_utils.feat_mean(df_label,df_select,["air_store_id","weekday"],"visitors", "air_week_mean_%s"%i)
        df_label=_utils.feat_kernelMedian(df_label,df_select,["air_store_id","weekday"],"visitors",PrEp,"air_week_kermed_%s"%i)
        df_label=_utils.feat_max(df_label,df_select,["air_store_id","weekday"],"visitors","air_week_max_%s"%i)
        df_label=_utils.feat_min(df_label,df_select,["air_store_id","weekday"],"visitors","air_week_min_%s"%i)
        df_label=_utils.feat_std(df_label,df_select,["air_store_id","weekday"],"visitors","air_week_std_%s"%i)
        df_label=_utils.feat_count(df_label,df_select,["air_store_id","weekday"],"visitors","air_week_count_%s"%i)

        # df_label=_utils.feat_mean(df_label,df_select,["air_store_id","holiday"],"visitors", "air_holiday_mean_%s"%i)
        df_label=_utils.feat_kernelMedian(df_label,df_select,["air_store_id","holiday"],"visitors",PrEp,"air_holiday_kermed_%s"%i)
        df_label=_utils.feat_max(df_label,df_select,["air_store_id","holiday"],"visitors","air_holiday_max_%s"%i)
        df_label=_utils.feat_min(df_label,df_select,["air_store_id","holiday"],"visitors","air_holiday_min_%s"%i)
        df_label=_utils.feat_count(df_label,df_select,["air_store_id","holiday"],"visitors","air_holiday_count_%s"%i)

        #df_label=_utils.feat_mean(df_label,df_select,["air_genre_name","holiday"],"visitors", "air_genre_name_holiday_mean_%s"%i)
        df_label=_utils.feat_kernelMedian(df_label,df_select,["air_genre_name","holiday"],"visitors",PrEp,"air_genre_name_holiday_kermed_%s"%i)
        df_label=_utils.feat_max(df_label,df_select,["air_genre_name","holiday"],"visitors","air_genre_name_holiday_max_%s"%i)
        df_label=_utils.feat_min(df_label,df_select,["air_genre_name","holiday"],"visitors","air_genre_name_holiday_min_%s"%i)
        df_label=_utils.feat_count(df_label,df_select,["air_genre_name","holiday"],"visitors","air_genre_name_holiday_count_%s"%i)

        #df_label=_utils.feat_mean(df_label,df_select,["air_genre_name","weekday"],"visitors", "air_genre_name_weekday_mean_%s"%i)
        df_label=_utils.feat_kernelMedian(df_label,df_select,["air_genre_name","weekday"],"visitors",PrEp,"air_genre_name_weekday_kermed_%s"%i)
        df_label=_utils.feat_max(df_label,df_select,["air_genre_name","weekday"],"visitors","air_genre_name_weekday_max_%s"%i)
        df_label=_utils.feat_min(df_label,df_select,["air_genre_name","weekday"],"visitors","air_genre_name_weekday_min_%s"%i)
        df_label=_utils.feat_count(df_label,df_select,["air_genre_name","weekday"],"visitors","air_genre_name_weekday_count_%s"%i)

        #df_label=_utils.feat_mean(df_label,df_select,["air_area_name","holiday"],"visitors", "air_area_name_holiday_mean_%s"%i)
        df_label=_utils.feat_kernelMedian(df_label,df_select,["air_area_name","holiday"],"visitors",PrEp,"air_area_name_holiday_kermed_%s"%i)
        df_label=_utils.feat_max(df_label,df_select,["air_area_name","holiday"],"visitors","air_area_name_holiday_max_%s"%i)
        df_label=_utils.feat_min(df_label,df_select,["air_area_name","holiday"],"visitors","air_area_name_holiday_min_%s"%i)
        df_label=_utils.feat_count(df_label,df_select,["air_area_name","holiday"],"visitors","air_area_name_holiday_count_%s"%i)

        #df_label=_utils.feat_mean(df_label,df_select,["air_area_name","air_genre_name","holiday"],"visitors", "air_area_genre_name_holiday_mean_%s"%i)
        df_label=_utils.feat_kernelMedian(df_label,df_select,["air_area_name","air_genre_name","holiday"],"visitors",PrEp,"air_area_genre_name_holiday_kermed_%s"%i)
        df_label=_utils.feat_max(df_label,df_select,["air_area_name","air_genre_name","holiday"],"visitors","air_area_genre_name_holiday_max_%s"%i)
        df_label=_utils.feat_min(df_label,df_select,["air_area_name","air_genre_name","holiday"],"visitors","air_area_genre_name_holiday_min_%s"%i)
        df_label=_utils.feat_count(df_label,df_select,["air_area_name","air_genre_name","holiday"],"visitors","air_area_genre_name_holiday_count_%s"%i)

    return df_label