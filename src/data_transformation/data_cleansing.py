import polars as pl

def merge_reservation(
    air_reserve: pl.LazyFrame,
    hpg_reserve: pl.LazyFrame,
    store_id_relation: pl.LazyFrame,
    **kwargs
) -> pl.LazyFrame:
    #head = air_store_id,visit_datetime,reserve_datetime,reserve_visitors
    head = ['air_store_id', 'visit_datetime', 'reserve_datetime', 'reserve_visitors']
    hpg_reserve_to_air = (
        store_id_relation
        .join(hpg_reserve, on='hpg_store_id', how='inner')
        .select(head)
    )
    aggr_reserve =  (
        pl.concat([air_reserve, hpg_reserve_to_air])
        .group_by(['air_store_id', 'visit_datetime','reserve_datetime'])
        .agg(
            pl.col("reserve_visitors").sum().alias("reserve_visitors")
        )
    )
    return aggr_reserve