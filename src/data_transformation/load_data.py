import polars as pl

def load_inputs(input_dir)->dict[str,pl.LazyFrame]:
    datasets_pth = {
        "air_reserve": input_dir / "air_reserve.parquet",
        "air_store_info": input_dir / "air_store_info.parquet",
        "air_visit_data": input_dir / "air_visit_data.parquet",
        "hpg_reserve": input_dir / "hpg_reserve.parquet",
        "hpg_store_info": input_dir / "hpg_store_info.parquet",
        "store_id_relation": input_dir / "store_id_relation.parquet",
        "date_info": input_dir / "date_info.parquet",
    }
    datasets = {name: pl.scan_parquet(pth) for name, pth in datasets_pth.items()}
    return datasets