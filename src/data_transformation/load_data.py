import polars as pl

def load_inputs(input_dir)->dict[str,pl.LazyFrame]:
    datasets_pth = {
        "air_reserve": input_dir / "air_reserve.csv",
        "air_store_info": input_dir / "air_store_info.csv",
        "air_visit_data": input_dir / "air_visit_data.csv",
        "hpg_reserve": input_dir / "hpg_reserve.csv",
        "hpg_store_info": input_dir / "hpg_store_info.csv",
        "store_id_relation": input_dir / "store_id_relation.csv",
        "date_info": input_dir / "date_info.csv",
    }
    datasets = {name: pl.scan_csv(pth) for name, pth in datasets_pth.items()}
    return datasets