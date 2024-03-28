import polars as pl

train_basetable = pl.read_csv("csv_files/train/train_base.csv")
def process_week_num(week_num):
    return week_num % 52 if week_num >= 52 else week_num

train_basetable = train_basetable.with_columns(
    pl.col("WEEK_NUM").map_elements(process_week_num).alias("WEEK_NUM")
)
train_static = pl.read_csv("csv_files/train/train_debitcard_1.csv")
train_static = train_static.with_columns([
    pl.col("last180dayaveragebalance_704A").cast(pl.Float64),
    pl.col("last180dayturnover_1134A").cast(pl.Float64),
    pl.col("last30dayturnover_651A").cast(pl.Float64)
])
deposit_sums = (
    train_static
    .group_by('case_id')
    .agg([
        pl.sum('last180dayaveragebalance_704A').alias('all_last180dayaveragebalance_704A'),
        pl.sum("last180dayturnover_1134A").alias('all_last180dayturnover_1134A'),
        pl.sum("last30dayturnover_651A").alias('all_last30dayturnover_651A')
    ])
)
train_static = train_static.drop('last180dayaveragebalance_704A',"last180dayturnover_1134A","last30dayturnover_651A")
train_static = train_static.join(deposit_sums, on='case_id', how='left')
train_static = train_static.filter(pl.col('num_group1') == 0)
train_static = train_static.drop('num_group1')

numeric_columns = [
    col for col in train_static.columns
    if train_static[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]
]

train_static = train_static.select(numeric_columns)



train_basetable = (
    train_basetable
    .join(train_static, on='case_id', how='left')
    .drop('date_decision')
    .drop('MONTH')
)


train_basetable = train_basetable.drop_nulls()


train_basetable = train_basetable.to_pandas()

train_basetable.to_csv("csv_files/train_basetable_merged.csv",index=False)


import pandas as pd
from sklearn.ensemble import RandomForestClassifier

static_static_bc = pd.read_csv("csv_files/train_basetable_merged.csv", index_col='case_id')
X = static_static_bc.drop('target', axis=1)
y = static_static_bc['target']

from sklearnex import patch_sklearn
patch_sklearn()
model = RandomForestClassifier(n_estimators=250,max_depth=15, random_state=10,n_jobs=-1)
model.fit(X, y)

feature_importances = model.feature_importances_
features = X.columns
importance_dict = dict(zip(features, feature_importances))

week_num_importance = importance_dict['WEEK_NUM']
importance_ratio_dict = {feature: (importance / week_num_importance) for feature, importance in importance_dict.items()}
importance_ratio_dict = sorted(importance_ratio_dict.items(), key=lambda x: x[1], reverse=True)
for feature in importance_ratio_dict:
    print(feature)