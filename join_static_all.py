import polars as pl

train_basetable = pl.read_csv("csv_files/train/train_base.csv")
def process_week_num(week_num):
    return week_num % 52 if week_num >= 52 else week_num
train_basetable = train_basetable.with_columns(
    pl.col("WEEK_NUM").map_elements(process_week_num).alias("WEEK_NUM")
)
train_static = pl.concat(
    [
        pl.read_csv("csv_files/train/train_static_0_0.csv"),
        pl.read_csv("csv_files/train/train_static_0_1.csv")
    ],
    how="vertical_relaxed",
)

numeric_columns = [
    col for col in train_static.columns
    if train_static[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]
]
train_static = train_static.select(numeric_columns)

train_static = train_static.with_columns(
    pl.when(pl.col("currdebt_22A").is_null())
    .then(pl.col("currdebtcredtyperange_828A"))
    .when(pl.col("currdebtcredtyperange_828A").is_null())
    .then(pl.col("currdebt_22A"))
    .when((pl.col("currdebtcredtyperange_828A") == 0) & (pl.col("currdebt_22A") != 0))
    .then(pl.col("currdebt_22A"))
    .when((pl.col("currdebtcredtyperange_828A") != 0) & (pl.col("currdebt_22A") == 0))
    .then(pl.col("currdebt_22A"))
    .when((pl.col("currdebtcredtyperange_828A") != 0) & (pl.col("currdebt_22A") != 0))
    .then(pl.col("currdebt_22A"))
    .when((pl.col("currdebtcredtyperange_828A") == 0) & (pl.col("currdebt_22A") == 0))
    .then(0)
    .otherwise(None)
    .alias("current_debt")
)

train_basetable = (
    train_basetable
    .join(train_static, on='case_id', how='left')
    .drop('date_decision')
    .drop('MONTH')
)


threshold = 1/3

columns_to_drop = [col for col in train_basetable.columns if train_basetable[col].null_count() / train_basetable.height > threshold]

train_basetable = train_basetable.drop(columns_to_drop)

train_basetable = train_basetable.drop_nulls()
train_basetable = train_basetable.drop_nulls()


train_basetable = train_basetable.to_pandas()

train_basetable.to_csv("csv_files/train_basetable_static.csv",index=False)


import pandas as pd
from sklearn.ensemble import RandomForestClassifier

static_static_bc = pd.read_csv("csv_files/train_basetable_static.csv", index_col='case_id')
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
