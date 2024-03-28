import polars as pl

train_basetable = pl.read_csv("csv_files/train/train_base.csv")
def process_week_num(week_num):
    return week_num % 52 if week_num >= 52 else week_num

train_basetable = train_basetable.with_columns(
    pl.col("WEEK_NUM").map_elements(process_week_num).alias("WEEK_NUM")
)

train_static = pl.read_csv("csv_files/train/train_credit_bureau_b_2.csv")
deposit_sums = (
    train_static
    .group_by('case_id')
    .agg([
        pl.sum('pmts_dpdvalue_108P').alias('all_pmts_dpdvalue_108P'),
        pl.sum('pmts_pmtsoverdue_635A').alias('all_pmts_pmtsoverdue_635A')
    ])
)
train_static = train_static.join(deposit_sums, on='case_id', how='left')
train_static = train_static.sort(["case_id"])
train_static = train_static.unique(subset=['case_id'], keep='first')
train_static = train_static.drop('num_group1','num_group2','pmts_dpdvalue_108P','pmts_pmtsoverdue_635A')
print(train_static)
numeric_columns = [
    col for col in train_static.columns
    if train_static[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]
]

train_static = train_static.select(numeric_columns)

threshold = 1/3
columns_to_drop = [
    col for col in train_static.columns
    if (train_static[col].null_count() / train_static.height) > threshold
]
train_static = train_static.drop(columns_to_drop)


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