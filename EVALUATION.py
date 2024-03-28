import polars as pl
def set_table_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    for col in df.columns:
        if col[-1] == "D":
            df = df.with_columns(pl.col(col).str.strptime(pl.Date, "%Y-%m-%d").alias(col))
        elif col[-1] in ["M", "T"]:
            df = df.with_columns(pl.col(col).cast(pl.Utf8).alias(col))
        elif col[-1] in ["P", "A"]:
            df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
        elif col[-1] == "L":
            non_null_values = df.filter(pl.col(col).is_not_null())[col].limit(1).to_list()
            if non_null_values:
                first_non_null = non_null_values[0]
                if isinstance(first_non_null, bool):
                    df = df.with_columns(pl.col(col).cast(pl.Boolean).alias(col))
                elif isinstance(first_non_null, (float, int)):
                    df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
                else:
                    df = df.with_columns(pl.col(col).cast(pl.Utf8).alias(col))
    return df

import gc

# join train data

train_basetable = pl.read_csv("/kaggle/input/home-credit-credit-risk-model-stability/csv_files/train/train_base.csv")
def process_week_num(week_num):
    return week_num % 52 if week_num >= 52 else week_num
train_basetable = train_basetable.with_columns(
    pl.col("WEEK_NUM").map_elements(process_week_num).alias("WEEK_NUM")
)
train_basetable = train_basetable.drop(['date_decision','MONTH'])

train_static = pl.concat(
    [
        pl.read_csv("/kaggle/input/home-credit-credit-risk-model-stability/csv_files/train/train_static_0_0.csv").pipe(set_table_dtypes),
        pl.read_csv("/kaggle/input/home-credit-credit-risk-model-stability/csv_files/train/train_static_0_1.csv").pipe(set_table_dtypes)
    ],
    how="vertical_relaxed",
)
columns_of_interest = [
    'case_id',
    'price_1097A',
    'numrejects9m_859L',
    'pctinstlsallpaidlate1d_3546856L',
    'totalsettled_863A',
    'maxannuity_159A',
    'disbursedcredamount_1113A',
    'maxdpdlast24m_143P',
    'maxdpdlast12m_727P',
    'avgdpdtolclosure24_3658938P',
    'numinstlswithdpd10_728L',
    'annuity_780A',
    'maxdpdtolerance_374P',
    'credamount_770A',
    'maxdebt4_972A'
]
train_static = train_static[columns_of_interest]
train_basetable = train_basetable.join(train_static, on='case_id', how='left')
del columns_of_interest,train_static
gc.collect()

train_person_1 = (
    pl.read_csv("/kaggle/input/home-credit-credit-risk-model-stability/csv_files/train/train_person_1.csv").pipe(set_table_dtypes)
    .select(['case_id', 'mainoccupationinc_384A', 'num_group1'])
    .filter(pl.col('num_group1') == 0)
    .drop('num_group1')
)
train_basetable = train_basetable.join(train_person_1, on='case_id', how='left')
del train_person_1
gc.collect()

train_other_1 = pl.read_csv("/kaggle/input/home-credit-credit-risk-model-stability/csv_files/train/train_other_1.csv").pipe(set_table_dtypes)
train_other_1 = (
    train_other_1
    .select(['case_id', 'amtdepositbalance_4809441A','amtdebitincoming_4809443A','amtdebitoutgoing_4809440A','num_group1'])
    .filter(pl.col('num_group1') == 0)
    .drop('num_group1')
)
train_basetable = train_basetable.join(train_other_1, on='case_id', how='left')
del train_other_1
gc.collect()

train_deposit_1 = pl.read_csv("/kaggle/input/home-credit-credit-risk-model-stability/csv_files/train/train_deposit_1.csv").pipe(set_table_dtypes)
deposit_sums = (
    train_deposit_1
    .group_by('case_id')
    .agg(pl.sum('amount_416A').alias('deposit'))
)
train_deposit_1 = train_deposit_1.join(deposit_sums, on='case_id', how='left')
train_deposit_1 = (
    train_deposit_1
    .filter(pl.col('num_group1') == 0)
    .drop('num_group1')
    .drop('contractenddate_991D')
    .drop('openingdate_313D')
    .drop('amount_416A')
)
train_basetable = train_basetable.join(train_deposit_1, on='case_id', how='left')
del train_deposit_1
gc.collect()

train_credit_bureau_a_1 = pl.concat(
    [
        pl.read_csv("/kaggle/input/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_a_1_0.csv").pipe(set_table_dtypes),
        pl.read_csv("/kaggle/input/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_a_1_1.csv").pipe(set_table_dtypes),
        pl.read_csv("/kaggle/input/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_a_1_2.csv").pipe(set_table_dtypes),
        pl.read_csv("/kaggle/input/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_a_1_3.csv").pipe(set_table_dtypes)
    ],
    how="vertical_relaxed",
)
train_credit_bureau_a_1 = (
    train_credit_bureau_a_1
    .select(['case_id', 'debtoutstand_525A', 'totaloutstanddebtvalue_39A','overdueamountmax2_14A',
             'overdueamountmax_155A','monthlyinstlamount_332A','dpdmax_139P','numberofoverdueinstlmax_1039L','num_group1'])
    .filter(pl.col('num_group1') == 0)
    .drop('num_group1')
)
train_basetable = train_basetable.join(train_credit_bureau_a_1, on='case_id', how='left')
del train_credit_bureau_a_1
gc.collect()

train_credit_bureau_b_1 = pl.read_csv('/kaggle/input/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_b_1.csv').pipe(set_table_dtypes)
train_credit_bureau_b_1 = (
    train_credit_bureau_b_1
    .select(['case_id', 'totalamount_881A','dpdmax_851P','num_group1'])
    .filter(pl.col('num_group1') == 0)
    .drop('num_group1')
)
train_basetable = train_basetable.join(train_credit_bureau_b_1, on='case_id', how='left')
del train_credit_bureau_b_1
gc.collect()

train_credit_bureau_b_2 = pl.read_csv("/kaggle/input/home-credit-credit-risk-model-stability/csv_files/train/train_credit_bureau_b_2.csv").pipe(set_table_dtypes)
sums = (
    train_credit_bureau_b_2
    .group_by('case_id')
    .agg([
        pl.sum('pmts_dpdvalue_108P').alias('all_pmts_dpdvalue_108P'),
        pl.sum('pmts_pmtsoverdue_635A').alias('all_pmts_pmtsoverdue_635A')
    ])
)
train_credit_bureau_b_2 = train_credit_bureau_b_2.join(sums, on='case_id', how='left')
train_credit_bureau_b_2 = train_credit_bureau_b_2.sort(["case_id"])
train_credit_bureau_b_2 = train_credit_bureau_b_2.unique(subset=['case_id'], keep='first')
train_credit_bureau_b_2 = train_credit_bureau_b_2.select(['case_id', 'all_pmts_dpdvalue_108P','all_pmts_pmtsoverdue_635A'])
train_basetable = train_basetable.join(train_credit_bureau_b_2, on='case_id', how='left')
del train_credit_bureau_b_2
gc.collect()

train_tax_registry_a_1 = pl.read_csv('/kaggle/input/home-credit-credit-risk-model-stability/csv_files/train/train_tax_registry_a_1.csv').pipe(set_table_dtypes)
train_tax_registry_a_1 = (
    train_tax_registry_a_1
    .select(['case_id', 'amount_4527230A','num_group1'])
    .filter(pl.col('num_group1') == 0)
    .drop('num_group1')
)
train_basetable = train_basetable.join(train_tax_registry_a_1, on='case_id', how='left')
del train_tax_registry_a_1
gc.collect()

train_tax_registry_b_1 = pl.read_csv('/kaggle/input/home-credit-credit-risk-model-stability/csv_files/train/train_tax_registry_b_1.csv').pipe(set_table_dtypes)
train_tax_registry_b_1 = (
    train_tax_registry_b_1
    .select(['case_id', 'amount_4917619A','num_group1'])
    .filter(pl.col('num_group1') == 0)
    .drop('num_group1')
)
train_basetable = train_basetable.join(train_tax_registry_b_1, on='case_id', how='left')
del train_tax_registry_b_1
gc.collect()

train_tax_registry_c_1 = pl.read_csv('/kaggle/input/home-credit-credit-risk-model-stability/csv_files/train/train_tax_registry_c_1.csv').pipe(set_table_dtypes)
train_tax_registry_c_1 = (
    train_tax_registry_c_1
    .select(['case_id', 'pmtamount_36A','num_group1'])
    .filter(pl.col('num_group1') == 0)
    .drop('num_group1')
)
train_basetable = train_basetable.join(train_tax_registry_c_1, on='case_id', how='left')
del train_tax_registry_c_1
gc.collect()

train_debitcard_1 = pl.read_csv('/kaggle/input/home-credit-credit-risk-model-stability/csv_files/train/train_debitcard_1.csv').pipe(set_table_dtypes)
train_debitcard_1 = train_debitcard_1.with_columns([
    pl.col("last180dayaveragebalance_704A").cast(pl.Float64),
    pl.col("last180dayturnover_1134A").cast(pl.Float64)
])
credit_balance = (
    train_debitcard_1
    .group_by('case_id')
    .agg([
        pl.sum('last180dayaveragebalance_704A').alias('all_last180dayaveragebalance_704A'),
        pl.sum("last180dayturnover_1134A").alias('all_last180dayturnover_1134A')
    ])
)
train_debitcard_1 = train_debitcard_1.join(credit_balance, on='case_id', how='left')
train_debitcard_1 = (
    train_debitcard_1
    .select(['case_id', 'all_last180dayaveragebalance_704A','all_last180dayturnover_1134A','num_group1'])
    .filter(pl.col('num_group1') == 0)
    .drop('num_group1')
)
train_basetable = train_basetable.join(train_debitcard_1, on='case_id', how='left')
del train_debitcard_1
gc.collect()

train_applprev_1 = pl.concat(
    [
        pl.read_csv("/kaggle/input/home-credit-credit-risk-model-stability/csv_files/train/train_applprev_1_0.csv").pipe(set_table_dtypes),
        pl.read_csv("/kaggle/input/home-credit-credit-risk-model-stability/csv_files/train/train_applprev_1_1.csv").pipe(set_table_dtypes)
    ],
    how="vertical_relaxed",
)
train_applprev_1 = (
    train_applprev_1
    .select(['case_id', 'annuity_853A','mainoccupationinc_437A','num_group1'])
    .filter(pl.col('num_group1') == 0)
    .drop('num_group1')
)
train_basetable = train_basetable.join(train_applprev_1, on='case_id', how='left')
del train_applprev_1
gc.collect()

train_basetable = train_basetable.sort("case_id")
train_basetable = train_basetable.to_pandas()
train_basetable.to_csv("/kaggle/working/train_basetable_merged.csv",index=False)
train_basetable_shared_features = train_basetable.dropna(axis=1, how='any')
train_basetable_shared_features.to_csv("/kaggle/working/train_basetable_merged_shared_features.csv",index=False)

del train_basetable
gc.collect()

import pandas as pd

import logging
logging.getLogger().setLevel(logging.ERROR)

from sklearnex import patch_sklearn
patch_sklearn()


df = pd.read_csv("/kaggle/working/train_basetable_merged.csv",index_col='case_id')


majority_class = df[df.target == 0]
minority_class = df[df.target == 1]

minority_sample = minority_class.sample(n=1000, random_state=10)
majority_sample = majority_class.sample(n=31847, random_state=10)
test_set = pd.concat([minority_sample, majority_sample])

test_set_shared = test_set[['WEEK_NUM','target','numrejects9m_859L','disbursedcredamount_1113A','annuity_780A','credamount_770A','mainoccupationinc_384A']]

df = df.drop(minority_sample.index)
df = df.drop(majority_sample.index)
df.to_csv("/kaggle/working/train.csv")

df_shared = df[['WEEK_NUM','target','numrejects9m_859L','disbursedcredamount_1113A','annuity_780A','credamount_770A','mainoccupationinc_384A']]
df_shared.to_csv("/kaggle/working/train_shared.csv")

test_y = test_set['target']
test_X = test_set.drop('target',axis = 1)

test_y_shared = test_set_shared['target']
test_X_shared = test_set_shared.drop('target',axis = 1)

test_y = test_y.sort_index()
test_X = test_X.sort_index()

test_y.to_csv('/kaggle/working/test_y.csv')
test_X.to_csv('/kaggle/working/test_X.csv')

test_y_shared = test_set_shared['target']
test_X_shared = test_set_shared.drop('target',axis = 1)
test_y_shared = test_y_shared.sort_index()
test_X_shared = test_X_shared.sort_index()
test_y_shared.to_csv('/kaggle/working/test_y_shared.csv')
test_X_shared.to_csv('/kaggle/working/test_X_shared.csv')

import pandas as pd
from sklearn.ensemble import StackingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample



train_data = pd.read_csv('/kaggle/working/train.csv', index_col='case_id')

models = {}


def assessment(csv_name):
    case_ids = []
    all_predictions = []
    df = pd.read_csv(csv_name, index_col='case_id')

    for i in range(len(df)):

        sample = df.iloc[i]
        #print(sample.name)
        case_ids.append(sample.name)
        sample = sample.dropna()
        non_null_columns = sample.index.tolist()
        sample_dict = {col: [val] for col, val in zip(non_null_columns, sample.values)}
        sample = pd.DataFrame(sample_dict)
        sample.columns = sample.columns.map(str)
        sample = sample.astype('float32')
        columns_key = '_'.join(non_null_columns)

        if columns_key not in models:

            non_null_columns_target = non_null_columns.copy()
            non_null_columns_target.append('target')
            chosen = train_data[non_null_columns_target]
            chosen = chosen.dropna()

            majority_class = chosen[chosen.target == 0]
            minority_class = chosen[chosen.target == 1]

            if len(minority_class) <= 3 or len(majority_class) <= 3:
                all_predictions.append(0)

            else:
                majority_downsampled = resample(majority_class,
                                                replace=False,
                                                n_samples=len(minority_class) * 7,
                                                random_state=10)
                chosen = pd.concat([majority_downsampled, minority_class])

                X = chosen.drop('target', axis=1)
                X.columns = X.columns.map(str)
                X = X.astype('float32')
                y = chosen['target']

                base_learner = [('lgb', LGBMClassifier(verbose=-1, device='gpu',gpu_platform_id=0, gpu_device_id=0, n_estimators=300, learning_rate=0.05,
                                                       max_depth=15, num_leaves=25, random_state=10,
                                                       force_col_wise=True))]
                meta_learner = LogisticRegression(random_state=10, n_jobs=-1, max_iter=1000)
                cv_method = StratifiedKFold(n_splits=4, shuffle=True, random_state=10)
                stacked_model = StackingClassifier(
                    estimators=base_learner,
                    final_estimator=meta_learner,
                    stack_method='predict_proba',
                    cv=cv_method
                )

                stacked_model.fit(X, y)
                result = stacked_model.predict_proba(sample)
                all_predictions.append(result[:, 1][0])
                models[columns_key] = stacked_model

        else:
            result = models[columns_key].predict_proba(sample)
            all_predictions.append(result[:, 1][0])

    output = {
        'case_id': case_ids,
        'score': all_predictions
    }
    df = pd.DataFrame(output)

    return df

results = assessment('test_X.csv')
results.set_index('case_id', inplace=True)
sorted_df = results.sort_index()
sorted_df.to_csv('/kaggle/working/submission.csv')


import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

predict = sorted_df['score'].values
true = test_y.values

auc = roc_auc_score(true, predict)
print(auc)

y_pred = np.where(predict >= 0.35, 1, 0)
f1 = f1_score(true, y_pred, average='binary')
print(f"F1 Score: {f1}")



train_data = pd.read_csv('/kaggle/working/train.csv', index_col='case_id')

models_LR = {}


def assessment_LR(csv_name):
    case_ids = []
    all_predictions = []
    df = pd.read_csv(csv_name, index_col='case_id')

    for i in range(len(df)):

        sample = df.iloc[i]
        # print(sample.name)
        case_ids.append(sample.name)
        sample = sample.dropna()
        non_null_columns = sample.index.tolist()
        sample_dict = {col: [val] for col, val in zip(non_null_columns, sample.values)}
        sample = pd.DataFrame(sample_dict)
        sample.columns = sample.columns.map(str)
        sample = sample.astype('float32')
        columns_key = '_'.join(non_null_columns)

        if columns_key not in models_LR:

            non_null_columns_target = non_null_columns.copy()
            non_null_columns_target.append('target')
            chosen = train_data[non_null_columns_target]
            chosen = chosen.dropna()

            majority_class = chosen[chosen.target == 0]
            minority_class = chosen[chosen.target == 1]

            if len(minority_class) <= 3 or len(majority_class) <= 3:
                all_predictions.append(0)

            else:
                majority_downsampled = resample(majority_class,
                                                replace=False,
                                                n_samples=len(minority_class) * 7,
                                                random_state=10)
                chosen = pd.concat([majority_downsampled, minority_class])

                X = chosen.drop('target', axis=1)
                X.columns = X.columns.map(str)
                X = X.astype('float32')
                y = chosen['target']

                model = LogisticRegression(random_state=10, n_jobs=-1, max_iter=1000)

                model.fit(X, y)
                result = model.predict_proba(sample)
                all_predictions.append(result[:, 1][0])
                models_LR[columns_key] = model

        else:
            result = models_LR[columns_key].predict_proba(sample)
            all_predictions.append(result[:, 1][0])

    output = {
        'case_id': case_ids,
        'score': all_predictions
    }
    df = pd.DataFrame(output)

    return df


import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

results_LR = assessment_LR('test_X.csv')
results_LR.set_index('case_id', inplace=True)
sorted_df_LR = results_LR.sort_index()
sorted_df_LR.to_csv('/kaggle/working/submission_LR.csv')


predict = sorted_df_LR['score'].values
true = test_y.values

auc = roc_auc_score(true, predict)
print(auc)

y_pred = np.where(predict >= 0.35, 1, 0)
f1 = f1_score(true, y_pred, average='binary')
print(f"F1 Score: {f1}")



base_learner = [('lgb', LGBMClassifier(verbose=-1, device='gpu',gpu_platform_id=0, gpu_device_id=0, n_estimators=300, learning_rate=0.05,
                                                       max_depth=15, num_leaves=25, random_state=10,
                                                       force_col_wise=True))]
meta_learner = LogisticRegression(random_state=10, n_jobs=-1, max_iter=1000)
cv_method = StratifiedKFold(n_splits=4, shuffle=True, random_state=10)
stacked_model = StackingClassifier(
                    estimators=base_learner,
                    final_estimator=meta_learner,
                    stack_method='predict_proba',
                    cv=cv_method
                )

df_shared_X = df_shared.drop('target',axis=1)
df_shared_y = df_shared['target']
stacked_model.fit(df_shared_X, df_shared_y)
result = stacked_model.predict_proba(test_X_shared)

auc = roc_auc_score(test_y_shared.values, result[:, 1])
print(auc)

y_pred = np.where(result[:, 1] >= 0.35, 1, 0)
f1 = f1_score(test_y_shared.values, y_pred, average='binary')
print(f"F1 Score: {f1}")




model = LogisticRegression(random_state=10, n_jobs=-1, max_iter=1000)

df_shared_X = df_shared.drop('target',axis=1)
df_shared_y = df_shared['target']
model.fit(df_shared_X, df_shared_y)
result = model.predict_proba(test_X_shared)

auc = roc_auc_score(test_y_shared.values, result[:, 1])
print(auc)

y_pred = np.where(result[:, 1] >= 0.35, 1, 0)
f1 = f1_score(test_y_shared.values, y_pred, average='binary')
print(f"F1 Score: {f1}")