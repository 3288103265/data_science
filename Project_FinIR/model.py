import pandas as pd 
import numpy as np
import matplotlib as  plt
import glob

# 处理铝20天
label_20 = pd.read_csv('/home/wph/course/data_science/Project_FinIR/Train_data/Label_LMEAluminium_train_20d.csv', names=['index','date','label'],header=0,index_col='date').drop('index',axis=1)
lv = pd.read_csv('/home/wph/course/data_science/Project_FinIR/Train_data/LMEAluminium3M_train.csv',names=['index','date','open','high','low','close','volume'], header=0,index_col='date').drop('index', axis=1)
lv_oi = pd.read_csv('/home/wph/course/data_science/Project_FinIR/Train_data/LMEAluminium_OI_train.csv', names=['index','date','oi'],header=0,index_col='date').drop('index',axis=1)

feature_list = [lv, lv_oi]

indices_path = glob.glob(r'Train_data/Indices*')
indices_path.sort()
indices_name = ['dxy','nky', 'shsz','spx','sx5e','ukx','vix']
for i, path in enumerate(indices_path):
    feature_list.append(pd.read_csv(path, names=['index','date',indices_name[i]], header=0, index_col='date').drop('index',axis=1))

train = label_20.join(feature_list, how='left')

# missing value
cols = train.select_dtypes('number')
print('Missing ratio:')
for col in cols:
    print('{}:{}'.format(col, train[col].isna().sum()/len(train)))
    med = train[col].median()
    train[col].fillna(med,inplace=True)

target = train['label'].values
data = train.drop('label',axis=1).values


import sklearn
import lightgbm as lgb

from sklearn.model_selection import train_test_split

train_size = int(len(data)*0.8)
X_valid = data[train_size:]
Y_valid = target[train_size:]
X_train = data[:train_size]
Y_train = target[:train_size]
# 打乱顺序
X_train = np.random.shuffle(X_train)
Y_train = np.random.shuffle(Y_train)


D_train = lgb.Dataset(X_train, Y_train)
D_valid = lgb.Dataset(X_valid, Y_valid)

# LightGBM
params = dict(boosting_type='gbdt',
            objective='binary',
            # max_depth=10,
            # num_leaves=10**2-1,
            learning_rate=0.001,
            min_child_weight=1,
            # colsample_bytree=0.8,
            # colsample_bynode=0.8,
            reg_alpha=0,
            reg_lambda=0.1,
            n_jobs=-1,
            random_state=2020,
            # device='gpu',
            # gpu_platform_id=0,
            # gpu_device_id=0,
            #   first_metric_only=True,
            metric=['binary_error']
            )
eval_results = dict()
num_round = 5000
valid_sets = [D_train, D_valid]
valid_names = ['train', 'valid']
lgb_reg = lgb.train(params,
                    D_train,
                    num_boost_round=num_round,
                    valid_sets=valid_sets,
                    valid_names=valid_names,
                    early_stopping_rounds=50,
                    evals_result=eval_results,
                    verbose_eval=20
                    # early_stopping_rounds=50
                    )