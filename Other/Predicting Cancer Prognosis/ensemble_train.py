import os, sys
import random
import argparse
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import dill as pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt
import torchtuples.callbacks as cb

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)

# from modules.earlystoppers import LossEarlyStopper
from modules.utils import load_yaml, get_logger, save_yaml

from xgbse import XGBSEKaplanNeighbors, XGBSEKaplanTree, XGBSEDebiasedBCE, XGBSEBootstrapEstimator, XGBSEStackedWeibull
from xgbse.converters import convert_to_structured
from xgbse.extrapolation import extrapolate_constant_risk

# Ignore some warnings
import warnings
warnings.filterwarnings(action='ignore')

# CONFIG
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)

TRAIN_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/train_config.yaml')
config = load_yaml(TRAIN_CONFIG_PATH)

# DEBUG
DEBUG = 0

# DATA
TRAIN_DATASET = config['DIRECTORY']['train_dataset']
TEST_DATASET = config['DIRECTORY']['test_dataset']

TRAIN_DATA_DIR = os.path.join(PROJECT_DIR, TRAIN_DATASET)
TEST_DATA_DIR = os.path.join(PROJECT_DIR, TEST_DATASET)

# SEED
RANDOM_SEED = config['SEED']['random_seed']

# DATALOADER
VAL_SIZE = config['DATALOADER']['val_size']
NUM_WORKERS = config['DATALOADER']['num_workers']
SHUFFLE = config['DATALOADER']['shuffle']
PIN_MEMORY = config['DATALOADER']['pin_memory']
DROP_LAST = config['DATALOADER']['drop_last']
N_SPLIT = config['DATALOADER']['n_split']

# TRAIN
TREE_METHOD = config['TRAIN']['tree_method']
LEARNING_RATE = config['TRAIN']['learning_rate']
MAX_DEPTH = config['TRAIN']['max_depth']
SUBSAMPLE = config['TRAIN']['subsample']
MIN_CHILD_WEIGHT = config['TRAIN']['min_child_weight']
COLSAMPLE_BYNODE = config['TRAIN']['colsample_bynode']

# TRAIN2
TREE_METHOD2 = config['TRAIN2']['tree_method']
LEARNING_RATE2 = config['TRAIN2']['learning_rate']
MAX_DEPTH2 = config['TRAIN2']['max_depth']
SUBSAMPLE2 = config['TRAIN2']['subsample']
MIN_CHILD_WEIGHT2 = config['TRAIN2']['min_child_weight']
COLSAMPLE_BYNODE2 = config['TRAIN2']['colsample_bynode']

#EARLY_STOPPING_PATIENCE = config['TRAIN']['early_stopping_patience']

# columns
PREPROCESS = config['COLUMNS']['preprocess']
COLUMNS = config['COLUMNS']['cols']

# label encoding
LABELS_ENCODING = config['LABEL_ENCODING']

DEFAULT_PARAMS = {
                    "objective": "survival:aft",
                    "eval_metric": "aft-nloglik",
                    "aft_loss_distribution": "normal",
                    "aft_loss_distribution_scale": 1,
                    "tree_method": TREE_METHOD,
                    "learning_rate": LEARNING_RATE,
                    "max_depth": MAX_DEPTH,
                    "booster": "dart",
                    "subsample": SUBSAMPLE,
                    "min_child_weight": MIN_CHILD_WEIGHT,
                    "colsample_bynode": COLSAMPLE_BYNODE,
                }

DEFAULT_PARAMS2 = {
                    "objective": "survival:aft",
                    "eval_metric": "aft-nloglik",
                    "aft_loss_distribution": "normal",
                    "aft_loss_distribution_scale": 1,
                    "tree_method": TREE_METHOD2,
                    "learning_rate": LEARNING_RATE2,
                    "max_depth": MAX_DEPTH2,
                    "booster": "dart",
                    "subsample": SUBSAMPLE2,
                    "min_child_weight": MIN_CHILD_WEIGHT2,
                    "colsample_bynode": COLSAMPLE_BYNODE2,
                }

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    KST = timezone(timedelta(hours=9))
    parser.add_argument('--train_serial', type=str, default = datetime.now(tz=KST).strftime("%Y%m%d_%H%M%S"))
    #parser.add_argument('--train_serial', type=str, default = datetime.now().strftime("%Y%m%d_%H%M%S"))
    args=parser.parse_args()

    ### Set random seed ###
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    ### Set device ###
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Set result directory ###
    result_dir = os.path.join(PROJECT_DIR, 'results', args.train_serial)
    os.makedirs(result_dir, exist_ok=True)

    ### Set system logger ###
    system_logger = get_logger(name='train', file_path=os.path.join(result_dir, 'train.log'))


    ### Load data ###
    df = pd.read_csv(TRAIN_DATA_DIR)
    df_test = pd.read_csv(TEST_DATA_DIR)

    # preprocess
    df.replace(LABELS_ENCODING, inplace = True)
    df_test.replace(LABELS_ENCODING, inplace = True)

    if PREPROCESS:
        for i in range(len(df)):
            if pd.isna(df.iloc[i]['height']):
                if df.iloc[i]['sex'] == 0:
                    df.iloc[i, df.columns.get_loc('height')] = df[df['sex'] == 0]['height'].mean()
                else:
                    df.iloc[i, df.columns.get_loc('height')] = df[df['sex'] == 1]['height'].mean()

            if pd.isna(df.iloc[i]['weight']):
                if df.iloc[i]['sex'] == 0:
                    df.iloc[i, df.columns.get_loc('weight')] = df[df['sex'] == 0]['weight'].mean()
                else:
                    df.iloc[i, df.columns.get_loc('weight')] = df[df['sex'] == 1]['weight'].mean()

        # add bmi feature
        df['bmi'] = df['weight'] / (df['height'] / 100) ** 2


        for i in range(len(df_test)):
            if pd.isna(df_test.iloc[i]['height']):
                if df_test.iloc[i]['sex'] == 0:
                    df_test.iloc[i, df_test.columns.get_loc('height')] = df[df['sex'] == 0]['height'].mean()
                else:
                    df_test.iloc[i, df_test.columns.get_loc('height')] = df[df['sex'] == 1]['height'].mean()

            if pd.isna(df_test.iloc[i]['weight']):
                if df_test.iloc[i]['sex'] == 0:
                    df_test.iloc[i, df_test.columns.get_loc('weight')] = df[df['sex'] == 0]['weight'].mean()
                else:
                    df_test.iloc[i, df_test.columns.get_loc('weight')] = df[df['sex'] == 1]['weight'].mean()

        # add bmi feature
        df_test['bmi'] = df_test['weight'] / (df_test['height'] / 100) ** 2

        COLUMNS.append('bmi')

    columns = [(col, None) for col in COLUMNS]

    x_mapper = DataFrameMapper(columns)

    ##############################################################################
    ##########################   STRATIFY_FOLD    ################################
    ##############################################################################

    skf = StratifiedKFold(n_splits=N_SPLIT, shuffle=SHUFFLE, random_state=RANDOM_SEED)
    result = []
    valid_score = []

    # Save config yaml file
    save_yaml(os.path.join(result_dir, 'train_config.yaml'), config)
    n = 0

    print(DEFAULT_PARAMS)
    system_logger.info(f"Params\n{DEFAULT_PARAMS}")


    for df_train_index, df_val_index in skf.split(df, df['dead']):
        df_train = df.iloc[df_train_index]
        df_val = df.iloc[df_val_index]

        print(f"{n}th train\nLoad dataset, train: {len(df_train)}, val: {len(df_val)}")
        system_logger.info(f"{n}th train\nLoad dataset, train: {len(df_train)}, val: {len(df_val)}")

        x_train = pd.DataFrame(x_mapper.fit_transform(df_train).astype('float32'))
        x_val = pd.DataFrame(x_mapper.fit_transform(df_val).astype('float32'))
        x_test = pd.DataFrame(x_mapper.fit_transform(df_test).astype('float32'))

        y_train = convert_to_structured(df_train['duration'], df_train['dead'])
        y_valid = convert_to_structured(df_val['duration'], df_val['dead'])

        get_target = lambda df: (df['duration'].values, df['dead'].values)
        y_val = get_target(df_val)

        ### Train ###
        xgbse = XGBSEDebiasedBCE(xgb_params = DEFAULT_PARAMS)

        xgbse_model = XGBSEBootstrapEstimator(xgbse, n_estimators=20)

        xgbse_model.fit(x_train,
            y_train,
            validation_data=(x_val, y_valid),
            early_stopping_rounds=50,
            verbose_eval = 200,
            )

        surv = xgbse_model.predict(x_val).transpose()

        ss_val = pd.DataFrame(index=list(range(int(max(surv.index) + 1))))
        ss_val = ss_val.join(surv)
        ss_val.fillna(method = 'bfill',inplace = True)

        # y_val[0]: duration , y_val[1]: event
        ev = EvalSurv(ss_val, y_val[0], y_val[1], censor_surv='km')
        score = ev.concordance_td(method='antolini')
        print("c-index : " , score)
        system_logger.info(f"c-index : {score}")
        valid_score.append(score)

        ### Test ###
        surv = xgbse_model.predict(x_test).transpose()
        #surv = xgbse_model.predict(x_test)
        #surv = extrapolate_constant_risk(surv, 5500, 10).transpose()

        ss_test = pd.DataFrame(index=list(range(5500+1)))
        ss_test = ss_test.join(surv)
        ss_test.fillna(method = 'bfill',inplace = True)
        ss_test.fillna(method = 'ffill',inplace = True)

        result.append(ss_test)
        n += 1

    print(valid_score)

    print("Average c-index : ", sum(valid_score)/len(valid_score))
    system_logger.info(f"Average c-index : {sum(valid_score)/len(valid_score)}")

    ### Save df ###
    ss_test = result[0]
    for i in range(1, len(result)):
        ss_test = ss_test.add(result[i])

    ss_test = ss_test.div(len(result))
    ss_test = ss_test.round(4)
    ss_test.rename(columns = lambda x: "ID_" + str(x) , inplace =True)
    ss_test.to_csv(os.path.join(result_dir,'result_df.csv'),index_label = 'duration')