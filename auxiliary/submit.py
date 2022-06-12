import os
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from shutil import rmtree
from utils import TARGET_COLS

if __name__ == '__main__':
    eve_preds = np.load('preds/eve_preds.npy')
    dids = np.load('device_eve_id.npy')
    eve_preds = pd.DataFrame(eve_preds,columns=TARGET_COLS)
    eve_preds['device_id'] = dids
    noeve_preds = pd.read_csv('noeve_result.csv')
    submission = pd.concat([noeve_preds,eve_preds])
    submission.to_csv('submission.csv',index=False)
    os.remove('noeve_result.csv')
    os.remove('device_eve_id.npy')
    for path in ['preds','inp','temp_data']:
        rmtree(path)


#%%