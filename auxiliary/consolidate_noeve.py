import os
import sys
sys.path.append('.')
import numpy as np
import pandas as pd
from shutil import rmtree
from utils import TARGET_COLS

def consolidate_noeve():
    root_path = 'preds/submission'
    noeve_nnlr = os.path.join(root_path,'no_eve_lgb.npy')
    noeve_lgb = os.path.join(root_path,'no_eve_nnlr.npy')
    noeve_nnlr = pd.DataFrame(np.load(noeve_nnlr),columns=TARGET_COLS)
    noeve_lgb = pd.DataFrame(np.load(noeve_lgb),columns=TARGET_COLS)
    
    consolidated_result = noeve_nnlr * 0.65 + noeve_lgb * 0.35
    
    device_id = np.load('device_noeve_id.npy')
    consolidated_result['device_id'] = device_id
    consolidated_result.to_csv('noeve_result.csv',index=False)
    rmtree('preds')
    os.remove('device_noeve_id.npy')

if __name__ == '__main__':
    consolidate_noeve()
    

#%%
from utils import TARGET_COLS
z = np.load('preds/eve_preds.npy')
dids_eve = np.load('inp/events/test/device_eve_id.npy')
dids_noeve = np.load('inp/events/test/device_noeve_id.npy')
dids = np.concatenate([dids_eve,dids_noeve])
z = pd.DataFrame(z,columns=TARGET_COLS)
z['device_id'] = dids
z.to_csv('submission1.csv',index=False)
