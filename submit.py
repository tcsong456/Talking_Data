import os
import argparse
import pandas as pd
import numpy as np
from shutil import rmtree
from utils import TARGET_COLS

def eve_preds_frame():
    eve_preds = np.load('preds/eve_preds.npy')
    dids_eve = np.load('device_eve_id.npy')
    eve_preds = pd.DataFrame(eve_preds,columns=TARGET_COLS)
    eve_preds['device_id'] = dids_eve
    os.remove('device_eve_id.npy')
    return eve_preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_argument = parser.add_argument
    add_argument('--mode',type=str,choices=['final_submit','eve_probe_submit'])
    args = parser.parse_args()
    
    eve_preds = eve_preds_frame()
    if args.mode == 'final_submit':
        noeve_preds = pd.read_csv('noeve_result.csv')
        s = pd.concat([eve_preds,noeve_preds]).set_index('device_id')
        s1 = pd.read_csv('submission_eve.csv',index_col='device_id')
        sub = s * 0.5 + s1 * 0.5
        sub.reset_index().to_csv('submission.csv',index=False)
        for path in ['preds','temp_data','inp']:
            rmtree(path)
        for csv_path in ['noeve_result.csv','submission_eve.csv']:
            os.remove(csv_path)
    elif args.mode == 'eve_probe_submit':
        eve_preds.to_csv('submission_eve.csv',index=False)
        rmtree('preds')


#%%
