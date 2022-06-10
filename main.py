import os
import argparse
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
import pandas as pd
from glob import glob
from utils import (logger,
                   Timer,
                   numpy_metric,
                   TARGET_COLS)
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from trainer import (NoEventStackSaver,
                     WholeNoEventStackSaver,
                     NNTrainer,
                     LRTrainer,
                     LgbTrainer)

def linear_combine(alpha=100,path=''):
    ds_val,ds_te = [],[]
    model = Ridge(alpha=alpha)
    path_val = path
    path_test = path.replace('val','test')

    for file in glob(os.path.join(f'{path_val}/*')):
        if 'label' not in file and file.endswith('.npy'):
            data_val = np.load(file)
            ds_val.append(data_val)
    for file in glob(os.path.join(f'{path_test}/*')):
        data_te = np.load(file)
        ds_te.append(data_te)
    y_val = np.load(os.path.join(path_val,'label.npy'))
    y_te_path = os.path.join(path_test,'label.npy')
    if os.path.exists(y_te_path):
        y_te = np.load(y_te_path)
    else:
        y_te = None

    preds = []
    coefs = np.zeros([len(ds_val)])
    for col in range(12):
        xval,xte = [],[]
        for dval,dte in zip(ds_val,ds_te):
            xval.append(dval[:,col])
            xte.append(dte[:,col])
        xval = np.stack(xval,axis=1)
        xte = np.stack(xte,axis=1)
        y = y_val[:,col]
        model.fit(xval,y)
        coefs += model.coef_
        pred = model.predict(xte)
        preds.append(pred)
    coefs /= len(preds)
    preds = np.stack(preds,axis=1)
    if y_te is not None:
        score = numpy_metric(preds,y_te)
        logger.info(f'staking score:{score:.5f}')
    ds_te.append(y_te)
    ds_val.append(y_val)
        
    return preds,coefs,ds_val,ds_te

def optimize(inits,args):
    ts,label = args[:-1],args[-1]
    pred_matrix = np.zeros([ts[0].shape[0],12])
    for i in range(12):
        col = 0
        for j,(t,init) in enumerate(zip(ts,inits)):
            col += t[:,i] * init
        pred_matrix[:,i] = col
    if label is not None:
        score = numpy_metric(pred_matrix,label)
        return score
    else:
        return pred_matrix,label

def build_dataset(args,base_data):
    running_params = {'batch_size':args.batch_size,
                      'epochs':args.epochs,
                      'lr':args.lr}
    if args.mode == 'eve_submit':
        no_eve_stacker = NoEventStackSaver(n_folds=args.n_folds,
                                           data_dict=base_data,
                                           random_state=args.random_state)
    elif args.mode == 'no_eve_submit':
        no_eve_stacker = WholeNoEventStackSaver(n_folds=args.n_folds,
                                                data_dict=base_data,
                                                random_state=args.random_state)
        
    with Timer(message='start building no_eve features'):
        for split_func in ['le_split','oh_split','mean_split','freq_split']:
            no_eve_stacker.save(split_func)
    
    for config_path in ['nn_noeve_1','nn_noeve_2','nn_noeve_3','nn_noeve_4','nn_noeve_5',
                        'nn_noeve_6','nn_noeve_7','nn_noeve_11']:
        nn_trainer = NNTrainer(data_dict=base_data,
                               n_folds=args.n_folds,
                               config_path=config_path,
                               **running_params)
        nn_trainer.submit(config_path)
    
    for config_path in ['lr_noeve_1','lr_noeve_2','lr_noeve_3','lr_noeve_4']:
        lr_trainer = LRTrainer(data_dict=base_data,
                               n_folds=args.n_folds,
                               config_path=config_path)
        lr_trainer.submit(config_path)
    
    for config_path in ['lgb_noeve_5','lgb_noeve_9','lgb_noeve_10' ]:
        lgb_trainer = LgbTrainer(data_dict=base_data,
                                 n_folds=args.n_folds,
                                 config_path=config_path)
        lgb_trainer.submit(config_path)

def main():
    parser = argparse.ArgumentParser()
    add_argument = parser.add_argument
    add_argument('--n_folds',type=int,default=5)
    add_argument('--random_state',type=int,default=7951)
    add_argument('--mode',type=str,choices=['no_eve_submit','eve_submit'])
    add_argument('--batch_size',type=int,default=128)
    add_argument('--lr',type=float,default=0.001)
    add_argument('--epochs',default=5,type=int)
    add_argument('--optimize_result',action='store_true',help='whether to optimize combined results or \
                 use specified predition')
    add_argument('--pred_path',type=str,help='specific prediction used for submission',default=None)
    add_argument('--save_path',type=str,help='save prediciton at specified location',default=None)
    add_argument('--submit',action='store_true')
    add_argument('--only_dataset',action='store_true')
    add_argument('--pred_store_path',type=str,default='inp/submission/val')
    args = parser.parse_args()

    build_dataset(args,base_data)

    device_ids = np.load('device_id.npy')
    if args.optimize_result:    
        _,coefs,args_val,args_te = linear_combine(100,args.pred_store_path)
        op = minimize(optimize,coefs,args=args_val,method='BFGS')
        preds_te,_ = optimize(op.x,args_te)
    elif args.pred_path:
        preds_te = np.load(f'preds/submission/test/{args.pred_path}.npy')
        preds_te = pd.DataFrame(preds_te,columns=TARGET_COLS)
    else:
        preds_te,*_ = linear_combine(100)
    preds_te = np.maximum(np.minimum(preds_te,1-10**-15),10**-15)
        
    if args.submit:
        preds_te = pd.DataFrame(preds_te,columns=TARGET_COLS)
        preds_te['device_id'] = device_ids
        preds_te.to_csv('submission.csv',index=False)
    else:
        np.save(args.save_path,preds_te)
                       
if __name__ == '__main__':
    main()


#%%
#base_data = produce_base_data(load_data('data'))
#lgb = np.load('preds/submission/lgb.npy')
#nn_lr = np.load('preds/submission/nn_lr.npy')
#sub = lgb * 0.35 + nn_lr * 0.65
#device_id = np.load('device_id.npy')
#sub = pd.DataFrame(sub,columns=TARGET_COLS)
#sub['device_id'] = device_id
#sub.to_csv('submission.csv',index=False)
