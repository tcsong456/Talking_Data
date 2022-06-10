import os
import json
import numpy as np
from models import LR
from scipy import sparse
from utils import (logger,
                   build_label,
                   numpy_metric)

class BaseTrainer:
    def __init__(self,
                 data_dict,
                 n_folds,
                 config_path):
        with open(f'configs/trainer_configs/{config_path}.json') as f:
            config = json.load(f)
        self.features = config['features']
        self.train = data_dict['train']
        self.n_folds = n_folds
        self.pred_path = 'preds/submission'
        
    def assemble_features(self,directory,return_label):
        features = []
        for feat in self.features:
            if not feat.endswith('.npy'):
                feat += '.npy'
            feat = np.load(os.path.join(directory,feat))
            features.append(feat)
        features = np.concatenate(features,axis=1)
        
        if return_label:
            dids = np.load(os.path.join(directory,'device_id.npy'))
            label = build_label(dids,self.train)
            return features,label.values
        return features
    
    def predict(self,X_tr,y_tr,X_val,X_te):
        X_tr = sparse.csr_matrix(X_tr)
        X_val = sparse.csr_matrix(X_val)
        X_te = sparse.csr_matrix(X_te)
        preds_val,preds_te = [],[]
        for i in range(12):
            y = y_tr[:,i]
            self.model.fit(X_tr,y)
            pred_val = self.model.predict(X_val)
            pred_te = self.model.predict(X_te)
            preds_val.append(pred_val)
            preds_te.append(pred_te)
        preds_val = np.stack(preds_val,axis=1)
        preds_te = np.stack(preds_te,axis=1)
        return preds_val,preds_te
    
    def save(self,preds_str,score_val,preds_val,preds_te):
        for path,data in zip([f'{self.pred_path}/val',f'{self.pred_path}/test'],[preds_val,preds_te]):
            os.makedirs(path,exist_ok=True)
            logger.info(f'saveing predictions to {path}')
            preds_str_ = preds_str[:-4] + f'_{score_val:.5f}.npy'
            np.save(os.path.join(path,preds_str_),data)

class Trainer(BaseTrainer):
    def __init__(self,
                 data_dict,
                 n_folds,
                 config_path):
        super().__init__(data_dict=data_dict,
                         n_folds=n_folds,
                         config_path=config_path)
        tmp,labels = {},{}
        root_path='inp/no_events'
        self.pred_path = 'preds/submission'
        
        for i in range(n_folds):
            cur_dir_tr = f'{root_path}/train/{i}'
            cur_dir_val = f'{root_path}/eval/{i}'
            x_tr,y_tr = self.assemble_features(cur_dir_tr,return_label=True)
            x_val,y_val = self.assemble_features(cur_dir_val,return_label=True)
            tmp[i] = [x_tr,x_val]
            labels[i] = [y_tr,y_val]
        
        cur_dir_te = f'{root_path}/test'
        x_te = self.assemble_features(cur_dir_te,return_label=False)
        
        self.tmp = tmp
        self.labels = labels
        self.root_path = root_path
        self.x_te = x_te
    
    def submit(self,preds_str):
        if not preds_str.endswith('.npy'):
            preds_str += '.npy'
        preds_val,labels_val = [],[]
        preds_te = np.zeros([self.x_te.shape[0],12])
        for i,(X,Y) in enumerate(zip(self.tmp.values(),self.labels.values())):
            x_tr,x_val = X
            y_tr,y_val = Y
            logger.info(f'submission round:{i},shape:{x_tr.shape}')
            
            pred_val,pred_te = self.predict(x_tr,y_tr,x_val,self.x_te,)
            preds_val.append(pred_val)
            preds_te += pred_te
            labels_val.append(y_val)
            
        preds_te /= self.n_folds
        preds_val = np.concatenate(preds_val)
        labels_val = np.concatenate(labels_val)
        score_val = numpy_metric(preds_val,labels_val)
        logger.info(f'eval score:{score_val:.5f}')
            
        self.save(preds_str,score_val,preds_val,preds_te)

class LRTrainer(Trainer):
    def __init__(self,
                 data_dict,
                 n_folds,
                 config_path):
        super().__init__(data_dict=data_dict,
                         n_folds=n_folds,
                         config_path=config_path)
        self.model = LR(penalty='l2',
                        C=0.05,
                        max_iter=30,
                        solver='lbfgs')

#%%
