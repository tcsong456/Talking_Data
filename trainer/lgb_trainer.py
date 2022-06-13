import os
import json
import numpy as np
from models import LightGBM
from scipy import sparse
from utils import (logger,
                   build_label,
                   numpy_metric)

class BaseTrainer:
    def __init__(self,
                 data_dict,
                 n_folds,
                 config_path):
        with open(f'configs/no_eve_configs/{config_path}.json') as f:
            config = json.load(f)
        self.features = config['features']
        self.train = data_dict['train']
        self.n_folds = n_folds
        self.pred_path = 'preds/submission'
        
    def assemble_features(self,directory,return_label):
        features,mean_features = [],[]
        for feat in self.features:
            if not feat.endswith('.npy'):
                feat += '.npy'
                feat = np.load(os.path.join(directory,feat))
            if 'mean' not in feat:
                features.append(feat)
            else:
                mean_features.append(feat)
        features = np.concatenate(features,axis=1)
        
        if return_label:
            dids = np.load(os.path.join(directory,'device_id.npy'))
            label = build_label(dids,self.train)
            return features,mean_features,label.values
        return features,mean_features
    
    def predict(self,X_tr,y_tr,X_val,X_te,M_tr,M_val,M_te):
        X_tr = sparse.csr_matrix(X_tr) if len(X_tr)>0 else []
        X_val = sparse.csr_matrix(X_val) if len(X_val)>0 else []
        X_te = sparse.csr_matrix(X_te) if len(X_te)>0 else []
        preds_val,preds_te = [],[]
        for i in range(12):
            y = y_tr[:,i]
            if len(M_tr) > 0 and len(M_val) > 0 and len(M_te) > 0:
                for m_tr,m_val,m_te in zip(M_tr,M_val,M_te):
                    mtr = sparse.csr_matrix(m_tr[:,i]).astype(np.float32)
                    mval = sparse.csr_matrix(m_val[:,i]).astype(np.float32)
                    mte = sparse.csr_matrix(m_te[:,i]).astype(np.float32)
                    X_tr = sparse.hstack([X_tr,mtr],axis=1)
                    X_val = sparse.hstack([X_val,mval],axis=1)
                    X_te = sparse.hstack([X_te,mte],axis=1)
                
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
            cur_dir_val = f'{root_path}/val/{i}'
            x_tr,m_tr,y_tr = self.assemble_features(cur_dir_tr,return_label=True)
            x_val,m_val,y_val = self.assemble_features(cur_dir_val,return_label=True)
            tmp[i] = [x_tr,x_val,m_tr]
            labels[i] = [y_tr,y_val,m_val]
        
        cur_dir_te = f'{root_path}/test'
        x_te,m_te = self.assemble_features(cur_dir_te,return_label=False)
        
        self.tmp = tmp
        self.labels = labels
        self.root_path = root_path
        self.x_te = x_te
        self.m_te = m_te
    
    def submit(self,preds_str):
        if not preds_str.endswith('.npy'):
            preds_str += '.npy'
        preds_val,labels_val = [],[]
        preds_te = np.zeros([self.x_te.shape[0],12])
        for i,(X,Y) in enumerate(zip(self.tmp.values(),self.labels.values())):
            x_tr,x_val,m_tr = X
            y_tr,y_val,m_val = Y
            logger.info(f'submission round:{i},shape:{x_tr.shape}')
            
            pred_val,pred_te = self.predict(x_tr,y_tr,x_val,self.x_te,m_tr,m_val,self.m_te)
            preds_val.append(pred_val)
            preds_te += pred_te
            labels_val.append(y_val)
            
        preds_te /= self.n_folds
        preds_val = np.concatenate(preds_val)
        labels_val = np.concatenate(labels_val)
        score_val = numpy_metric(preds_val,labels_val)
        logger.info(f'eval score:{score_val:.5f}')
            
        self.save(preds_str,score_val,preds_val,preds_te)

class LgbTrainer(Trainer):
    def __init__(self,
                 data_dict,
                 n_folds,
                 config_path):
        super().__init__(data_dict=data_dict,
                         n_folds=n_folds,
                         config_path=config_path)
        params = {
                    "num_leaves":20,
                    "min_data_in_leaf":200,
                    "feature_fraction":0.5,
                    "bagging_freq":1,
                    "learning_rate":0.01,
                    "metric":"l2_root",
                    "objective":"regression",
                    "verbose":-1
                    }
        
        self.model = LightGBM(params=params,
                              num_boost_round=200)