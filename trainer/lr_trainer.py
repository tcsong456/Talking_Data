import os
import json
import pickle
import numpy as np
import pandas as pd
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
        self.train = data_dict['train']
        self.n_folds = n_folds
        self.pred_path = 'preds/submission'
        root_path='inp/no_events'
        self.rooth_path = root_path
        
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

    def submit(self,preds_str):
        if not preds_str.endswith('.npy'):
            preds_str += '.npy'
        preds_val,labels_val = [],[]
        preds_te = np.zeros([self.x_te.shape[0],12])
        for i,(X,Y) in enumerate(zip(self.tmp.values(),self.labels.values())):
            x_tr,x_val = X
            y_tr,y_val = Y
            logger.info(f'submission round:{i},shape:{x_tr.shape}')
            
            pred_val,pred_te = self.predict(x_tr,y_tr,x_val,self.x_te)
            preds_val.append(pred_val)
            preds_te += pred_te
            labels_val.append(y_val)
            
        preds_te /= self.n_folds
        preds_val = np.concatenate(preds_val)
        labels_val = np.concatenate(labels_val)
        score_val = numpy_metric(preds_val,labels_val)
        logger.info(f'eval score:{score_val:.5f}')
            
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
        with open(f'configs/no_eve_configs/{config_path}.json') as f:
            config = json.load(f)
        self.features = config['features']
        
        tmp,labels = {},{}
        self.pred_path = 'preds/submission'
        
        for i in range(n_folds):
            cur_dir_tr = f'{self.root_path}/train/{i}'
            cur_dir_val = f'{self.root_path}/val/{i}'
            x_tr,y_tr = self.assemble_features(cur_dir_tr,return_label=True)
            x_val,y_val = self.assemble_features(cur_dir_val,return_label=True)
            tmp[i] = [x_tr,x_val]
            labels[i] = [y_tr,y_val]
        
        cur_dir_te = f'{self.root_path}/test'
        x_te = self.assemble_features(cur_dir_te,return_label=False)
        
        self.tmp = tmp
        self.labels = labels
        self.x_te = x_te

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

class LRProber(BaseTrainer):
    def __init__(self,
                 data_dict,
                 n_folds,
                 config_path):
        super().__init__(data_dict=data_dict,
                         n_folds=n_folds,
                         config_path=config_path)
        self.root_path = 'inp/events'
        tmp,labels = {},{}
        self.pred_path = 'preds/submission'
        with open(f'configs/eve_configs/{config_path}.json') as f:
            config = json.load(f)
        self.features = config['features']
        self.extrainfo = config['extrainfo']
        params = config['params']
        self.aux_attrs = {}
        for file in ['extrainfo','mean_split','oh_split']:
            file += '.pkl'
            load_dir = os.path.join(self.root_path,file)
            with open(load_dir,'rb') as f:
                self.aux_attrs[file[:-4]] = pickle.load(f)
        
        for i in range(n_folds):
            cur_dir_tr = f'{self.root_path}/train/{i}'
            cur_dir_val = f'{self.root_path}/val/{i}'
            x_tr,y_tr = self.assemble_features(cur_dir_tr,return_label=True,mode='train')
            x_val,y_val = self.assemble_features(cur_dir_val,return_label=True,mode='train')
            tmp[i] = [x_tr,x_val]
            labels[i] = [y_tr,y_val]
        cur_dir_te = f'{self.root_path}/test'
        x_te = self.assemble_features(cur_dir_te,return_label=False,mode='test')
        
        self.tmp = tmp
        self.labels = labels
        self.x_te = x_te
        self.model = LR(**params)
    
    def _generate_extra_features(self,x,dids,mode):
        temp_features = []
        temporal_feat = self.extrainfo['temporal']
        for temp in temporal_feat:
            temp_features.append(self.aux_attrs['extrainfo'][temp])
        temp_features = pd.concat(temp_features,axis=1)
        temp_features = dids.merge(temp_features,how='left',on='device_id').set_index('device_id').fillna(0).values
        
        brand_features = []
        brand_feat = self.extrainfo['brand']
        suffix = '_tr' if mode=='train' else '_te'
        for f in brand_feat:
            f_name = f.split('_',maxsplit=1)[1] + suffix
            bf = self.aux_attrs['oh_split'][f_name]
            brand_features.append(bf)
        brand_features = pd.concat(brand_features,axis=1)
        brand_features = dids.merge(bf,how='left',on='device_id').set_index('device_id').fillna(0).values
        features = np.concatenate([x,temp_features,brand_features],axis=1)
        return features
    
    def assemble_features(self,directory,return_label,mode):
        dids_eve = np.load(os.path.join(directory,'device_eve_id.npy'))
        dids_eve = pd.DataFrame(dids_eve,columns=['device_id'])
        dids_noeve = np.load(os.path.join(directory,'device_noeve_id.npy'))
        dids_noeve = pd.DataFrame(dids_noeve,columns=['device_id'])
        dids = pd.concat([dids_eve,dids_noeve])
        
        features = []
        for feat in self.features:
            if not feat.endswith('.npy'):
                feat += '.npy'
            feat = np.load(os.path.join(directory,feat))
            features.append(feat)
        topic_features = np.concatenate(features,axis=1)
        missing_len = dids_noeve.shape[0]
        zeros_fill = np.zeros([missing_len,topic_features.shape[1]],dtype=np.float32)
        topic_features = np.concatenate([topic_features,zeros_fill])
        
        features = self._generate_extra_features(topic_features,dids,mode)
        
        if return_label:
            label = build_label(dids,self.train)
            return features,label.values
        return features

#%%
