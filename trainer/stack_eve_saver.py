import os
import sys
sys.path.append('.')
import pickle
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
import pandas as pd
from helper import ExtraInfo
from trainer.stack_noeve_saver import BaseStackSaver,WholeNoEventStackSaver
from trainer.trainer_utils import auto_detect_col
from sklearn.model_selection import KFold
from auxiliary import Vectorizer,TopicTransformer

class EventStackSaver(WholeNoEventStackSaver):
    def __init__(self,
                 n_folds,
                 data_dict,
                 random_state):
        BaseStackSaver.__init__(self,n_folds=n_folds,
                                data_dict=data_dict,
                                random_state=random_state)
        self.data_dict = data_dict
        self.n_folds = n_folds
        self.random_state = random_state
        self.train = data_dict['train'].copy()
        self.test = data_dict['test'].copy()
        self.train_dids = self.train[['device_id']]
        self.test_dids = self.test[['device_id']]
        
    def _split(self,X):
        x_tr = X[X.index.isin(self.train['device_id'])]
        x_te = X[X.index.isin(self.test['device_id'])]
        return x_tr,x_te
    
    def _transform_and_save(self,X,dids,dids_noeve,mode,ind=None):
        x = self.vect.transform(X).astype(np.float32)
        if self.tp is not None:
            x = self.tp.transform(x).astype(np.float32)
        if hasattr(x,'toarray'):
            x = x.toarray()
        path = f'inp/events/{mode}/{ind}' if ind is not None else f'inp/events/{mode}'
        os.makedirs(path,exist_ok=True)
        
        save_str = os.path.join(path,self.save_name)
        np.save(save_str,x)
        dids_str = os.path.join(path,'device_eve_id.npy')
        if not os.path.exists(dids_str):
            np.save(dids_str,dids)
        dids_noeve_str = os.path.join(path,'device_noeve_id.npy')
        if not os.path.exists(dids_noeve_str):
            np.save(dids_noeve_str,dids_noeve)
    
    def run(self,X,n_components=10,use_topic=False,vectorizer='count',binary=False,min_df=5,prefix='words'):
        X_tr,X_te = self._split(X)
        self.vect = Vectorizer(vectorizer=vectorizer,
                               binary=binary,
                               min_df=min_df)
        X = pd.concat([X_tr,X_te])
        fit_value = self.vect.fit_transform(X)
        self.tp = None
        if use_topic:
            self.tp = TopicTransformer(vectorizer=vectorizer,
                                       binary=binary,
                                       n_components=n_components,
                                       min_df=min_df)
            self.tp.fit(fit_value)
            self.save_name = self.tp.get_feature_name(prefix=prefix)
        else:
            self.save_name = self.vect.get_feature_name(prefix=prefix)
            
        noeve_dids_tr = np.array(list(set(self.train['device_id']) - set(X_tr.index)))
        X_tr = X_tr.reset_index()
        kf = KFold(n_splits=self.n_folds,random_state=self.random_state,shuffle=True)
        for i,((ind_tr,ind_val),(ind_tr_,ind_val_)) in enumerate(zip(kf.split(X_tr),kf.split(noeve_dids_tr))):
            x_tr = X_tr.iloc[ind_tr]
            x_val = X_tr.iloc[ind_val]
            xid_noeve_tr = noeve_dids_tr[ind_tr_]
            xid_noeve_val = noeve_dids_tr[ind_val_]
            xid_eve_tr = np.array(x_tr['device_id'])
            xid_eve_val = np.array(x_val['device_id'])
            target_col = auto_detect_col(x_tr)
            x_tr = x_tr.set_index('device_id')[target_col]
            x_val = x_val.set_index('device_id')[target_col]
            
            self._transform_and_save(x_tr,xid_eve_tr,xid_noeve_tr,mode='train',ind=i)
            self._transform_and_save(x_val,xid_eve_val,xid_noeve_val,mode='val',ind=i)
        xid_eve_te = np.array(list(X_te.index))
        noeve_dids_te = np.array(list(set(self.test['device_id']) - set(X_te.index)))
        self._transform_and_save(X_te,xid_eve_te,noeve_dids_te,mode='test')
    
    def save_extra_attr(self):
        for attr in ['_mean_split','_oh_split']:
            func = getattr(self,attr)
            split_output = func()
            for k,v in split_output.items():
                split_output[k] = v.astype(np.float32)
            with open(f'inp/events/{attr[1:]}.pkl','wb') as f:
                pickle.dump(split_output,f)
                
        ex = ExtraInfo()
        extra_info = ex(self.data_dict)
        with open('inp/events/extrainfo.pkl','wb') as f:
            pickle.dump(extra_info,f)
    

#%%
