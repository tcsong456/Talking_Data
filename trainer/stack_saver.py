import os
import numpy as np
import pandas as pd
from collections import Counter
from utils import logger
from helper.helper_utils import (map_label,
                                 assemble_mean_encodes)
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from trainer.process_pbrand_model import find_middle

class BaseStackSaver:
    def __init__(self,
                 n_folds,
                 data_dict):
        self.n_folds = n_folds
        self.data_dict = data_dict
        train = self.data_dict['train']
        test = self.data_dict['test']
        valid_device_id = self._generate_device_id()
        self.valid_device_id = valid_device_id
        self.train_dids = train.merge(valid_device_id,on='device_id')['device_id'].tolist()
        self.train_no_dids = np.array(list(set(train['device_id']) - set(self.train_dids)))
        self.test_dids = test.merge(valid_device_id,on='device_id')['device_id'].tolist()
        self.test_no_dids = set(test['device_id']) - set(self.test_dids)
        self.phone_brand = data_dict['phone_brand'].set_index('device_id')
        self._make_dirs()
        
        self.train = train
        self.test = test
        
    def _generate_device_id(self):
        event_ids = self.data_dict['app_events'].groupby('event_id').size().to_frame()
        events_app = self.data_dict['events'].merge(event_ids,on='event_id')
        dids = events_app.groupby('device_id').size().reset_index()[['device_id']]
        return dids
    
    def _make_dirs(self):
        os.makedirs('inp/events',exist_ok=True)
        os.makedirs('inp/no_events',exist_ok=True)
        for sub_folder in ['train','eval','test']:
            for cur in ['inp/events','inp/no_events']:
                build_folder = os.path.join(cur,sub_folder)
                os.makedirs(build_folder,exist_ok=True)

class NoEventStackSaver(BaseStackSaver):
    def __init__(self,
                 n_folds,
                 data_dict,
                 random_state):
        super().__init__(n_folds=n_folds,
                         data_dict=data_dict)
        self.random_state = random_state
        self.target_cols = ['M32-38', 'F27-28', 'M39+', 'M23-26', 'M27-28', 'M29-31', 'F33-42',
                            'F43+', 'F24-26', 'F23-', 'F29-32', 'M22-']
        self.split_func_maps = {'oh_split':self._oh_split,
                                'le_split':self._le_split,
                                'mean_split':self._mean_split,
                                'freq_split':self._freq_split}
    
    def _label(self,dids):
        if not type(dids) is pd.DataFrame:
            dids = pd.DataFrame(dids,columns=['device_id'])
        labels = dids.merge(self.train,how='left',on='device_id')
        unique_group = np.array(self.target_cols)
        labels_ = pd.DataFrame(np.zeros([labels.shape[0],len(unique_group)]),columns=unique_group)
        labels = map_label(unique_group,labels,labels_)
        labels['device_id'] = list(dids.values.flatten())
        labels.set_index('device_id',inplace=True)
        return labels
    
    def _train_test_split(self,pbrand,dmodel,slevel):
        phone_brand_tr = pbrand[pbrand.index.isin(self.train_no_dids)]
        device_model_tr = dmodel[dmodel.index.isin(self.train_no_dids)]
        second_level_tr = slevel[slevel.index.isin(self.train_no_dids)]
        phone_brand_te = pbrand[pbrand.index.isin(self.test_no_dids)]
        device_model_te = dmodel[dmodel.index.isin(self.test_no_dids)]
        second_level_te = slevel[slevel.index.isin(self.test_no_dids)]
        
        return {'phone_brand_tr':phone_brand_tr,
                'device_model_tr':device_model_tr,
                'second_level_tr':second_level_tr,
                'phone_brand_te':phone_brand_te,
                'device_model_te':device_model_te,
                'second_level_te':second_level_te}
    
    def _oh_split(self):
        oh_phone_brand = pd.get_dummies(self.phone_brand['phone_brand'])
        oh_device_model = pd.get_dummies(self.phone_brand['device_model'])
        ph = self.phone_brand.copy()
        ph = find_middle(ph)
        ph['device_model'] = ph['device_model'].map(lambda row:row.split()[0])
        oh_second_level=  pd.get_dummies(ph['device_model'])
        oh_encoding_dict = self._train_test_split(oh_phone_brand,oh_device_model,oh_second_level)
        return oh_encoding_dict
    
    def _le_split(self):
        le = LabelEncoder()
        ph = self.phone_brand.copy()
        ph['le_phone_brand'] = le.fit_transform(ph['phone_brand'])
        ph['le_device_model'] = le.fit_transform(ph['device_model'])
        ph = find_middle(ph)
        ph['device_model'] = ph['device_model'].map(lambda row:row.split()[0])
        ph['le_second_level'] = le.fit_transform(ph['device_model'])
        le_encodeing_dict = self._train_test_split(ph['le_phone_brand'],ph['le_device_model'],ph['le_second_level'])
        return le_encodeing_dict
    
    def _mean_split(self):
        label = self._label(self.train.device_id)
        label_ori = label.merge(self.phone_brand,how='left',on='device_id').reset_index()
        test = self.test.copy()
        test_ori = test.merge(self.phone_brand,how='left',on='device_id').reset_index()
        ph_encodes = assemble_mean_encodes(label_ori,test_ori,'phone_brand',self.target_cols)
        demodel_encodes = assemble_mean_encodes(label_ori,test_ori,'device_model',self.target_cols)
        
        pbrand = self.phone_brand.copy()
        pbrand = find_middle(pbrand)
        pbrand['second_level'] = pbrand['device_model'].map(lambda row:row.split()[0])
        label_new = label.merge(pbrand,how='left',on='device_id').reset_index()
        test_new = test.merge(pbrand,how='left',on='device_id').reset_index()
        slevel_encodes = assemble_mean_encodes(label_new,test_new,'second_level',self.target_cols)
        mean_encoding_dict = self._train_test_split(ph_encodes,demodel_encodes,slevel_encodes)
        return mean_encoding_dict
    
    def _freq_split(self):
        events = self.data_dict['events'].copy()
        ph = self.phone_brand.copy()
        freq_pbrand_dict = Counter(events['phone_brand'])
        ph['pbrand'] = ph['phone_brand'].map(freq_pbrand_dict)
        freq_dmodel_dict = Counter(events['device_model'])
        ph['dmodel'] = ph['device_model'].map(freq_dmodel_dict)
        ph = find_middle(ph).rename(columns={'device_model':'second_level'})
        ph['second_level'] = ph['second_level'].map(lambda row:row.split()[0])
        events = events.merge(ph['second_level'],how='left',on='device_id')
        freq_sl_dict = Counter(events['second_level'])
        ph['slevel'] = ph['second_level'].map(freq_sl_dict)
        freq_encoding_dict = self._train_test_split(ph['pbrand'],ph['dmodel'],ph['slevel'])
        return freq_encoding_dict
    
    def _save_file(self,value,name,cur_dir,dids=None):
        if dids is not None:
            v = value[value.index.isin(dids)]
        else:
            v = value
        cur_save_name = os.path.join(cur_dir,name)
        if hasattr(v,'values'):
            v = v.values
        np.save(cur_save_name,v)
    
    def save(self,split_func):
        logger.info(f'saving {split_func} features')
        split_data_dict = self.split_func_maps[split_func]()
        kf = KFold(n_splits=self.n_folds,random_state=self.random_state,shuffle=False)
        for i,(ind_tr,ind_val) in enumerate(kf.split(self.train_no_dids)):
            cur_dir_tr = f'inp/no_events/train/{i}'
            cur_dir_val = f'inp/no_events/eval/{i}'
            os.makedirs(cur_dir_tr,exist_ok=True)
            os.makedirs(cur_dir_val,exist_ok=True)
            
            x_dids_tr = self.train_no_dids[ind_tr]
            x_dids_val = self.train_no_dids[ind_val]
            for k,v in split_data_dict.items():
                v = v.astype(np.float32)
                if 'tr' in k:
                    pos = k.find('tr')
                    name = k[:pos-1] + '_' + split_func + '.npy'
                    self._save_file(v,name,cur_dir_tr,x_dids_tr)
                    self._save_file(v,name,cur_dir_val,x_dids_val)
            
            for v,d in zip([x_dids_tr,x_dids_val],[cur_dir_tr,cur_dir_val]):
                self._save_file(v,'device_id.npy',d)
            
        cur_dir_te = 'inp/no_events/test'
        for k,v in split_data_dict.items():
            if 'te' in k:
                pos = k.find('te')
                name = k[:pos-1] + '.npy'
                self._save_file(v.astype(np.float32),name,cur_dir_te)
        self._save_file(np.array(list(self.test_no_dids)),'device_id.npy',cur_dir_te)

    #%%
