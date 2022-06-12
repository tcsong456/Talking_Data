import os
import json
import pickle
import numpy as np
import pandas as pd
from torch.optim import Adam
from models import NN
from utils import (logger,
                   tensor_metric,
                   numpy_metric,
                   build_label)
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import MinMaxScaler

class Ds(Dataset):
    def __init__(self,
                 data,
                 device_ids,
                 labels=None):
        self.data = data
        self.device_ids = device_ids
        self.labels = labels
    
    def __getitem__(self,index):
        d = self.data[index]
        ids = self.device_ids[index]
        if self.labels is not None:
            y = self.labels.loc[ids].values
            return d,y
        else:
            return d,np.array(ids)
    
    def __len__(self):
        return len(self.data)
    
class NNBase:
    def __init__(self,
                 data_dict,
                 n_folds,
                 config_path,
                 batch_size=128,
                 lr=0.001,
                 epochs=5):
        self.train = data_dict['train'].copy()
        self.test = data_dict['test'].copy()
        self.n_folds = n_folds
        self.pred_path = 'preds/submission'
        self.save_device_name = ''
        
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        
    def train_(self,dl,epoch):
        train_loss = 0
        for step,batch in enumerate(dl):
            b,y = batch;b,y = b.cuda(),y.cuda()
            preds = self.model(b)
            loss = tensor_metric(preds,y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        avg_loss = train_loss / (step + 1)
        logger.info(f'epoch:{epoch} train_loss:{avg_loss:.5f}')
    
    def predict(self,dl):
        total_loss = 0
        ps,ys = [],[]
        for step,batch in enumerate(dl):
            b,y = batch;b,y = b.cuda(),y.cuda()
            preds = self.model(b)
            ps.append(preds.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
            loss = tensor_metric(preds,y)
            total_loss += loss.item()
        avg_loss = total_loss / (step + 1)
        ps = np.concatenate(ps)
        ys = np.concatenate(ys)
        return ps,ys,avg_loss
    
    def predict_wo_label(self,dl):
        ps,dids = [],[]
        for step,batch in enumerate(dl):
            b,ids = batch;b = b.cuda()
            preds = self.model(b)
            dids.append(ids)
            ps.append(preds.detach().cpu().numpy())
        ps = np.concatenate(ps)
        dids = np.concatenate(dids).flatten()
        return ps,dids
    
    def build_loader(self,x,y=None,ind=None,mode='train',shuffle=False):
        if ind is not None:
            device_id_path = os.path.join(self.root_path,mode,str(ind))
        else:
            device_id_path = os.path.join(self.root_path,mode)
            
        dids_eve = np.load(os.path.join(device_id_path,'device_eve_id.npy'))
        dids_noeve = np.load(os.path.join(device_id_path,'device_noeve_id.npy'))
        dids = np.concatenate([dids_eve,dids_noeve])
        
        dataset = Ds(x,dids,y)
        dl = DataLoader(dataset,batch_size=self.batch_size,shuffle=shuffle)
        return dl
    
    def save(self,preds_str,score_val,preds_val,preds_te,label_name,labels_val,device_id):
        for path,data,label in zip([f'{self.pred_path}/val',f'{self.pred_path}/test'],[preds_val,preds_te],
                                   [labels_val,[]]):
            os.makedirs(path,exist_ok=True)
            logger.info(f'saveing predictions to {path}')
            preds_str_ = preds_str[:-4] + f'_{score_val:.5f}.npy'
            np.save(os.path.join(path,preds_str_),data)
            if len(label) > 0:
                if not label_name.endswith('.npy'):
                    label_name += '.npy'
                label_path = os.path.join(path,label_name)
                if not os.path.exists(label_path):
                    np.save(label_path,label)
        if not os.path.exists(self.save_device_name):
            np.save(self.save_device_name,device_id)
    
    def submit(self,preds_str):
        if not preds_str.endswith('.npy'):
            preds_str += '.npy'
        preds_te = np.zeros([self.x_te.shape[0],12])
        preds_val,labels_val = [],[]
        
        dl_te = self.build_loader(self.x_te,mode='test',shuffle=False)
        for i,(x,y) in enumerate(zip(self.X.values(),self.Y.values())):
            x_tr,x_val = x
            y_tr,y_val = y
            logger.info(f'submission round:{i},shape:{x_tr.shape}')
            
            input_size = x_tr.shape[1]
            self.model = NN(input_dim=input_size,
                            hidden_size_1=512,
                            hidden_size_2=128,
                            output_size=12).cuda()
            params = filter(lambda x:x.requires_grad,self.model.parameters())
            self.optimizer = Adam(params,lr=self.lr)
            
            dl_tr = self.build_loader(x_tr,y_tr,ind=i,mode='train',shuffle=True)
            dl_val = self.build_loader(x_val,y_val,ind=i,mode='val',shuffle=False)
            
            self.best_loss = np.inf
            for epoch in range(self.epochs):
                self.train_(dl_tr,epoch=epoch)
                pred_val,yval,eval_loss = self.predict(dl=dl_val)
                pred_te,device_id = self.predict_wo_label(dl=dl_te)
                logger.info(f'epoch:{epoch} eval loss:{eval_loss:.5f}\n')
                
                if eval_loss < self.best_loss:
                    ps_val = np.zeros([pred_val.shape[0],12])
                    ps_te = np.zeros([pred_te.shape[0],12])
                    for _ in range(10):
                        pred_val,_,_ = self.predict(dl=dl_val)
                        ps_val += pred_val
                    for _ in range(5):
                        pred_te,_ = self.predict_wo_label(dl=dl_te)
                        ps_te += pred_te
                    ps_val /= 5;ps_te /= 3
                    self.best_loss = eval_loss
                    self.best_preds_val = ps_val
                    self.best_preds_te = pred_te
            preds_val.append(self.best_preds_val)
            labels_val.append(yval)
            preds_te += self.best_preds_te
        
        preds_te /= self.n_folds
        preds_val = np.concatenate(preds_val)
        labels_val = np.concatenate(labels_val)
        score_val = numpy_metric(preds_val,labels_val)
        logger.info(f'eval score:{score_val:.5f}')
            
        self.save(preds_str,score_val,preds_val,preds_te,'label',labels_val,device_id)

class NNTrainer(NNBase):
    def __init__(self,
                 data_dict,
                 n_folds,
                 config_path,
                 batch_size=128,
                 lr=0.001,
                 epochs=5):
        super().__init__(data_dict=data_dict,
                         n_folds=n_folds,
                         config_path=config_path,
                         batch_size=batch_size,
                         lr=lr,
                         epochs=epochs)
        with open(f'configs/no_eve_configs/{config_path}.json') as f:
            config = json.load(f)
        self.features = config['features']
        root_path='inp/no_events'
        self.save_device_name = 'device_noeve_id.npy'
        self.X,self.Y = {},{}
        for i in range(n_folds):
            cur_dir_tr = f'{root_path}/train/{i}'
            cur_dir_val = f'{root_path}/eval/{i}'
            
            x_tr,y_tr = self._assemble_features(cur_dir_tr,return_label=True)
            x_val,y_val = self._assemble_features(cur_dir_val,return_label=True)
            self.X[i] = [x_tr,x_val]
            self.Y[i] = [y_tr,y_val]
        
        cur_dir_te = f'{root_path}/test'
        x_te = self._assemble_features(cur_dir_te,return_label=False)
        self.x_te = x_te
        self.root_path = root_path
        
    def _assemble_features(self,directory,return_label):
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
            return features,label
        return features

class NNEveTrainer(NNBase):
    def __init__(self,
             data_dict,
             n_folds,
             config_path,
             batch_size=128,
             lr=0.001,
             epochs=5):
        super().__init__(data_dict=data_dict,
                         n_folds=n_folds,
                         config_path=config_path,
                         batch_size=batch_size,
                         lr=lr,
                         epochs=epochs)
        root_path='inp/events'
        self.save_device_name = 'device_eve_id.npy'
        self.X,self.Y = {},{}
        with open(f'configs/eve_configs/{config_path}.json') as f:
            config = json.load(f)
        self.features = config['features']
        self.extrainfo = config['extrainfo']
        
        self.aux_attrs = {}
        for file in ['extrainfo','mean_split','oh_split']:
            file += '.pkl'
            load_dir = os.path.join(root_path,file)
            with open(load_dir,'rb') as f:
                self.aux_attrs[file[:-4]] = pickle.load(f)
        
        for i in range(self.n_folds):
            cur_dir_tr = f'{root_path}/train/{i}'
            cur_dir_te = f'{root_path}/val/{i}'
            x_tr,y_tr = self._assemble_features(cur_dir_tr,return_label=True,mode='train')
            x_val,y_val = self._assemble_features(cur_dir_te,return_label=True,mode='train')
            self.X[i] = [x_tr,x_val]
            self.Y[i] = [y_tr,y_val]
        cur_dir_te = f'{root_path}/test'
        x_te = self._assemble_features(cur_dir_te,return_label=False,mode='test')
        self.x_te = x_te
        self.root_path = root_path
    
    def _generate_extra_features(self,x,dids,mode):
        mms = MinMaxScaler()
        temp_features,temp_mms_features = [],[]
        temporal_feat = self.extrainfo['temporal']
        temporal_mms_feat = self.extrainfo['temporal_mms']
        for temp,temp_mms in zip(temporal_feat,temporal_mms_feat):
            temp_features.append(self.aux_attrs['extrainfo'][temp])
            temp_mms_features.append(self.aux_attrs['extrainfo'][temp_mms])
        temp_features = pd.concat(temp_features,axis=1)
        temp_mms_features = pd.concat(temp_mms_features,axis=1)
        temp_features = dids.merge(temp_features,how='left',on='device_id').set_index('device_id').fillna(0).values
        temp_mms_features = dids.merge(temp_mms_features,how='left',on='device_id').set_index('device_id').fillna(0)
        temp_mms_features = mms.fit_transform(temp_mms_features)
        
        brand_features = []
        brand_feat = self.extrainfo['brand']
        suffix = '_tr' if mode=='train' else '_te'
        for f in brand_feat:
            f_name = f.split('_',maxsplit=1)[1] + suffix
            if 'oh' in f:
                bf = self.aux_attrs['oh_split'][f_name]
            elif 'mean' in f:
                bf = self.aux_attrs['mean_split'][f_name]
            brand_features.append(bf)
        brand_features = pd.concat(brand_features,axis=1)
        brand_features = dids.merge(brand_features,how='left',on='device_id').set_index('device_id').fillna(0).values
        features = np.concatenate([x,temp_features,temp_mms_features,brand_features],axis=1)
        return features
    
    def _assemble_features(self,directory,return_label,mode):
        dids = np.load(os.path.join(directory,'device_id.npy'))
        dids = pd.DataFrame(dids,columns=['device_id'])

        topic_features = []
        for feat in self.features:
            feat += '.npy'
            feat = np.load(os.path.join(directory,feat))
            topic_features.append(feat)
        topic_features = np.concatenate(topic_features,axis=1)
        features = self._generate_extra_features(topic_features,dids,mode)

        if  return_label:
            label = build_label(dids,self.train)
            return features,label
        return features
        
class NNEveProber(NNEveTrainer):
    def __init__(self,
             data_dict,
             n_folds,
             config_path,
             batch_size=128,
             lr=0.001,
             epochs=5):
        super().__init__(data_dict=data_dict,
                         n_folds=n_folds,
                         config_path=config_path,
                         batch_size=batch_size,
                         lr=lr,
                         epochs=epochs)
        
    def _assemble_features(self,directory,return_label,mode): 
        dids_eve = np.load(os.path.join(directory,'device_eve_id.npy'))
        dids_eve = pd.DataFrame(dids_eve,columns=['device_id'])
        dids_noeve = np.load(os.path.join(directory,'device_noeve_id.npy'))
        dids_noeve = pd.DataFrame(dids_noeve,columns=['device_id'])
        dids = pd.concat([dids_eve,dids_noeve])
        
        topic_features = []
        for feat in self.features:
            feat += '.npy'
            feat = np.load(os.path.join(directory,feat))
            topic_features.append(feat)
        topic_features = np.concatenate(topic_features,axis=1)
        missing_len = dids_noeve.shape[0]
        zeros_fill = np.zeros([missing_len,topic_features.shape[1]],dtype=np.float32)
        topic_features = np.concatenate([topic_features,zeros_fill])
        
        features = self._generate_extra_features(topic_features,dids,mode)
        if  return_label:
            label = build_label(dids,self.train)
            return features,label
        return features
        
        

    #%%

