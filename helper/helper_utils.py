import re
import numpy as np
import pandas as pd
from functools import partial
from nltk.stem import PorterStemmer
from collections.abc import Iterable
from sklearn.model_selection import StratifiedKFold

def map_label(ug,la,la_):
    for i,row1 in la.iterrows():
        g = row1['group']
        idx = np.where(ug==g)[0][0]
        c = ug[idx]
        la_.loc[i,c] = 1
    return la_

def join_string(ls):
    assert isinstance(ls,Iterable)
    inside_dtype = [type(s) is str for s in ls]
    assert np.all(inside_dtype),'expect all elements inside list to be type string'
    return ' '.join(ls)

def preprocess_category(row):
    row = row.replace('-',' ').replace('/',' ').replace('(','').replace(')','').replace(',','').lower()
    row = re.sub(r'\s+',' ',row)
    
    stemmer = PorterStemmer()
    strings = [stemmer.stem(c) if '\'' not in c else c for c in row.split() ]
    row = ' '.join(strings)
    return row

def mean_encoding(data,feat,target_name,target=None,outer_splits=20,innter_splits=10):
    data['target_mean'] = 0
    skf = StratifiedKFold(n_splits=outer_splits,shuffle=True,random_state=123)
    for ind_tr,ind_te in skf.split(data[feat],data[target_name]):
        inner_fold,inner_target = data.iloc[ind_tr],data.loc[ind_tr,target_name]
        inner_skf = StratifiedKFold(n_splits=innter_splits,shuffle=True,random_state=5) 
        inner_means = []
        default_inner_mean = inner_fold.groupby(feat)[target_name].mean()
        for inner_tr,inner_te in inner_skf.split(inner_fold,inner_target):
            inner_mean = data.iloc[inner_tr].groupby(feat)[target_name].mean()
            inner_means.append(inner_mean)
        
        inner_means = pd.concat(inner_means,axis=1,join='outer').mean(axis=1).fillna(default_inner_mean)
        data.loc[ind_te,'target_mean'] = data.loc[ind_te,feat].map(inner_means)
        output = data[[feat,'target_mean']]

    train_me_mean = output.groupby(feat)['target_mean'].mean()
    target_output = (target.loc[:,feat].map(train_me_mean)).to_frame(feat) if target is not None else None
    output.rename(columns={'target_mean':target_name},inplace=1)
    output.set_index(feat,inplace=True)
    target_output.rename(columns={feat:target_name},inplace=True)
    target_output[feat] = target[feat]
    target_output.set_index(feat,inplace=True)

    return output.astype(np.float32),target_output.astype(np.float32)

def assemble_mean_encodes(label,test,feat_name,target_cols):
    def assemble(encodes,data):
        encodes = pd.concat(encodes,axis=1)
        valid_mask = encodes.index.isin(cnt[feat_name])
        valid_index = data.loc[valid_mask,'device_id'].tolist()
        invalid_index = data.loc[~valid_mask,'device_id'].tolist()
        valid_encodes,invalid_encodes = encodes[valid_mask],encodes[~valid_mask]
        valid_encodes['device_id'] = valid_index;valid_encodes.set_index('device_id',inplace=True)
        invalid_encodes['device_id'] = invalid_index;invalid_encodes.set_index('device_id',inplace=True)
        fill_value = np.full(invalid_encodes.shape,fill_value=-1)
        invalid_encodes = pd.DataFrame(fill_value,columns=invalid_encodes.columns,index=invalid_encodes.index)
        encodes = pd.concat([valid_encodes,invalid_encodes])
        return encodes
    
    encodes_tr,encodes_te = [],[]
    cnt = label.groupby(feat_name).size()
    cnt = cnt[cnt>=50].reset_index()[[feat_name]]
    for col in target_cols:
        mean_encode_tr,mean_encode_te = mean_encoding(label,feat_name,col,test)
        encodes_tr.append(mean_encode_tr)
        encodes_te.append(mean_encode_te)
        
    encodes_tr = assemble(encodes_tr,label)
    encodes_te = assemble(encodes_te,test)
    encodes = pd.concat([encodes_tr,encodes_te])
    return encodes

def process_events(events):
    def split_time_zone(hour):
        if hour >= 0 and hour < 6:
            return 'midnight'
        elif hour >= 6 and hour <12:
            return 'morning'
        elif hour >= 12 and hour < 18:
            return 'afternoon'
        else:
            return 'evening'

    events['cnt'] = 1
    events['hour'] = events['timestamp'].dt.hour
    events['dow'] = events['timestamp'].dt.dayofweek
    events['time_zone'] = events['hour'].map(split_time_zone)
    return events

def produce_base_data(data_dict):
    label_cate = data_dict['label_categories'].copy()
    label_cate['category'] = label_cate['category'].fillna('').map(preprocess_category)
    app_labels = data_dict['app_labels'].copy()
    app_events = data_dict['app_events'].copy()
    active_app_events = app_events[app_events['is_active']==1]
    events = data_dict['events'].copy()
    events = process_events(events)
    events = events.merge(data_dict['phone_brand_device_model'],how='left',on='device_id')
        
    base_data = {'label_cate':label_cate,
                 'app_labels':app_labels,
                 'app_events':app_events,
                 'active_app_events':active_app_events,
                 'events':events,
                 'train':data_dict['gender_age_train'],
                 'test':data_dict['gender_age_test'],
                 'phone_brand':data_dict['phone_brand_device_model']}
    return base_data

def temporal_prefix(row,row_col,target_col):
    def add_prefix(char,prefix):
        if type(prefix) != str:
            prefix = str(prefix)
        if not prefix.endswith('_'):
            return prefix + '_' + char
        else:
            return prefix + char
        
    tp = row[row_col]
    content = row[target_col].split()
    if len(content) > 0:
        func = partial(add_prefix,prefix=tp)
        new_content = ' '.join(map(func,content))
        return new_content
    else:
        return ''

#%%
