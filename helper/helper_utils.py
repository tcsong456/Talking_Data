import re
import numpy as np
from helper.utils import load_data
from nltk.stem import PorterStemmer
from collections.abc import Iterable

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

def produce_base_data(path):
    data_dict = load_data(path)
    
    label_cate = data_dict['label_categories'].copy()
    label_cate['category'] = label_cate['category'].fillna('').map(preprocess_category)
    app_labels = data_dict['app_labels'].copy()
    app_events = data_dict['app_events'].copy()
    active_app_events = app_events[app_events['is_active']==1]
    events = data_dict['events'].copy()
    events = process_events(events)
        
    base_data = {'label_cate':label_cate,
                 'app_labels':app_labels,
                 'app_events':app_events,
                 'active_app_events':active_app_events,
                 'events':events}
    return base_data

#%%