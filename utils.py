import os
import logging
import logzero
import numpy as np
import pandas as pd
from glob import glob
from time import time

dtype_appevents_dict = {'event_id':np.int32,
                        'is_installed':np.int8,
                        'is_active':np.int8}
dtype_applabels_dict = {'label_id':np.int16}
dtype_events_dict = {'event_id':np.int32,
                     'longitude':np.float16,
                     'latitude':np.float16}
dtype_train_dict = {'age':np.int8}
dtype_labelcate_dict = {'label_id':np.int16}

def custome_logger(name):
    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logger = logzero.setup_logger(formatter=formatter,
                                  level=logging.INFO,
                                  name=name)
    return logger
logger = custome_logger('talking_data')

def load_data(path):
    data_dict = {}
    data_map = {'app_events':dtype_appevents_dict,
                'app_labels':dtype_applabels_dict,
                'events':dtype_events_dict,
                'train':dtype_train_dict,
                'label_categories':dtype_labelcate_dict}
    
    load_path = os.path.join(path,'*')
    for file in glob(load_path):
        file_name = os.path.basename(file)
        file_name = file_name[:-4]
        if file_name in data_map:
            load_dtypes = data_map[file_name]
        else:
            load_dtypes = None
        if file_name == 'events':
            parse_dates = ['timestamp']
        else:
            parse_dates = False
        data_dict[file_name] = pd.read_csv(file,dtype=load_dtypes,parse_dates=parse_dates)
    data_dict['phone_brand_device_model'] = data_dict['phone_brand_device_model'].drop_duplicates('device_id')
    return data_dict

class Timer:
    def __init__(self,
                 logger=logger,
                 message='Start counting!'):
        self.message = message
        self.logger = logger
    
    def __enter__(self):
        self.logger.info(self.message)
        self.start_time = time()
        return self
    
    def __exit__(self,*args,**kwargs):
        end_time = time()
        interval = end_time - self.start_time
        self.logger.info(f'took {interval:.5f} seconds to complete')
        #%%