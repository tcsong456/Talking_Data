import os
import pandas as pd
from glob import glob

def generate_device_id(data_dict):
    event_ids = data_dict['app_events'].groupby('event_id').size().to_frame()
    events_app = data_dict['events'].merge(event_ids,on='event_id')
    dids = events_app.groupby('device_id').size().reset_index()[['device_id']]
    return dids

def auto_detect_col(inp):
    assert type(inp) is pd.DataFrame
    columns = inp.columns.tolist()
    assert len(columns)==2,'input data can only have 2 columns'
    columns.remove('device_id')
    return columns[0]

def auto_generate_dataset_names(path):
    files = []
    for file in glob(f'{path}/*'):
       file_name = os.path.basename(file)
       file_name = file_name[:-4]
       files.append(file_name)
    return files