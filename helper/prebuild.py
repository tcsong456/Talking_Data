
class TopicCategory:
    def __call__(self,data_dict):
        app_cate = data_dict['app_labels'].merge(data_dict['label_cate'],on='label_id')
        app_cate = app_cate.groupby('app_id')['category'].apply(list).map(join_string)
        events_cate = data_dict['app_events'].merge(app_cate,how='left',on='app_id')
        events_cate = events_cate.groupby('event_id')['category'].apply(list).map(join_string)
        device_cate = data_dict['events'].merge(events_cate,on='event_id')
        device_cate = device_cate.groupby('device_id')['category'].apply(list).map(join_string)
        return device_cate
        

#%%
from helper.utils import Timer
with Timer():
    tc = TopicCategory()
    z = tc(base_data)


            #%%
z
#%%
import pandas as pd
x = base_data['app_labels'].set_index('label_id')
with Timer():
    base_data['app_labels'].merge(z,on='label_id')
    pd.concat(,z)
#print('dsfsf')
