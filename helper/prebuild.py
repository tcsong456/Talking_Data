from functools import partial
from helper.helper_utils import (join_string,
                                 temporal_prefix)

class TopicCategory:
    def __call__(self,data_dict):
        app_cate = data_dict['app_labels'].merge(data_dict['label_cate'],on='label_id')
        app_cate = app_cate.groupby('app_id')['category'].apply(list).map(join_string)
        events_cate = data_dict['app_events'].merge(app_cate,how='left',on='app_id')
        events_cate = events_cate.groupby('event_id')['category'].apply(list).map(join_string)
        device_cate = data_dict['events'].merge(events_cate,on='event_id')
        device_cate = device_cate.groupby('device_id')['category'].apply(list).map(join_string)
        return device_cate

class BaseTopicLabel:
    def _add_prefix(self,row_col,inp,target_col):
        map_func = partial(temporal_prefix,row_col=row_col,target_col=target_col)
        new_col = row_col + '_new'
        inp[new_col] = inp.apply(map_func,axis=1)
        inp = inp.groupby('device_id')[new_col].apply(list).map(join_string).to_frame().rename(columns={new_col:target_col})
        return inp
    
    def _device_feature(self,data_dict,app_labels,event_col):
        app_events = data_dict[event_col].merge(app_labels,how='left',on='app_id')
        events_labels = app_events.groupby('event_id')['label_id'].apply(list).map(join_string)
        device_labels = data_dict['events'].merge(events_labels,on='event_id')
        return device_labels
    
    def __call__(self,data_dict):
        app_labels = data_dict['app_labels'].copy()
        app_labels['label_id'] = app_labels['label_id'].map(str)
        app_labels = app_labels.groupby('app_id')['label_id'].apply(list).map(join_string)
        device_labels = self._device_feature(data_dict,app_labels,event_col='app_events')
        active_device_labels = self._device_feature(data_dict,app_labels,event_col='active_app_events')
        device = device_labels.groupby('device_id')['label_id'].size()
        self.all_device_ids = device.reset_index()[['device_id']]
        return device_labels,active_device_labels

class TopicLabel(BaseTopicLabel):
    def __call__(self,data_dict):
        device_labels,active_device_labels = super().__call__(data_dict)
        device_labels = device_labels.groupby('device_id')['label_id'].apply(list).map(join_string)
        device_active_labels = active_device_labels.groupby('device_id')['label_id'].apply(list).map(join_string)
        device_active_labels = self.all_device_ids.merge(device_active_labels,how='left',on='device_id').fillna('')
        device_active_labels = device_active_labels.set_index('device_id')['label_id']
        return device_labels,device_active_labels
        
class TopicTimeZoneLabel(BaseTopicLabel):
    def __call__(self,data_dict):
        device_labels,active_device_labels = super().__call__(data_dict)
        device_tz_labels = self._add_prefix('time_zone',device_labels,'label_id')
        device_acitve_tz_labels = self._add_prefix('time_zone',active_device_labels,'label_id')
        device_tz_labels = device_tz_labels.groupby('device_id')['label_id'].apply(list).map(join_string)
        device_acitve_tz_labels = device_acitve_tz_labels.groupby('device_id')['label_id'].apply(list).map(join_string)
        device_acitve_tz_labels = self.all_device_ids.merge(device_acitve_tz_labels,how='left',on='device_id').fillna('')
        device_acitve_tz_labels = device_acitve_tz_labels.set_index('device_id')['label_id']
        return device_tz_labels,device_acitve_tz_labels
    
class TopicPbrandLabel(BaseTopicLabel):
    def __call__(self,data_dict):
        device_labels,active_device_labels = super().__call__(data_dict)
        device_pbrand_labels = self._add_prefix('phone_brand',device_labels,'label_id')
        device_active_pbrand_labels = self._add_prefix('phone_brand',active_device_labels,'label_id')
        device_pbrand_labels = device_pbrand_labels.groupby('device_id')['label_id'].apply(list).map(join_string)
        device_active_pbrand_labels = device_active_pbrand_labels.groupby('device_id')['label_id'].apply(list).map(join_string)
        device_active_pbrand_labels = self.all_device_ids.merge(device_active_pbrand_labels,how='left',on='device_id').fillna('')
        device_active_pbrand_labels = device_active_pbrand_labels.set_index('device_id')['label_id']
        return device_pbrand_labels,device_active_pbrand_labels

class BaseTopicApp(BaseTopicLabel):
    def _device_feature(self,data_dict,event_col):
        app_events = data_dict[event_col].copy()
        app_events['app_id'] = app_events['app_id'].map(str)
        events_app = app_events.groupby('event_id')['app_id'].apply(list).map(join_string)
        device_app = data_dict['events'].merge(events_app,on='event_id')
        return device_app
        
    def __call__(self,data_dict):
        device_app = self._device_feature(data_dict,'app_events')
        active_device_app = self._device_feature(data_dict,'active_app_events')
        device = device_app.groupby('device_id')['app_id'].size()
        self.all_device_id = device.reset_index()[['device_id']]
        return device_app,active_device_app

class TopicApp(BaseTopicApp):
    def __call__(self,data_dict):
        device_app,active_device_app = super().__call__(data_dict)
        device_app = device_app.groupby('device_id')['app_id'].apply(list).map(join_string)
        device_active_app = active_device_app.groupby('device_id')['app_id'].apply(list).map(join_string)
        device_active_app = self.all_device_id.merge(device_active_app,how='left',on='device_id').fillna('')
        device_active_app = device_active_app.set_index('device_id')['app_id']
        return device_app,device_active_app

class TopicTimeZoneApp(BaseTopicApp):
    def __call__(self,data_dict):
        device_app,active_device_app = super().__call__(data_dict)
        device_tz_app = self._add_prefix('time_zone',device_app,'app_id')
        device_active_tz_app = self._add_prefix('time_zone',active_device_app,'app_id')
        device_tz_app = device_tz_app.groupby('device_id')['app_id'].apply(list).map(join_string)
        device_active_tz_app = device_active_tz_app.groupby('device_id')['app_id'].apply(list).map(join_string)
        device_active_tz_app = self.all_device_id.merge(device_active_tz_app,how='left',on='device_id').fillna('')
        device_active_tz_app = device_active_tz_app.set_index('device_id')['app_id']
        return device_tz_app,device_active_tz_app
    
class TopicPbrandApp(BaseTopicApp):
    def __call__(self,data_dict):
        device_app,active_device_app = super().__call__(data_dict)
        device_pbrand_app = self._add_prefix('phone_brand',device_app,'app_id')
        device_active_pbrand_app = self._add_prefix('phone_brand',active_device_app,'app_id')
        device_pbrand_app = device_pbrand_app.groupby('device_id')['app_id'].apply(list).map(join_string)
        device_active_pbrand_app = device_active_pbrand_app.groupby('device_id')['app_id'].apply(list).map(join_string)
        device_active_pbrand_app = self.all_device_id.merge(device_active_pbrand_app,how='left',on='device_id').fillna('')
        device_active_pbrand_app = device_active_pbrand_app.set_index('device_id')['app_id']
        return device_pbrand_app,device_active_pbrand_app

class BaseTopicCombineApp(BaseTopicLabel):
    def _device_feature(self,data_dict,apps,event_col):
        app_events = data_dict[event_col].copy()
        app_events = app_events.merge(apps,on='app_id').drop('app_id',axis=1).rename(columns={'hash_id':'app_id'})
        app_events['app_id'] = app_events['app_id'].map(str)
        event_app = app_events.groupby('event_id')['app_id'].apply(list).map(join_string)
        device_app = data_dict['events'].merge(event_app,on='event_id')
        return device_app
        
    def __call__(self,data_dict):
        app_labels = data_dict['app_labels'].copy()
        apps = app_labels.groupby('app_id')['label_id'].apply(list).map(sorted).map(lambda row:' '.join(list(map(str,row)))).map(hash).to_frame('hash_id')
        device_combine_app = self._device_feature(data_dict,apps,'app_events')
        active_device_combine_app = self._device_feature(data_dict,apps,'active_app_events')
        device = device_combine_app.groupby('device_id')['app_id'].size()
        self.all_device_id = device.reset_index()[['device_id']]
        return device_combine_app,active_device_combine_app

class TopicCombineApp(BaseTopicCombineApp):
    def __call__(self,data_dict):
        device_combine_app,active_device_combine_app = super().__call__(data_dict)
        device_combine_app = device_combine_app.groupby('device_id')['app_id'].apply(list).map(join_string)
        active_device_combine_app = active_device_combine_app.groupby('device_id')['app_id'].apply(list).map(join_string)
        active_device_combine_app = self.all_device_id.merge(active_device_combine_app,how='left',on='device_id').fillna('')
        active_device_combine_app = active_device_combine_app.set_index('device_id')['app_id']
        return device_combine_app,active_device_combine_app

class TopicTimeZoneCombineApp(BaseTopicCombineApp):
    def __call__(self,data_dict):
        device_combine_app,active_device_combine_app = super().__call__(data_dict)
        device_tz_combine_app = self._add_prefix('time_zone',device_combine_app,'app_id')
        device_active_tz_combine_app = self._add_prefix('time_zone',active_device_combine_app,'app_id')
        device_tz_combine_app = device_tz_combine_app.groupby('device_id')['app_id'].apply(list).map(join_string)
        device_active_tz_combine_app = device_active_tz_combine_app.groupby('device_id')['app_id'].apply(list).map(join_string)
        device_active_tz_combine_app = self.all_device_id.merge(device_active_tz_combine_app,how='left',on='device_id').fillna('')
        device_active_tz_combine_app = device_active_tz_combine_app.set_index('device_id')['app_id']
        return device_tz_combine_app,device_active_tz_combine_app

class TopicPbrandCombineApp(BaseTopicCombineApp):
    def __call__(self,data_dict):
        device_combine_app,active_device_combine_app = super().__call__(data_dict)
        device_pbrand_combine_app = self._add_prefix('phone_brand',device_combine_app,'app_id')
        device_active_pbrand_combine_app = self._add_prefix('phone_brand',active_device_combine_app,'app_id')
        device_pbrand_combine_app = device_pbrand_combine_app.groupby('device_id')['app_id'].apply(list).map(join_string)
        device_active_pbrand_combine_app = device_active_pbrand_combine_app.groupby('device_id')['app_id'].apply(list).map(join_string)
        device_active_pbrand_combine_app = self.all_device_id.merge(device_active_pbrand_combine_app,how='left',on='device_id').fillna('')
        device_active_pbrand_combine_app = device_active_pbrand_combine_app.set_index('device_id')['app_id']
        return device_pbrand_combine_app,device_active_pbrand_combine_app

#%%