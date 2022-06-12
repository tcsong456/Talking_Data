import sys
sys.path.append('.')
import json
import argparse
import pandas as pd
from utils import (
                   Timer,
                   load_data)
from multiprocessing import Pool
from helper import produce_base_data
from trainer import (auto_generate_dataset_names,
                     EventStackSaver)

class TransformFeature:
    def __init__(self,
                 stacker,
                 config_path):
        with open(f'configs/{config_path}.json') as f:
            config = json.load(f)
        self.config = config
        self.stacker = stacker
    
    def build(self,ds):
        data = pd.read_pickle(f'temp_data/{ds}.pkl')
        cur_config = self.config[ds]
        with Timer(message=f'starting building feature using dataset:{ds}'):
            for v in cur_config.values():
                self.stacker.run(data,**v)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_argument = parser.add_argument
    add_argument('--n_folds',type=int,default=5)
    add_argument('--random_state',type=int,default=7951)
    add_argument('--num_of_cores',type=int,default=5)
    args = parser.parse_args()
    
    base_data = produce_base_data(load_data('data'))
    file_names = auto_generate_dataset_names('temp_data')
    stacker = EventStackSaver(n_folds=args.n_folds,
                              data_dict=base_data,
                              random_state=args.random_state)
    tf = TransformFeature(stacker=stacker,
                          config_path='vect_config')
    
    with Pool(args.num_of_cores) as pool:
        pool.map(tf.build,file_names)
        pool.close()
        pool.join()
                
#%%
'''
cate:441
app:8653
app_active:4052
app_combine:2858
app_combine_active:1723
label:467
label_active:438
app_brand:18837
app_brand_active:8381
app_combine:10034
app_combine_active:5317
label_brand:9550
label_brand_active:6881
app_tz:17904
app_tz_active:7771
app_combine_tz:6317
app_combine_tz_active:3643
label_tz:1837
label_tz_active:1689
'''