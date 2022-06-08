import os
import argparse
import warnings
warnings.filterwarnings(action='ignore')
from helper import (TopicCategory,
                    TopicLabel,
                    TopicTimeZoneLabel,
                    TopicPbrandLabel,
                    TopicApp,
                    TopicTimeZoneApp,
                    TopicPbrandApp,
                    TopicCombineApp,
                    TopicTimeZoneCombineApp,
                    TopicPbrandCombineApp,
                    produce_base_data)
from utils import (logger,
                   load_data,
                   Timer)
from functools import partial
from multiprocessing  import Pool
from trainer import NoEventStackSaver

def save_topic_data(topic_func,data,args,cate_mode):
    topic_func = topic_func()
    class_name = topic_func.__class__.__name__.lower()
    save_path = os.path.join(args.temp_data_path,class_name)
    logger.info(f'creating dataset:{class_name}')
    
    if cate_mode:
        data = topic_func(data)
    else:
        data,active_data = topic_func(data)
        active_save_path = save_path + '_active.pkl'
        active_data.to_pickle(active_save_path)
    
    save_path += '.pkl'
    data.to_pickle(save_path)

def main():
    parser = argparse.ArgumentParser()
    add_argument = parser.add_argument
    add_argument('--temp_data_path',type=str,default='temp_data',help='path to save topic feature data')
    add_argument('--data_path',type=str,default='data',help='the path to load real data')
    add_argument('--num_of_cores',type=int,default=5,help='number of cores to use for multiprocessing')
    add_argument('--mode',type=str,choices=['no_eve','eve'])
    add_argument('--n_folds',type=int,default=5)
    add_argument('--random_state',type=int,default=7951)
    args = parser.parse_args()
    
    os.makedirs(args.temp_data_path,exist_ok=True)
    data_dict = load_data(args.data_path)
    base_data = produce_base_data(data_dict)
    del data_dict
    
#    topic_feature_list = [TopicLabel,TopicTimeZoneLabel,TopicPbrandLabel,TopicApp,
#                          TopicTimeZoneApp,TopicPbrandApp,TopicCombineApp,TopicTimeZoneCombineApp,TopicPbrandCombineApp]
#    save_topic_func = partial(save_topic_data,data=base_data,args=args,cate_mode=False)
#    with Pool(args.num_of_cores) as pool:
#        pool.map(save_topic_func,topic_feature_list)
#        pool.close()
#        pool.join()
#    save_topic_data(TopicCategory,base_data,args,cate_mode=True)
    
    no_eve_stacker = NoEventStackSaver(n_folds=args.n_folds,
                                       data_dict=base_data,
                                       random_state=args.random_state)
    with Timer(message='start building no_eve features'):
        for split_func in ['le_split','oh_split','mean_split','freq_split']:
            no_eve_stacker.save(split_func)
    
if __name__ == '__main__':
    main()


#%%
#base_data = produce_base_data(load_data('data'))
#z = np.load('inp/no_events/eval/4/device_model_freq_split.npy')

