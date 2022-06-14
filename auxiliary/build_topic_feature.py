import os
import sys
sys.path.append('.')
import argparse
from functools import partial
from multiprocessing  import Pool
from utils import (logger,
                   load_data,
                   Timer)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_argument = parser.add_argument
    add_argument('--data_path',type=str,default='data',help='the path to load real data')
    add_argument('--temp_data_path',type=str,default='temp_data',help='path to save topic feature data')
    add_argument('--use_multiprocess',action='store_true')
    add_argument('--num_of_cores',type=int,default=5,help='number of cores to use for multiprocessing')
    args = parser.parse_args()
    
    os.makedirs(args.temp_data_path,exist_ok=True)
    data_dict = load_data(args.data_path)
    base_data = produce_base_data(data_dict)
    del data_dict

    topic_feature_list = [TopicLabel,TopicTimeZoneLabel,TopicPbrandLabel,TopicApp,
                          TopicTimeZoneApp,TopicPbrandApp,TopicCombineApp,TopicTimeZoneCombineApp,TopicPbrandCombineApp]
    if args.use_multiprocess:
        save_topic_func = partial(save_topic_data,data=base_data,args=args,cate_mode=False)
        with Timer(message='Starting building topic features'):
            with Pool(args.num_of_cores) as pool:
                pool.map(save_topic_func,topic_feature_list)
                pool.close()
                pool.join()
    else:
        for tfl in topic_feature_list:
            save_topic_data(tfl,base_data,args,cate_mode=False)
    save_topic_data(TopicCategory,base_data,args,cate_mode=True)
        
        