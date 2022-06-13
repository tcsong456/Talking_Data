import argparse
import warnings
import sys
sys.path.append('.')
warnings.filterwarnings(action='ignore')
from helper import produce_base_data
from utils import load_data,Timer
from trainer.stack_noeve_saver import (NoEventStackSaver,
                                       WholeNoEventStackSaver)
from trainer import (NNTrainer,
                     LRTrainer,
                     LgbTrainer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_argument = parser.add_argument
    add_argument('--n_folds',type=int,default=5)
    add_argument('--random_state',type=int,default=7951)
    add_argument('--mode',type=str,choices=['no_eve_submit','eve_submit'])
    add_argument('--batch_size',type=int,default=128)
    add_argument('--lr',type=float,default=0.001)
    add_argument('--epochs',default=5,type=int)
    args = parser.parse_args()
    
    data_dict = load_data('data')
    base_data = produce_base_data(data_dict)
    del data_dict
    
    base_params = {'n_folds':args.n_folds,
                   'data_dict':base_data}
    nn_params = {'batch_size':args.batch_size,
                 'epochs':args.epochs,
                 'lr':args.lr}
    if args.mode == 'eve_submit':
        no_eve_stacker = NoEventStackSaver(**base_params,
                                           random_state=args.random_state)
    elif args.mode == 'no_eve_submit':
        no_eve_stacker = WholeNoEventStackSaver(**base_params,
                                                random_state=args.random_state)
        
    with Timer(message='start building no_eve features'):
        for split_func in ['le_split','oh_split','mean_split','freq_split']:
            no_eve_stacker.save(split_func)
    
    for config_path in ['nn_noeve_1','nn_noeve_2','nn_noeve_3','nn_noeve_4','nn_noeve_5',
                        'nn_noeve_6','nn_noeve_7','nn_noeve_11']:
        nn_trainer = NNTrainer(**base_params,
                               **nn_params,
                               config_path=config_path,)
        nn_trainer.submit(config_path)
    
    for config_path in ['lr_noeve_1','lr_noeve_2','lr_noeve_3','lr_noeve_4']:
        lr_trainer = LRTrainer(**base_params,
                               config_path=config_path)
        lr_trainer.submit(config_path)
    
    for config_path in ['lgb_noeve_5','lgb_noeve_9','lgb_noeve_10' ]:
        lgb_trainer = LgbTrainer(**base_params,
                                 config_path=config_path)
        lgb_trainer.submit(config_path)

#%%