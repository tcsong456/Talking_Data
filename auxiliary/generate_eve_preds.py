import argparse
import warnings
import sys
sys.path.append('.')
warnings.filterwarnings(action='ignore')
from helper import produce_base_data
from utils import load_data
from trainer import NNEveTrainer,NNEveProber

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_argument = parser.add_argument
    add_argument('--n_folds',type=int,default=5)
    add_argument('--random_state',type=int,default=7951)
    add_argument('--batch_size',type=int,default=128)
    add_argument('--lr',type=float,default=0.001)
    add_argument('--epochs',default=5,type=int)
    add_argument('--mode',type=str,choices=['eve_submit','eve_probe'])
    args = parser.parse_args()
    
    data_dict = load_data('data')
    base_data = produce_base_data(data_dict)
    del data_dict
    
    base_params = {'n_folds':args.n_folds,
                   'data_dict':base_data}
    nn_params = {'batch_size':args.batch_size,
                 'epochs':args.epochs,
                 'lr':args.lr}
    for i in range(1,11):
        config_path = f'nn_eve_{i}'
        if args.mode == 'eve_submit':
            nn_trainer = NNEveTrainer(**base_params,
                                       **nn_params,
                                       config_path=config_path)
        elif args.mode == 'eve_probe':
            nn_trainer = NNEveProber(**base_params,
                                       **nn_params,
                                       config_path=config_path)
        nn_trainer.submit(config_path)


#%%
