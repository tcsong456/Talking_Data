import lightgbm as lgb

class LightGBM:
    def __init__(self,
                 params,
                 early_stopping_rounds=50,
                 num_boost_round=250):
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds=early_stopping_rounds
    
    def fit_with_validation(self,X_tr,X_val,y_tr):
        d_train = lgb.Dataset(X_tr,label=y_tr)
        d_valid = lgb.Dataset(X_val)
        eval_results = {}
        bst = lgb.train(self.params,
                        d_train,
                        valid_sets=[d_train,d_valid],
                        valid_names=['train','valid'],
                        eval_results=eval_results,
                        num_boost_round=self.num_boost_round,
                        early_stopping_rounds=self.early_stopping_rounds)
        self.bst = bst
        return self
    
    def fit_wo_validation(self,X_tr,y_tr):
        d_train = lgb.Dataset(X_tr,label=y_tr)
        bst = lgb.train(self.params,
                        d_train,
                        num_boost_round=self.num_boost_round)
        self.bst = bst
        return self
    
    def predict(self,X_test):
        preds = self.bst.predict(X_test,num_iteration=self.num_boost_round or self.bst.best_iteration)
        return preds
    