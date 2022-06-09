from sklearn.linear_model import LogisticRegression

class LR:
    def __init__(self,
                 penalty='l2',
                 C=10.0,
                 max_iter=100,
                 multi_class='auto'):
        clf = LogisticRegression(penalty=penalty,
                                 max_iter=max_iter,
                                 C=C,
                                 multi_class=multi_class)
        self.clf = clf
    
    def fit(self,X,y):
        self.clf.fit(X,y)
        return self
    
    def predict(self,X_te):
        preds = self.clf.predict_proba(X_te)
        return preds[:,1]


#%%
