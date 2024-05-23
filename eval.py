import numpy as np
import functools

from sklearn.metrics import f1_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder

def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool_)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()

def get_pred(embeddings, y):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool_)

    X = normalize(X, norm='l2')
    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=1 - 0.1,shuffle=False)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    return y_pred

def label_classification(embeddings, y, ratio):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool_)

    X = normalize(X, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=1 - ratio)

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)
    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    return micro,macro

import torch
def get_dis_with_center(x,y,z1,z2):
    min_y=torch.min(y)
    y=y-min_y
    max_y=torch.max(y)
    mu=[]
    origin_x=x.clone()
    origin_y=y.clone()
    x=torch.cat([x,z1,z2],dim=0)
    y=torch.cat([y,y,y],dim=0)
    y = y.detach().cpu().numpy()
    x=torch.nn.functional.normalize(x)
    norms=[]
    for label in range(max_y+1):
        indice=np.where(y==label)[0]
        temp=x[indice,:]
        mu.append(torch.mean(temp,dim=0))
    x=origin_x
    y=origin_y
    Y=y.clone()
    y = y.detach().cpu().numpy()
    x=torch.nn.functional.normalize(x)
    W=torch.stack(mu)
    y_pred=torch.mm(x,W.T).detach().cpu().numpy()
    y_pred = prob_to_one_hot(y_pred)
    Y = Y.reshape(-1, 1)
    Y=Y.detach().cpu()
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool_)
    print(y_pred.shape,Y.shape)
    micro = f1_score(Y, y_pred, average="micro")

    for label in range(max_y+1):
        indice=np.where(y==label)[0]
        temp=x[indice,:]
        temp1=temp[torch.randperm(temp.size(0))]
        temp=temp-temp1
        norm=torch.norm(temp,p=2,dim=1)
        norm=torch.mean(norm).item()
        norms.append(norm)
        
    delta=np.mean(norms) 
    mu=torch.stack(mu)
    mu_y=mu[y]
    temp=torch.mul(x,mu_y)
    sim_y=torch.sum(temp,dim=1)
    norm1=torch.norm(x,dim=1)
    norm2=torch.norm(mu_y,dim=1)
    sim_y=sim_y/norm1/norm2
    
    mu_i=torch.stack([torch.sum(mu,dim=0)]*(max_y+1))

    mu_i=(mu_i-mu)/(max_y)
    mu_i=mu_i[y]
    temp=torch.mul(x,mu_i)
    sim_i=torch.sum(temp,dim=1)
    norm1=torch.norm(x,dim=1)
    norm2=torch.norm(mu_i,dim=1)
    sim_i=sim_i/norm1/norm2
    
    sim_y,var_y=torch.mean(sim_y).item(),torch.var(sim_y).item()
    sim_i,var_i=torch.mean(sim_i).item(),torch.var(sim_i).item()
    print(sim_y,sim_i,var_y,var_i)
    return sim_y,sim_i,delta,micro

def get_distance_with_pos_neg(z1,z2):
    norm=torch.norm(z1-z2,dim=1)
    pos_norm=torch.mean(norm).item()
    neg_norm=[]
    for _ in range(10):
        tempz=z2[torch.randperm(z2.size(0))]
        norm=torch.norm(z1-tempz,dim=1)
        neg_norm.append(norm)
    neg_norm=torch.mean(neg_norm).item()
    print(pos_norm,neg_norm)
        
    
    