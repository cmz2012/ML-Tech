import numpy as np
import svmutil
import pandas as pd

def validate(X_train,y_train):
    N=len(y_train)
    samples=np.zeros(N,dtype=bool)
    samples[np.random.permutation(np.arange(N))[:1000]]=True
    y_split_train=list(np.array(y_train)[samples])
    X_split_train=list(np.array(X_train)[samples])
    y_split_test=list(np.array(y_train)[~samples])
    X_split_test=list(np.array(X_train)[~samples])
    c=0.1
    ret=list()
    for t in range(-1,4):
        gamma=10**t
        model=svmutil.svm_train(y_split_train,X_split_train,'-g %s -c %s'%(gamma,c))
        _,(accuracy,_,_),_=svmutil.svm_predict(y_split_test,X_split_test,model)
        ret.append((accuracy,t))
    return max(ret)[1]

def main():
    y_train, X_train = svmutil.svm_read_problem('../features-libsvm.train')
    # y_test, X_test = svmutil.svm_read_problem('../features-libsvm.test')
    y_train = list(2 * (np.array(y_train) == 0) - 1)
    # y_test = list(2 * (np.array(y_test) == 0) - 1)
    R=1000
    res=list()
    for r in range(R):
        print("round %s"%r)
        res.append(validate(X_train,y_train))
    pd.Series(res).hist()

if __name__=='__main__':
    main()