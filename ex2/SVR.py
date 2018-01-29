import svmutil as st
import numpy as np

'''this script include problem13 and problem14 about SVR for classification'''

def data_transform(X):
    '''X's type is np.array, we need to transform it to the form 
    of list of dictionary which is suited to libsvm pakage'''
    N,d=X.shape
    return [dict(zip(range(1,d+1),X[i])) for i in range(N)]

def label_transform(y):
    '''y is form of np.array, we need to transform it to list type'''
    return list(y)

def grid(X_train,y_train,X_test,y_test,gamma,C):
    ret=list()
    X_train_T=data_transform(X_train)
    print(X_train_T)
    y_train_T=label_transform(y_train)
    X_test_T=data_transform(X_test)
    y_test_T=label_transform(y_test)
    print(y_train_T)
    for g in gamma:
        for c in C:
            model=st.svm_train(y_train_T,X_train_T,'-s 3 -t 2 -g %s -c %s -p 0.5'%(g,c))
            train_predict,_,_=st.svm_predict(y_train_T,X_train_T,model)
            train_error=np.mean(np.sign(train_predict)!=y_train)
            test_predict,_,_=st.svm_predict(y_test_T,X_test_T,model)
            test_error=np.mean(np.sign(test_predict)!=y_test)
            ret.append((train_error,test_error,g,c))
    return ret

def main():
    data = np.loadtxt('./hw2_lssvm_all.dat')
    train = data[:400]
    test = data[400:]
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]
    gamma = [32, 2, 0.125]
    C = [0.001, 1, 1000]
    all_res=grid(X_train,y_train,X_test,y_test,gamma,C)
    for res in all_res:
        print(res)

    '''the results are as follow:
(0.40000000000000002, 0.47999999999999998, 32, 0.001)
(0.0, 0.47999999999999998, 32, 1)
(0.0, 0.47999999999999998, 32, 1000)
(0.40000000000000002, 0.47999999999999998, 2, 0.001)
(0.0, 0.47999999999999998, 2, 1)
(0.0, 0.47999999999999998, 2, 1000)
(0.40000000000000002, 0.47999999999999998, 0.125, 0.001)
(0.035000000000000003, 0.41999999999999998, 0.125, 1)
(0.0, 0.46999999999999997, 0.125, 1000)'''

if __name__ == '__main__':
    main()