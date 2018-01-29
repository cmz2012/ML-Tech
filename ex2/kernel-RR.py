import numpy as np

'''including problem11 and problem12 about Kernel Ridge Regression in exercise2'''

def grid(X_train,y_train,X_test,y_test,gamma,lamb):
    '''gamma and lamb both are a parameter list in the Gaussian-RBF kenel model,
    we compute all the possible combination in the kernel ridge regression model'''
    N,_=X_train.shape
    ret=list()
    n,_=X_test.shape
    for g in gamma:
        for l in lamb:
            K = np.zeros((N, N))
            for i in range(N):
                xi=X_train[i]
                for j in range(i,N):
                    sub=xi-X_train[j]
                    val=np.exp(-g*np.sum(sub*sub))
                    K[i][j]=val
                    K[j][i]=val
            beta=(np.linalg.inv(K+l*np.eye(N,N))*y_train).sum(axis=1)
            ## compute Ein error
            train_predict=np.sign((K*beta).sum(axis=1))
            train_err=np.mean(train_predict!=y_train)

            ## compute Eout error
            T=np.zeros((n,N))
            for i in range(n):
                xi=X_test[i]
                for j in range(N):
                    sub=xi-X_train[j]
                    val=np.exp(-g*np.sum(sub*sub))
                    T[i][j]=val
            test_predict=np.sign((T*beta).sum(axis=1))
            test_error=np.mean(test_predict!=y_test)

            ret.append((train_err,test_error,g,l))
    return ret

def main():
    data=np.loadtxt('./hw2_lssvm_all.dat')
    train=data[:400]
    test=data[400:]
    X_train=train[:,:-1]
    y_train=train[:,-1]
    X_test=test[:,:-1]
    y_test=test[:,-1]
    gamma=[32,2,0.125]
    lamb=[0.001,1,1000]
    all_res=grid(X_train,y_train,X_test,y_test,gamma,lamb)
    for res in all_res:
        print(res)

    '''the results are as follow:
(0.0, 0.45000000000000001, 32, 0.001)
(0.0, 0.45000000000000001, 32, 1)
(0.0, 0.45000000000000001, 32, 1000)
(0.0, 0.44, 2, 0.001)
(0.0, 0.44, 2, 1)
(0.0, 0.44, 2, 1000)
(0.0, 0.46000000000000002, 0.125, 0.001)
(0.029999999999999999, 0.45000000000000001, 0.125, 1)
(0.24249999999999999, 0.39000000000000001, 0.125, 1000)
    '''

if __name__=="__main__":
    main()
