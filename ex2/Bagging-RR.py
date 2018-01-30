import svmutil as st
import numpy as np

'''this script includes problem 15 and problem 16 which are about to Bagging techniques'''

def KRR_classifier(X_train,y_train,lamb,g,bag_num):
    '''bagging bag_num data from X_train and use lamb for KRR classification'''
    N,d=X_train.shape
    samples=np.random.choice(np.arange(N),size=bag_num,replace=True)
    X=X_train[samples]
    y=y_train[samples]
    K=np.zeros((bag_num,bag_num))
    for i in range(bag_num):
        xi=X[i]
        for j in range(i,bag_num):
            sub=xi-X[j]
            val=np.exp(-g*np.sum(sub*sub))
            K[i][j]=val
            K[j][i]=val
    beta = (np.linalg.inv(K + lamb * np.eye(bag_num, bag_num)) * y).sum(axis=1)
    return beta,samples

def aggre_classifier(X_train,y_train,lamb_list,g,bag_num):
    clf=list()
    for lamb in lamb_list:
        clf.append(KRR_classifier(X_train,y_train,lamb,g,bag_num))
    return clf

def predict(clf,X_train,X_test,y_test,g):
    N,_=X_test.shape
    y_predict = list()
    for i in range(N):
        Xi = X_test[i]
        vote = list()
        for beta, samples in clf:
            k = list()
            for s in samples:
                sub = Xi - X_train[s]
                k.append(np.exp(-g * np.sum(sub * sub)))
            vote.append(np.sign(np.sum(beta * np.array(k))))
        y_predict.append(1 if sum(vote) > 0 else -1)
    return np.mean(np.array(y_predict)!=y_test)

def main():
    data = np.loadtxt('./hw2_lssvm_all.dat')
    data=np.hstack((np.ones((len(data),1)),data))
    train = data[:400]
    test = data[400:]
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]
    N,_=X_test.shape
    lamb_list=[0.01,0.1,1,10,100]
    g=2.
    clf=aggre_classifier(X_train,y_train,lamb_list,g,200)
    print(predict(clf,X_train,X_train,y_train,g))
    print(predict(clf,X_train,X_test,y_test,g))

if __name__ == '__main__':
    main()