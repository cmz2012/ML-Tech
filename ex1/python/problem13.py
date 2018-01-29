'''transfrom the text train data and test data to the libsvm standart format
each row in the usual data form is <label> <feature1> <feature2> ...
each row in the libsvm format is <label> <1:feature1> <2:feature2> ...'''
import numpy as np
import pandas as pd
import svmutil

def transform(filename,newname):
    data=np.loadtxt(filename)
    with open(newname,'w') as nf:
        n,d=data.shape
        S=''
        for i in range(n):
            y=data[i,0]
            s=list()
            s.append(str(y))
            for j in range(1,d):
                s.append(str(j)+':'+str(data[i,j]))
            S+=' '.join(s)+'\n'
        nf.write(S)

def Problem11(X_train,y_train):
    '''binary classification 0 versus not 0'
    y_train and y_test are list of int label
    X_train and X_test are list of dict, which is the form of {featureTH:feature}:'''
    label_train=list(2*(np.array(y_train)==0)-1)
    #label_test=list(2*(np.array(y_test)==0)-1)
    w_len=list()
    for t in range(-5,5,2):
        c=10**t
        model=svmutil.svm_train(label_train,X_train,'-t 0 -c '+str(c))
        SV=model.get_SV()
        cof=model.get_sv_coef()
        '''SV is a list of dict {1:feature1,2:feature2,...}
        cof is a list of (alpha*y)'''
        SV=np.array(pd.DataFrame(SV).ix[:,1:])
        cof=np.array(cof)
        w=np.dot(SV.T,cof)
        w_len.append(np.sqrt(np.sum(w*w)))
    return w_len

def Problem12(X_train,y_train):
    '''the input is same as the problem 11
    but in this problem , we need set the digit 8 as positive and others are negative
    and evaluate the ein as the param c changes'''
    label_train = list(2 * (np.array(y_train) == 8) - 1)
    #ein=list()
    nSV=list()
    Q=2
    coef0=1
    for t in range(-5,5,2):
        c=10**t
        model=svmutil.svm_train(label_train,X_train,'-t 1 -g 1 -c %s -d %s -r %s'%(c,Q,coef0))
        predict_label,_,_=svmutil.svm_predict(label_train,X_train,model)
        error=np.mean(np.array(predict_label)!=np.array(label_train))
        print(error)
        nSV.append(model.get_nr_sv())
    return nSV


def problem14(X_train,y_train):
    label=list(2*(np.array(y_train)==0)-1)
    train=np.array(pd.DataFrame(X_train))
    for t in range(-3,2):
        c=10**t
        model=svmutil.svm_train(label,X_train,'-g 80 -c %s'%c)
        sv_indices=model.get_sv_indices()

def main():
    #transform('features.train','features-libsvm.train')
    #transform('features.test','features-libsvm.test')
    y_train,X_train=svmutil.svm_read_problem('../features-libsvm.train')
    #w_len=Problem11(X_train,y_train)
    nSV=Problem12(X_train,y_train)
    np.savez('../sSV.npz',sSV=np.array(nSV))

if __name__=="__main__":
    main()