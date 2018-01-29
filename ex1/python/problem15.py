import numpy as np
import pandas as pd
import svmutil
import matplotlib.pyplot as plt


def main():
    y_train,X_train=svmutil.svm_read_problem('../features-libsvm.train')
    y_test,X_test=svmutil.svm_read_problem('../features-libsvm.test')
    c=0.1
    y_train=list(2*(np.array(y_train)==0)-1)
    y_test=list(2*(np.array(y_test)==0)-1)
    eout=list()
    for t in range(-3,2):
        gamma=10**t
        model=svmutil.svm_train(y_train,X_train,'-g %s -c %s'%(gamma,c))
        _,(accuracy,_,_),_=svmutil.svm_predict(y_test,X_test,model)
        eout.append(100-accuracy)
    plt.plot(range(-3,2),eout)
    plt.xlabel("log10(gamma)")
    plt.ylabel("Eout on test set")
    plt.title("C=0.1")
    plt.show()

if __name__=="__main__":
    main()