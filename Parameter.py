from sko.DE import DE
import numpy as np
import math
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

print('svm:')
train_data=np.loadtxt("train.txt")
test_data=np.loadtxt("test.txt")

X_train = train_data[:,0:-1]
T_train = train_data[:,-1]
X_test  = test_data[:,0:-1]
T_test  = test_data[:,-1]
times=0
def obj_func(p):
    global times
    times=times+1
    if times%300==0:
        print(times)
    C,g,e = p
    global X_train,T_train,X_test,T_test
    KF = KFold(n_splits=10,shuffle=True,random_state=100)
    sv = SVR(kernel = 'rbf',C=C,gamma=g,epsilon=e)
    MSE=0
    for k,(train,test) in enumerate(KF.split(X_train,T_train)):
        x_train=X_train[train]
        y_train=T_train[train]
        x_test=X_train[test]    
        y_test=T_train[test]
        sv.fit(x_train,y_train)
        test_pre_T=sv.predict(x_test)
        #train_pre_T=sv.predict(X_train)
        MSE=MSE+np.mean((test_pre_T-y_test)**2)
    MSE=MSE/10
    return MSE

de = DE(func=obj_func, n_dim=3, size_pop=30, max_iter=400, lb=[0,0,0], ub=[100,100,10],)

best_x, best_y = de.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
np.savetxt("best_c_y_e.txt",best_x)
np.savetxt("best_mse.txt",best_y)