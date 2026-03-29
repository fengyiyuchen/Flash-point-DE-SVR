import numpy as np
from sklearn import linear_model
import math
from sklearn.svm import SVR


print('svm:')
train_data=np.loadtxt("train.txt")
test_data=np.loadtxt("test.txt")

X_train = train_data[:,0:-1]
T_train = train_data[:,-1]
X_test  = test_data[:,0:-1]
T_test  = test_data[:,-1]

sv = SVR(kernel = 'rbf',C=35.59272123,gamma=3.09286766,epsilon=0.00673876153)
sv.fit(X_train,T_train)
test_pre_T=sv.predict(X_test)
train_pre_T=sv.predict(X_train)

Max_min=np.loadtxt("max_min.txt")#Range of attribute column values
Min=np.loadtxt("min.txt")#Minimum value of attribute column value

test_pre_T_=test_pre_T*Max_min[-1]+Min[-1]
train_pre_T_=train_pre_T*Max_min[-1]+Min[-1]
T_train_=T_train*Max_min[-1]+Min[-1]
T_test_=T_test*Max_min[-1]+Min[-1]

print('------train:')
score1=sv.score(X_train,T_train)
print('R^2：%.4f'%score1)
MSE1=np.mean((train_pre_T_-T_train_)**2)
RMSE1=math.sqrt(MSE1)
print('RMSE:%.4f'%RMSE1)
MAE1=np.mean(np.abs(train_pre_T_-T_train_))
print('MAE:%.4f'%MAE1)
MAPE1=np.mean(np.abs((train_pre_T_-T_train_)/T_train_))*100
print('MAPE:%.4f'%MAPE1)
SE1=math.sqrt(np.sum((train_pre_T_-T_train_)**2)/(T_train.size-1))
print('SE:%.4f'%SE1)

print('------test:')
score2=sv.score(X_test,T_test)
print('R^2：%.4f'%score2)
MSE2=np.mean((test_pre_T_-T_test_)**2)
RMSE2=math.sqrt(MSE2)
print('RMSE:%.4f'%RMSE2)
MAE2=np.mean(np.abs(test_pre_T_-T_test_))
print('MAE:%.4f'%MAE2)
MAPE2=np.mean(np.abs((test_pre_T_-T_test_)/T_test_))*100
print('MAPE:%.4f'%MAPE2)
SE2=math.sqrt(np.sum((test_pre_T_-T_test_)**2)/(T_test.size-1))
print('SE:%.4f'%SE2)
ytr_=np.mean(T_train_)*np.ones((T_test_.size,1))
Q2ext=1-MSE2/np.mean((test_pre_T_-ytr_)**2)
print('Q2ext:%.4f'%Q2ext)

np.savetxt("train_pre.txt",train_pre_T_)
np.savetxt("test_pre.txt",test_pre_T_)