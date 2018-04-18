import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

pddata = pd.read_csv('LogiReg_data.txt',header = None,names=['Exam1','Exam2','Admitted'])
pddata.head()
print(pddata.shape)

pos = pddata[pddata['Admitted'] == 1]
neg = pddata[pddata['Admitted'] == 0]

plt.scatter(pos['Exam1'],pos['Exam2'],label='Admitted'); plt.scatter(neg['Exam1'],pos['Exam2'],label='Not Admitted'); plt.legend();plt.xlabel('Exam1 Score');plt.ylabel('Exam2 Score')


# 构建LR，求解参数1和参数2

# 分六部构建：
# 1.sigmoid函数
# 2. model函数，返回结果值
# 3. cost函数， 根据参数计算损失
# 4. gradient函数，计算每个参数的梯度方向
# 5. descent函数： 进行参数更新
# 6. accuracy函数，计算准确率


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def model(X,theta):
    return sigmoid(np.dot(X,theta.T))


pddata.insert(0,'ones',1)
orig_data = np.array(pddata)
cols = orig_data.shape[1]
x = orig_data[:,:3]
y = orig_data[:,0]
theta = np.zeros([1,3])


def cost(X,y,theta):
    return np.sum(-y * np.log(model(X,theta)) - (1 - y) * np.log(1 - model(X,theta))) / len(X)

print(cost(X,y,theta))


def gradient(X,y,theta):
    grad = np.zeros_like(theta)
    error = (model(X,theta) - y).ravel()
    for i in range(len(theta.ravel())):
        term = np.dot(error,X[:,i])
        grad[0,i] = term / len(X)
    return grad


##比较3钟不同的梯度下降方法

stop_iter = 0
stop_cost = 1
stop_grad = 2

def stopcriterion(type,value,threshold):
    # 设定三种不同的停止策略
    if type == stop_iter:
        return value > threshold
    elif type == stop_cost:
        return abs(value[-1] - value[-2]) < threshold
    elif type == stop_grad:
        return np.linalg.norm(value) < threshold


def shuffledata(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:,:cols-1]
    y = data[:,cols-1:]
    return X,y

def descent(data,theta,bathsize,stoptype,thresh,alpha):
    init_time = time.time()
    i = 0  #迭代次数
    k = 0  #batch
    X,y = shuffledata(data)
    grad = np.zeros_like(theta)
    costs = [cost(X,y,theta)]

    while True:
        grad = gradient(X[k:k + bathsize],y[k:k+bathsize],theta)
        if k >= n:
            k = 0
            X,y = shuffledata(data)
        theta = theta - alpha*grad
        costs.append(cost(X,y,theta))
        i += 1
        if stoptype == stop_iter:
            value = i
        elif stoptype == stop_cost:
            value = costs
        elif stoptype == stop_grad:
            value = grad

        if stopcriterion(stoptype,value,thresh):
            break
    retun theta,i-1,costs,grad,time.time() - init_time


def runexpe(data,theta,bathsize,stoptype,thresh,alpha):
    theta,iter,costs,grad,dur = descent(data,theta,bathsize,stoptype,thresh,alpha)
    if (data[:,1] > 2).sum() > 1:
        name = 'original'
    else:
        name = 'scaled'
    name += 'data - learning rate:{} - '.format(alpha)
    if bathsize == n:
        strdesctype = 'gradient'
    elif bathsize == 1:
        strdesctype = 'stochastic'
    else:
        strdesctype = 'mini-batch ({})'.format(bathsize)
    name += strdesctype + 'descent - stop:'
    if stoptype == stop_iter:
        strstop = '{} iterations'.format(thresh)
    elif stoptype == stop_cost:
        strstop = 'costs change < {}'.format(thresh)
    else:
        strstop = 'gradient norm < {}'.format(thresh)
    name = += strstop
    print('***{}\nTheta:{} - Iter:{} - Last cost:{} - Duration:{}s'.format(name,theta,iter,cost[-1],dur))
    plt.plot(np.arange(len(cost)),costs,'r');plt.xlabel('Iterations');plt.ylabel('Cost');plt.title(name.upper() + ' - Error vs. Iteration')
    return theta


def predict(X,theta):
    return [1 if x >=0.5 else 0 for x in model(X,theta)]















