
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class MOE:
    def __init__(self, train_x, train_y, k = 4, lamda = 0.1,inter = 50, lazy = 0.99, type = 'classification'):
        self._x = np.array(train_x)
        self._y = np.array(train_y)
        one = np.ones([len(train_x), 1])
        self._x = np.hstack([one, self._x])
        self._m, self._n = self._x.shape
        self._label_num = self._y.shape[1]


        self._k = k
        self._lamda = lamda
        self._inter = inter
        self._lazy = lazy
        self._type = type

        self._M = np.random.random_sample((self._k, self._n)) * 0.02 - 0.1
        self._V = np.random.random_sample((self._k, self._n, self._label_num)) * 0.02 - 0.1
        self._loglsoo_list = []
    def sgdTrain(self, test_x, test_y, ):
        self._test_x = np.array(test_x)
        self._test_y = np.array(test_y)
        one = np.ones([len(test_x), 1])
        self._test_x = np.hstack([one, self._test_x])

        iters = 0
        while(1):
            for t in range(self._m):
                xt = self._x[t, :]
                rt = self._y[t, :]
                g = np.exp(np.dot(self._M, xt.transpose())).transpose()
                g = g / np.sum(g, axis=0)
                w = np.zeros((self._label_num, self._k))
                for j in range(self._label_num):
                    w[j, :] = np.dot(self._V[:,:,j],xt.transpose()).transpose()
                yi = np.dot(w, g.transpose()).transpose()

                ysi = np.exp(yi)
                ysi = ysi / np.sum(ysi)

                yih = np.exp(w)
                yih = yih / np.array([np.sum(yih, axis = 0)] * self._label_num)

                fh = np.exp(np.sum(np.log(yih) * np.array([rt] * self._k).transpose(), axis = 0)) * g
                fh = fh / np.sum(fh)

                for r in range(self._k):
                    for j in range(self._label_num):
                        dv = self._lamda * (rt[j] - yih[j, r]) * fh[r] * xt
                        self._V[r, :, j] = self._V[r, :, j] + dv
                    dm = self._lamda * (fh[r] - g[r]) * xt
                    self._M[r, :] = self._M[r, :] + dm
            self._lamda = self._lamda * self._lazy
            logloss = self.logloss()
            self._loglsoo_list.append(logloss)
            print ("iters = ",iters, "logloss = ", logloss)
            iters = iters + 1

            if iters > self._inter:
                break
    def ftrlTrain(self, test_x, test_y, alfa = 1.0, beta = 0.2, lamda1 = 0.0, lamda2 = 0.0):
        self._test_x = np.array(test_x)
        self._test_y = np.array(test_y)
        one = np.ones([len(test_x), 1])
        self._test_x = np.hstack([one, self._test_x])

        self._alfa = alfa
        self._beta = beta
        self._lamda1 = lamda1
        self._lamda2 = lamda2

        z_v = np.zeros((self._k, self._n, self._label_num))
        z_m = np.zeros((self._k, self._n))
        n_v = np.zeros((self._k, self._n, self._label_num))
        n_m = np.zeros((self._k, self._n))

        iters = 0
        while(1):
            for t in range(self._m):
                xt = self._x[t, :]
                rt = self._y[t, :]
                g = np.exp(np.dot(self._M, xt.transpose())).transpose()
                g = g / np.sum(g, axis=0)
                w = np.zeros((self._label_num, self._k))
                for j in range(self._label_num):
                    w[j, :] = np.dot(self._V[:,:,j],xt.transpose()).transpose()
                yi = np.dot(w, g.transpose()).transpose()

                ysi = np.exp(yi)
                ysi = ysi / np.sum(ysi)

                yih = np.exp(w)
                yih = yih / np.array([np.sum(yih, axis = 0)] * self._label_num)

                fh = np.exp(np.sum(np.log(yih) * np.array([rt] * self._k).transpose(), axis = 0)) * g
                fh = fh / np.sum(fh)

                for r in range(self._k):
                    for j in range(self._label_num):
                        dv = -(rt[j] - yih[j, r]) * fh[r] * xt
                        sigma_v = (np.sqrt(n_v[r, :, j] + dv * dv) - np.sqrt(n_v[r, :, j])) / self._alfa
                        z_v[r, :, j] = z_v[r, :, j] + dv - sigma_v * self._V[r, :, j]
                        n_v[r, :, j] = n_v[r, :, j] + dv * dv
                        #for q in range(self._n):
                            #if(abs(z_v[r, q, j]) > self._lamda1):
                                #self._V[r, q, j] = -1.0 / ((self._beta + np.sqrt(n_v[r, q, j])) / self._alfa + self._lamda2) * (z_v[r, q, j] - np.sign(z_v[r, q, j]) * self._lamda1)
                            #else:
                                #self._V[r, q, j] = 0.0
                        self._V[r, :, j] = -1.0 / ((self._beta + np.sqrt(n_v[r, :, j])) / self._alfa + self._lamda2) * (z_v[r, :, j] - np.sign(z_v[r, :, j]) * self._lamda1)
                        index = np.where(np.abs(z_v[r, :, j]) <= self._lamda1)
                        self._V[r, index, j] = 0
                    dm = -(fh[r] - g[r]) * xt
                    sigma_m = (np.sqrt(n_m[r, :] + dm * dm) - np.sqrt(n_m[r, :])) / self._alfa
                    z_m[r, :] = z_m[r, :] + dm - sigma_m * self._M[r, :]
                    n_m[r, :] = n_m[r, :] + dm * dm
                    #for q in range(self._n):
                        #if(abs(z_m[r, q]) > self._lamda1):
                            #self._M[r, q] = -1.0 / ((self._beta + np.sqrt(n_m[r, q])) / self._alfa + self._lamda2) * (z_m[r, q] - np.sign(z_m[r, q]) * self._lamda1)
                        #else:
                            #self._M[r, q] = 0
                    self._M[r, :] = -1.0 / ((self._beta + np.sqrt(n_m[r, :])) / self._alfa + self._lamda2) * (z_m[r, :] - np.sign(z_m[r, :]) * self._lamda1)
                    index = np.where(np.abs(z_m[r, :]) <= self._lamda1)
                    self._M[r, index] = 0

            logloss = self.logloss()
            self._loglsoo_list.append(logloss)
            print ("iters = ",iters, "logloss = ", logloss)
            iters = iters + 1

            if iters > self._inter:
                break
    def ftrlTrainUpdate(self, test_x, test_y, alfa = 1.0, beta = 0.2, lamda1 = 0.0, lamda2 = 0.0):
        self._test_x = np.array(test_x)
        self._test_y = np.array(test_y)
        one = np.ones([len(test_x), 1])
        self._test_x = np.hstack([one, self._test_x])

        self._alfa = alfa
        self._beta = beta
        self._lamda1 = lamda1
        self._lamda2 = lamda2
        z_v = np.zeros((self._k, self._n, self._label_num))
        z_m = np.zeros((self._k, self._n))
        n_v = np.zeros((self._k, self._n, self._label_num))
        n_m = np.zeros((self._k, self._n))
        iters = 0
        while(1):
            for t in range(self._m):
                xt = self._x[t, :]
                rt = self._y[t, :]
                g = np.exp(np.dot(self._M, xt.transpose())).transpose()
                g = g / np.sum(g, axis=0)
                w = np.zeros((self._label_num, self._k))
                for j in range(self._label_num):
                    w[j, :] = np.dot(self._V[:,:,j],xt.transpose()).transpose()
                yi = np.dot(w, g.transpose()).transpose()

                ysi = np.exp(yi)
                ysi = ysi / np.sum(ysi)
                yih = np.exp(w)
                yih = yih / np.array([np.sum(yih, axis = 0)] * self._label_num)
                fh = np.exp(np.sum(np.log(yih) * np.array([rt] * self._k).transpose(), axis = 0)) * g
                fh = fh / np.sum(fh)
                #=============update V using FTRL(Follow-the-regularized-Leader)==============
                dv = -((np.array([rt] * self._k) - yih.transpose())* fh.reshape(self._k, 1)).reshape(self._k, 1,self._label_num) * \
                   np.array([np.array([xt] * self._label_num).transpose()] * self._k)
                sigma_v = (np.sqrt(n_v + dv * dv) - np.sqrt(n_v)) / self._alfa
                z_v = z_v + dv - sigma_v * self._V
                n_v = n_v + dv * dv

                self._V = -1.0 / ((self._beta + np.sqrt(n_v)) / self._alfa + self._lamda2) * (z_v - np.sign(z_v) * self._lamda1)
                index = np.where(np.abs(z_v) <= self._lamda1)
                self._V[index] = 0
                #=============update M using FTRL(Follow-the-regularized-Leader)==============
                dm = -(np.array(fh) - np.array(g)).reshape(self._k, 1) * np.array([xt] * self._k)
                sigma_m = (np.sqrt(n_m + dm * dm) - np.sqrt(n_m)) / self._alfa
                z_m = z_m + dm - sigma_m * self._M
                n_m = n_m + dm * dm
                self._M = -1.0 / ((self._beta + np.sqrt(n_m)) / self._alfa + self._lamda2) * (z_m - np.sign(z_m) * self._lamda1)
                index = np.where(np.abs(z_m) <= self._lamda1)
                self._M[index] = 0
            logloss = self.logloss()
            self._loglsoo_list.append(logloss)
            print ("iters = ",iters, "logloss = ", logloss)
            iters = iters + 1
            if iters > self._inter:
                break
    def deci(self, xi):
        xi = xi.reshape(1, self._n)
        w = np.zeros((self._label_num, self._k))
        for j in range(self._label_num):
            w[j, :] = np.dot(self._V[:, :, j], xi.transpose())[:, 0]
        g = np.exp(np.dot(self._M, xi.transpose()))
        g = g / np.sum(g)
        y = np.dot(w, g)
        p = np.exp(y)
        p = p / np.sum(p)
        return p
    def logloss(self):
        m = self._test_x.shape[0]
        logloss = 0
        for i in range(m):
            yi = self._test_y[i, :]
            p = self.deci(self._test_x[i, :])[:, 0]
            logloss = logloss + np.log2(np.power(p[0], yi[0])) + np.log2(np.power(p[1], yi[1]))
        return -logloss / m
    def predict(self, test_x):
        one = np.ones([len(test_x), 1])
        test_x = np.hstack([one, test_x])
        p_list = []
        for xi in test_x:
            p = self.deci(xi)
            p_list.append(p[:, 0])
        p_list = np.array(p_list)
        m, n = test_x.shape
        r = np.zeros((m, self._label_num))
        index = p_list[:, 0] >= 0.5
        r[index, 0] = 1
        index = p_list[:, 1] >= 0.5
        r[index, 1] = 1
        return r
def data():
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for line in open('train_x.txt', 'r'):
        ele = line.strip().split('\t')
        t = [float(e) for e in ele]
        train_x.append(t)
    for line in open('train_y.txt', 'r'):
        ele = line.strip().split('\t')
        t = [int(e) for e in ele]
        train_y.append(t)
    for line in open('test_x.txt', 'r'):
        ele = line.strip().split('\t')
        t = [float(e) for e in ele]
        test_x.append(t)
    for line in open('test_y.txt', 'r'):
        ele = line.strip().split('\t')
        t = [int(e) for e in ele]
        test_y.append(t)
    return [np.array(train_x), np.array(train_y),np.array(test_x), np.array(test_y),]

def loadCircleData(num_data):
    center = np.array([5.0, 5.0])
    radiu_inner = 2
    radiu_outer = 4
    num_inner = num_data / 3
    num_outer = num_data - num_inner

    data = []
    label= []
    theta = 0.0
    for i in range(int(num_inner)):
        pho = (theta % 360) * math.pi / 180
        tmp = np.zeros(2, np.float32)
        tmp[0] = radiu_inner * math.cos(pho) + np.random.rand(1) + center[0]
        tmp[1] = radiu_inner * math.sin(pho) + np.random.rand(1) + center[1]
        data.append(tmp)
        label.append([1,0])
        theta += 2

    theta = 0.0
    for i in range(int(num_outer)):
        pho = (theta % 360) * math.pi / 180
        tmp = np.zeros(2, np.float32)
        tmp[0] = radiu_outer * math.cos(pho) + np.random.rand(1) + center[0]
        tmp[1] = radiu_outer * math.sin(pho) + np.random.rand(1) + center[1]
        data.append(tmp)
        label.append([0,1])
        theta += 1

    X_train, X_test, y_train, y_test = train_test_split( data, label, test_size = 0.3, random_state = 42)

    return np.array(X_train),  np.array(X_test),  np.array(y_train),  np.array(y_test)

def loadBandData(num_samples):
    Mat_Label = np.array([[5.0, 2.], [5.0, 8.0]])
    label=np.zeros((num_samples, 2), np.int8)
    num_dim = Mat_Label.shape[1]
    data = np.zeros((num_samples, num_dim), np.float32)
    data[:num_samples / 2, :] = (np.random.rand(num_samples / 2, num_dim) - 0.5) * np.array(
        [3, 1]) + Mat_Label[0]
    label[:num_samples / 2, :]=np.array(int(num_samples / 2) *[[1,0]])
    data[num_samples / 2: num_samples, :] = (np.random.rand(num_samples / 2,
                                            num_dim) - 0.5) * np.array([3, 1]) + Mat_Label[1]
    label[num_samples / 2: num_samples, :]=np.array([[0,1]]*int(num_samples-num_samples/2))

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=42)

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

if __name__ == '__main__':
    train_x, test_x, train_y,  test_y = loadCircleData(1000)
    # train_x, test_x, train_y, test_y = loadBandData(1000)

    K = 4
    I = 100
    #original sgd
    sgd_moe = MOE(train_x, train_y, k = K, inter = I)
    sgd_moe.sgdTrain(test_x, test_y)
    p_list = sgd_moe.predict(test_x)
    fig = plt.figure(1)
    ax = fig.add_subplot(3,1,1)
    ax.plot(train_x[train_y[:, 0] == 1, 0], train_x[train_y[:, 0] == 1, 1], 'r*')
    plt.plot(train_x[train_y[:, 1] == 1, 0], train_x[train_y[:, 1] == 1, 1], 'b.')
    plt.legend(['positive', 'negtive'])
    plt.title('train data')
    ax = fig.add_subplot(3,1,2)
    ax.plot(test_x[p_list[:, 0] == 1, 0], test_x[p_list[:, 0] == 1, 1], 'r*')
    ax.plot(test_x[p_list[:, 1] == 1, 0], test_x[p_list[:, 1] == 1, 1], 'b.')
    ax.legend(['positive', 'negtive'])
    plt.title('original sgd prediction result')
    #ftrl
    ftrl_moe = MOE(train_x, train_y, k = K, inter = I)
    ftrl_moe.ftrlTrainUpdate(test_x, test_y)
    p_list = ftrl_moe.predict(test_x)
    ax = fig.add_subplot(3,1,3)
    ax.plot(test_x[p_list[:, 0] == 1, 0], test_x[p_list[:, 0] == 1, 1], 'r*')
    ax.plot(test_x[p_list[:, 1] == 1, 0], test_x[p_list[:, 1] == 1, 1], 'b.')
    plt.legend(['positive', 'negtive'])
    plt.title('ftrl prediction result')
    fig = plt.figure(2)
    plt.plot(sgd_moe._loglsoo_list, '-r.')
    plt.plot(ftrl_moe._loglsoo_list, '-bo')
    plt.xlabel("run number")
    plt.ylabel("log loss")
    plt.legend(['sgd logloss', "ftrl logloss"])
    plt.show()
