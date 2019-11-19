import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from activation import IdentityActivator, SigmoidActivator, TanhActivator, ReluActivator
from utils import init_weight

class Layer(object):
    def __init__(self, input_dim, output_dim, activation=None, learning_rate=0.2):
        self.input_dim = input_dim              # 上层的神经元个数
        self.output_dim = output_dim            # 该层的神经元个数
        self.learning_rate = learning_rate

        if activation is None:
            self.activator = IdentityActivator()
        elif activation is "sigmoid":
            self.activator = SigmoidActivator()
        elif activation is "tanh":
            self.activator = TanhActivator()
        elif activation is "relu":
            self.activator = ReluActivator()
        else:
            raise Exception('Non-supported activation function')
        
        # 初始化权重矩阵，偏置项
        self.W, self.b = init_weight(input_dim, output_dim)

    # 单层的前向传播
    def forward(self, A_prev):
        self.A_prev = A_prev                                # 上层神经元的输出
        self.Z_curr = np.dot(self.W, A_prev) + self.b       # 本层神经元的输入
        self.A_curr = self.activator.forward(self.Z_curr)   # 本层神经元的输出
        return self.A_curr

    # 单层的反向传播
    def backward(self, dA_curr):
        m = dA_curr.shape[1]        # 样本数量
        self.dA_curr = dA_curr
        self.dZ_curr = self.dA_curr * self.activator.backward(self.Z_curr)
        self.dW_curr = np.dot(self.dZ_curr, self.A_prev.T) / m
        self.db_curr = np.sum(self.dZ_curr, axis=1, keepdims=True) / m
        self.dA_prev = np.dot(self.W.T, self.dZ_curr)
        return self.dA_prev

    # 权值更新
    def update(self):
        self.W -= self.learning_rate * self.dW_curr
        self.b -= self.learning_rate * self.db_curr

class FNN(object):
    def __init__(self, input_dim, nn_architecture, learning_rate):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.num_of_layers = len(nn_architecture)
        self.layers = []
        prev_dim = input_dim
        for output_dim, activation in nn_architecture:
            self.layers.append(Layer(prev_dim, output_dim, activation, learning_rate))
            prev_dim = output_dim
        
    def full_forward_propagation(self, X):
        A_prev = X
        for i in range(self.num_of_layers):
            A_curr = self.layers[i].forward(A_prev)
            A_prev = A_curr

        # 最终输出等于最后一层的结果
        self.Y_pred = A_curr
        return self.Y_pred
    
    def full_backward_propagation(self, Y):
        dA_curr = - (np.divide(Y, self.Y_pred) - np.divide(1 - Y, 1 - self.Y_pred))
        for i in reversed(range(self.num_of_layers)):
            dA_prev = self.layers[i].backward(dA_curr)
            dA_curr = dA_prev

    def update_weight(self):
        for i in reversed(range(self.num_of_layers)):
            self.layers[i].update()

    def calc_cost(self, Y, Y_pred):
        m = Y_pred.shape[1]
        cost = -1 / m * (np.dot(Y, np.log(Y_pred).T) + np.dot(1 - Y, np.log(1 - Y_pred).T))
        return np.squeeze(cost)

    def calc_accuracy(self, Y, Y_pred):
        Y_pred_ = np.copy(Y_pred)
        Y_pred_[Y_pred_ > 0.5] = 1
        Y_pred_[Y_pred_ <= 0.5] = 0
        return (Y_pred_ == Y).all(axis=0).mean()

    def train(self, X, Y, epochs):
        for i in range(epochs):
            Y_pred = self.full_forward_propagation(X)
            self.full_backward_propagation(Y)
            self.update_weight()
            
            cost = self.calc_cost(Y, Y_pred)
            accuracy = self.calc_accuracy(Y, Y_pred)
            if (i % 500 == 0):
                print("Iteration: {} - cost: {} - accuracy: {}".format(i, cost, accuracy))

    def summary(self):
        for i in range(self.num_of_layers):
            print("Layer %d: input_dim: %d output_dim %d" % (i, self.layers[i].input_dim, self.layers[i].output_dim), end='\t')
            print("W: %s b: %s" %(self.layers[i].W.shape, self.layers[i].b.shape))


# the function making up the graph of a dataset
def make_plot(X, y, plot_name, file_name=None):
    plt.style.use('classic')
    plt.figure(figsize=(16,12))
    axes = plt.gca()
    axes.set(xlabel="$X_1$", ylabel="$X_2$")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='black')
    if (file_name):
        plt.savefig(file_name)
        plt.close()

if __name__ == "__main__":
    X, y = make_moons(n_samples = 1000, noise=0.2, random_state=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    make_plot(X, y, "Dataset")

    nn_architecture = [(25, 'relu'), (50, 'relu'), (50, 'relu'), (25, 'relu'), (1, 'sigmoid')]

    myFNN = FNN(input_dim=2, nn_architecture=nn_architecture, learning_rate=0.1)
    # myFNN.summary()
    myFNN.train(X_train.T, y_train.reshape((y_train.shape[0], 1)).T, 10000)
    
    Y_test_pred = myFNN.full_forward_propagation(X_test.T)
    acc_test = myFNN.calc_accuracy(y_test.reshape((y_test.shape[0], 1)).T, Y_test_pred)
    print("Test set accuracy: {}".format(acc_test)) 