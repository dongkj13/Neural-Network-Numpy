import random
import numpy as np
from utils import init_weight

def sigmoid(x): 
    return 1. / (1 + np.exp(-x))

def sigmoid_derivative(values): 
    return values*(1-values)

def tanh_derivative(values): 
    return 1. - values ** 2

class RNNParam:
    def __init__(self, units, input_dim):
        self.units = units
        self.input_dim = input_dim
        concat_len = input_dim + units

        self.w = init_weight(units, concat_len)
        self.b = init_weight(units)
        # derivative of loss function w.r.t. all parameters
        self.w_diff = np.zeros((units, concat_len))
        self.b_diff = np.zeros(units)

    def update_param(self, lr = 1):
        self.w -= lr * self.w_diff
        self.b -= lr * self.b_diff
        # reset diffs to zero
        self.w_diff = np.zeros_like(self.w)
        self.b_diff = np.zeros_like(self.b)


class RNNState:
    def __init__(self, units, input_dim):
        self.h = np.zeros(units)
        self.dh_prev = np.zeros_like(self.h)
    
class RNNCell:
    def __init__(self, rnn_param, rnn_state):
        self.state = rnn_state
        self.param = rnn_param
        # non-recurrent input concatenated with recurrent input
        self.xc = None

    def forward(self, x, h_prev = None):
        # if this is the first lstm node in the network
        if h_prev is None: h_prev = np.zeros_like(self.state.h)

        self.h_prev = h_prev

        # concatenate x(t) and h(t-1)
        self.xc = np.hstack((x,  h_prev))
        self.state.h = np.tanh(np.dot(self.param.w, self.xc) + self.param.b)

    def backward(self, dh):
        # dh = dL(t) / dh(t)
        dh_input = tanh_derivative(self.state.h) * dh

        # diffs w.r.t. inputs
        self.param.w_diff += np.outer(dh_input, self.xc)
        self.param.b_diff += dh_input

        dxc = np.dot(self.param.w.T, dh_input)

        # save prev diffs
        self.state.dh_prev = dxc[self.param.input_dim:]     # dL(t-1) / dh(t-1)

class RNNNetwork():
    def __init__(self, rnn_param):
        self.rnn_param = rnn_param
        self.rnn_cell_list = []

    def create_rnn_cell_list(self, time_step):
        for _ in range(time_step):
            rnn_state = RNNState(self.rnn_param.units, self.rnn_param.input_dim)
            self.rnn_cell_list.append(RNNCell(self.rnn_param, rnn_state))

    def forward_propagation_through_time(self, x_list):
        # create all lstm cell list
        T = len(x_list)
        self.create_rnn_cell_list(T)

        h_prev = None
        for idx in range(T):
            self.rnn_cell_list[idx].forward(x_list[idx],  h_prev)
            h_prev = self.rnn_cell_list[idx].state.h

    def backward_propagation_through_time(self, y_list, loss_function):
        T = len(y_list)
        loss = 0
        dh_prev = np.zeros(self.rnn_param.units)
        for idx in reversed(range(T)):
            loss += loss_function.loss(self.rnn_cell_list[idx].state.h, y_list[idx])       # Loss = sum(l(t))
            dh = loss_function.deriv_loss(self.rnn_cell_list[idx].state.h, y_list[idx])    # dl(t) / dh(t)
            dh += dh_prev                                                                   # dL(t+1) / dh(t)
            self.rnn_cell_list[idx].backward(dh)
            dh_prev = self.rnn_cell_list[idx].state.dh_prev
        return loss

class LossFunction:
    @classmethod
    def loss(self, pred, label):
        return np.sum((pred[0] - label) ** 2) / 2

    @classmethod
    def deriv_loss(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        # diff = pred - label
        return diff

if __name__ == '__main__':
    np.random.seed(0)

    rnn_units = 100
    input_dim = 50
    time_step = 4

    rnn_param = RNNParam(rnn_units, input_dim)
    rnn_net = RNNNetwork(rnn_param)
    # y_list = [-0.5, 0.4, 0.1, -0.8]
    y_list = [np.random.uniform(-1, 1, 1) for _ in range(time_step)]
    x_list = [np.random.random(input_dim) for _ in range(time_step)]

    for epochs in range(5000):
        rnn_net.forward_propagation_through_time(x_list)
        loss = rnn_net.backward_propagation_through_time(y_list, LossFunction)
        rnn_param.update_param(lr=0.1)
        if epochs % 1000 == 0:
            print("iter", "%2s" % str(epochs), end=": ")
            print("y_pred = [" +
                ", ".join(["% 2.5f" % rnn_net.rnn_cell_list[ind].state.h[0] for ind in range(len(y_list))]) +
                "]", end=", ")
            print("loss:", "%.3e" % loss)
    
    # 前1维输出
    for i in range(1):
        print("y_true[%d] = [%s]" %(i, ", ".join(["% 2.5f" % yy[i] for yy in y_list])))
        print("y_pred[%d] = [%s]" %(i, ", ".join(["% 2.5f" % rnn_net.rnn_cell_list[ind].state.h[i] for ind in range(len(y_list))])))
