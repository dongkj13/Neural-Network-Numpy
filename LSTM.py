import random
import numpy as np
from utils import init_weight

def sigmoid(x): 
    return 1. / (1 + np.exp(-x))

def sigmoid_derivative(values): 
    return values*(1-values)

def tanh_derivative(values): 
    return 1. - values ** 2

class LstmParam:
    def __init__(self, units, input_dim):
        self.units = units
        self.input_dim = input_dim
        concat_len = input_dim + units

        self.wg = init_weight(units, concat_len)
        self.wi = init_weight(units, concat_len)
        self.wf = init_weight(units, concat_len)
        self.wo = init_weight(units, concat_len)
        self.bg = init_weight(units)
        self.bi = init_weight(units)
        self.bf = init_weight(units)
        self.bo = init_weight(units)
        # derivative of loss function w.r.t. all parameters
        self.wg_diff = np.zeros((units, concat_len)) 
        self.wi_diff = np.zeros((units, concat_len)) 
        self.wf_diff = np.zeros((units, concat_len)) 
        self.wo_diff = np.zeros((units, concat_len)) 
        self.bg_diff = np.zeros(units) 
        self.bi_diff = np.zeros(units)  
        self.bf_diff = np.zeros(units)  
        self.bo_diff = np.zeros(units)  

    def update_param(self, lr = 1):
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff
        # reset diffs to zero
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi) 
        self.wf_diff = np.zeros_like(self.wf) 
        self.wo_diff = np.zeros_like(self.wo) 
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi) 
        self.bf_diff = np.zeros_like(self.bf) 
        self.bo_diff = np.zeros_like(self.bo) 

class LstmState:
    def __init__(self, units, input_dim):
        self.g = np.zeros(units)
        self.i = np.zeros(units)
        self.f = np.zeros(units)
        self.o = np.zeros(units)
        self.s = np.zeros(units)
        self.h = np.zeros(units)
        self.dh_prev = np.zeros_like(self.h)
        self.dL_ds_prev = np.zeros_like(self.s)
    
class LstmCell:
    def __init__(self, lstm_param, lstm_state):
        self.state = lstm_state
        self.param = lstm_param
        # non-recurrent input concatenated with recurrent input
        self.xc = None

    def forward(self, x, s_prev = None, h_prev = None):
        # if this is the first lstm node in the network
        if s_prev is None: s_prev = np.zeros_like(self.state.s)
        if h_prev is None: h_prev = np.zeros_like(self.state.h)

        self.s_prev = s_prev
        self.h_prev = h_prev

        # concatenate x(t) and h(t-1)
        self.xc = np.hstack((x,  h_prev))
        self.state.g = np.tanh(np.dot(self.param.wg, self.xc) + self.param.bg)
        self.state.i = sigmoid(np.dot(self.param.wi, self.xc) + self.param.bi)
        self.state.f = sigmoid(np.dot(self.param.wf, self.xc) + self.param.bf)
        self.state.o = sigmoid(np.dot(self.param.wo, self.xc) + self.param.bo)
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        # self.state.h = np.tanh(self.state.s) * self.state.o
        self.state.h = self.state.s * self.state.o      # peephole LSTM

    def backward(self, dh, dL_plus_1_ds):
        # dh = dL(t) / dh(t)
        # dL_plus_1_ds = dL(t+1) / ds(t)
        # ds = self.state.o * dh * tanh_derivative(np.tanh(self.state.s)) + dL_plus_1_ds       # dL(t) / ds(t)
        # do = np.tanh(self.state.s) * dh                      # dL(t) / do(t)
        ds = self.state.o * dh  + dL_plus_1_ds      # dL(t) / ds(t)
        do = self.state.s * dh                      # dL(t) / do(t)
        di = self.state.g * ds                      # dL(t) / di(t)
        dg = self.state.i * ds                      # dL(t) / dg(t)
        df = self.s_prev * ds                       # dL(t) / df(t)

        # diffs w.r.t. vector inside sigma / tanh function
        di_input = sigmoid_derivative(self.state.i) * di
        df_input = sigmoid_derivative(self.state.f) * df
        do_input = sigmoid_derivative(self.state.o) * do
        dg_input = tanh_derivative(self.state.g) * dg

        # diffs w.r.t. inputs
        self.param.wi_diff += np.outer(di_input, self.xc)
        self.param.wf_diff += np.outer(df_input, self.xc)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input

        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        # save prev diffs
        self.state.dL_ds_prev = ds * self.state.f           # dL(t) / ds(t-1)
        self.state.dh_prev = dxc[self.param.input_dim:]     # dL(t-1) / dh(t-1)

class LstmNetwork():
    def __init__(self, lstm_param):
        self.lstm_param = lstm_param
        self.lstm_cell_list = []

    def create_lstm_cell_list(self, time_step):
        for _ in range(time_step):
            lstm_state = LstmState(self.lstm_param.units, self.lstm_param.input_dim)
            self.lstm_cell_list.append(LstmCell(self.lstm_param, lstm_state))

    def forward_propagation_through_time(self, x_list):
        # create all lstm cell list
        T = len(x_list)
        self.create_lstm_cell_list(T)

        s_prev = h_prev = None
        for idx in range(T):
            self.lstm_cell_list[idx].forward(x_list[idx], s_prev, h_prev)
            s_prev = self.lstm_cell_list[idx].state.s
            h_prev = self.lstm_cell_list[idx].state.h

    def backward_propagation_through_time(self, y_list, loss_function):
        T = len(y_list)
        loss = 0
        dh_prev = np.zeros(self.lstm_param.units)
        dL_plus_1_ds = np.zeros(self.lstm_param.units)
        for idx in reversed(range(T)):
            loss += loss_function.loss(self.lstm_cell_list[idx].state.h, y_list[idx])       # Loss = sum(l(t))
            dh = loss_function.deriv_loss(self.lstm_cell_list[idx].state.h, y_list[idx])    # dl(t) / dh(t)
            dh += dh_prev                                                                   # dL(t+1) / dh(t)
            self.lstm_cell_list[idx].backward(dh, dL_plus_1_ds)
            dh_prev = self.lstm_cell_list[idx].state.dh_prev
            dL_plus_1_ds = self.lstm_cell_list[idx].state.dL_ds_prev
        return loss

class LossFunction:
    @classmethod
    def loss(self, pred, label):
        return np.sum((pred - label) ** 2) / 2

    @classmethod
    def deriv_loss(self, pred, label):
        diff = pred - label
        return diff

if __name__ == '__main__':
    np.random.seed(0)

    lstm_units = 100
    input_dim = 50
    time_step = 4

    lstm_param = LstmParam(lstm_units, input_dim)
    lstm_net = LstmNetwork(lstm_param)
    # y_list = [ -0.5,  0.2, -0.1, 0.7]
    y_list = [np.random.random(lstm_units) for _ in range(time_step)]
    x_list = [np.random.random(input_dim) for _ in range(time_step)]

    for epochs in range(1000):
        lstm_net.forward_propagation_through_time(x_list)
        loss = lstm_net.backward_propagation_through_time(y_list, LossFunction)
        lstm_param.update_param(lr=0.1)
        if epochs % 100 == 0:
            print("iter", "%2s" % str(epochs), end=": ")
            print("y_pred = [" +
                ", ".join(["% 2.5f" % lstm_net.lstm_cell_list[ind].state.h[0] for ind in range(len(y_list))]) +
                "]", end=", ")
            print("loss:", "%.3e" % loss)
    
    # 前5维输出
    for i in range(5):
        print("y_true[%d] = [%s]" %(i, ", ".join(["% 2.5f" % yy[i] for yy in y_list])))
        print("y_pred[%d] = [%s]" %(i, ", ".join(["% 2.5f" % lstm_net.lstm_cell_list[ind].state.h[i] for ind in range(len(y_list))])))
