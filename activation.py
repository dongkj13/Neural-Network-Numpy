import numpy as np

class IdentityActivator():
    def forward(self, x):
        return x

    def backward(self, x):
        return 1.0


class SigmoidActivator(object):
    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, x):
        sig = self.forward(x)
        return sig * (1 - sig)


class TanhActivator(object):
    def forward(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def backward(self, x):
        tanh = self.forward(x)
        return 1 - tanh * tanh

class ReluActivator(object):
    def forward(self, x):
        return np.maximum(0,x)
    
    def backward(self, x):
        return 1.0 * (x > 0)
