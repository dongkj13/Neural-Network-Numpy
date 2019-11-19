import numpy as np

def init_weight(input_dim, output_dim):
    np.random.seed()
    weight = np.random.randn(output_dim, input_dim) * 0.1
    bias = np.random.randn(output_dim, 1) * 0.1
    return weight, bias