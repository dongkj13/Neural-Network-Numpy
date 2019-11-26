import numpy as np
np.random.seed()

def init_weight(*args):
    weight = np.random.randn(*args) * 0.1
    return weight