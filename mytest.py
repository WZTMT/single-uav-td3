import numpy as np
import torch
from env.multirotor import Multirotor, generate_target


def test():
    for i in range(100):
        x, y, z = generate_target()
        print(x, y, z)


if __name__ == '__main__':
    np.random.seed(1)
    x, y, z = generate_target()
    test()
