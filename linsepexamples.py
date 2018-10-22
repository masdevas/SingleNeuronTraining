from random import randint
import numpy as np

def generate_linear_separable_examples(count_example, high_limits):
    res_list = list()
    norm_for_subspace = []
    # In x1 * n1 + .. + xn * nn = S, S is 1, xi is 1 / high_limits[i] 
    norm_for_subspace = np.array([1 / limit for limit in high_limits])
    dimension = len(high_limits)
    for index_example in range(count_example):
        example = np.array([randint(0, high_limits[index_dimension]) for index_dimension in range(dimension)])
        if np.dot(example, norm_for_subspace) >= 1:
            res_list.append((example, 1))
        else:
            res_list.append((example, 0))
    return res_list
