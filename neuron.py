from math import exp
from math import log
import numpy as np
import linsepexamples

def sigmoid(arg):
    return 1 / ( 1 + exp(-arg))

def log_create(name_datafile):
    open(name_datafile, 'w').close()

def log_set(name_datafile, name_set, examples_set):
    with open(name_datafile, 'a') as log_set:
        log_set.write("\n\n" + name_set + " : \n\n")
        for example in examples_set:
            log_set.write(str(example) + "\n")

# Cross Entropy is loss function:
# L(D) = - 1/n * ( y1 * ln(f(x1)) + (1 - y1) * (ln(1 - f(x1))) + ... 
#   ... +  yn * ln(f(xn)) + (1 - yn) * (ln(1 - f(xn)))),
#   Where n - size of batch; yi - mark on example i; xi - example i; f - sigmoid(linearcomb)
# Note : end_index not include
def cross_entropy(trained_weights, examples_set, index_begin_example, index_end_example):
    cr_entropy = np.float128(0)
    for index_example in range(index_begin_example, index_end_example):
        for_show = 1 - sigmoid(np.dot(trained_weights, examples_set[index_example][0]))
        mini_const = 7e-300
        cr_entropy -= \
            examples_set[index_example][1] * \
            log(sigmoid(np.dot(trained_weights, examples_set[index_example][0])) + mini_const) + \
            (1 - examples_set[index_example][1]) * \
            log(1 - sigmoid(np.dot(trained_weights, examples_set[index_example][0])) + mini_const)
    return cr_entropy / (index_end_example - index_begin_example)

def gradient_cross_entropy(trained_weights, examples_set, index_example):
    list_for_gradient = []
    sigmoid_value = sigmoid(np.dot(trained_weights, examples_set[index_example][0]))
    mark_of_class = examples_set[index_example][1]
    for index_gradient_component in range(len(trained_weights)):
        component_of_gradient = (mark_of_class * (1 - sigmoid_value) - (1 - mark_of_class) * \
            sigmoid_value) * examples_set[index_example][0][index_gradient_component]
        list_for_gradient.append(component_of_gradient)
    gradient_loss = np.array(list_for_gradient)
    return gradient_loss

def update_trained_weights(trained_weights, gradient_loss, learning_rate):
    print
    trained_weights += learning_rate * gradient_loss

def stochastic_gradient_descent(trained_weights, training_set, learning_rate):
    for index_example in range(len(training_set)):
        gradient_loss = gradient_cross_entropy(trained_weights, training_set, index_example)
        update_trained_weights(trained_weights, gradient_loss, learning_rate)

def do_train():
    dimensions = [10, 10]
    name_datafile = 'neuron_data.txt'
    name_lossesfile = 'neuron_loss.txt'
    count_training_examples = 100
    count_tests_examples = 100
    count_ages = 500
    learning_rate = np.float128(0.002)
    trained_weights = np.array([np.float128(0.5)] * len(dimensions))
    log_create(name_datafile)
    log_create(name_lossesfile)
    training_set = linsepexamples.generate_linear_separable_examples( \
            count_training_examples, dimensions)
    log_set(name_datafile, "Training set", training_set)
    test_set = linsepexamples.generate_linear_separable_examples( \
            count_tests_examples, dimensions)
    log_set(name_datafile, "Test set", test_set)
    with open(name_lossesfile, "w") as log_losses:
        for number_age in range(count_ages):
            stochastic_gradient_descent(trained_weights, training_set, learning_rate)
            loss = cross_entropy(trained_weights, test_set, 0, len(test_set))
            log_losses.write(" Age " + str(number_age) + ", Loss " + str(loss) + \
                " Weights " + str(trained_weights) + "\n")


do_train()
