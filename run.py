from network import Network, Layer, WeightMatrix
from training import BackPropagation
from activationfunction import IdentityActivationFunction, SigmodActivationFunction
from optimization import GradientDescent
from costfunction import SquaredErrorCostFunction

network = Network([Layer(2, IdentityActivationFunction), Layer(2, SigmodActivationFunction), Layer(3, SigmodActivationFunction), Layer(1, SigmodActivationFunction)])
backpropogation = BackPropagation(network, SquaredErrorCostFunction)
weight_updates = backpropogation([0,0], 0)
gradient_descent = GradientDescent()
for index, weight_matrix in enumerate(network.weight_matrices):
    network.weight_matrices[index] = gradient_descent(weight_matrix.matrix, weight_updates[index])
    
