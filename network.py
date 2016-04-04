import numpy

class WeightMatrix:

  def __init__(self, dimensions):
    self.matrix = numpy.random.rand(dimensions[0], dimensions[1])

class Layer:

  def __init__(self, size, activation_function):
    self.size = size
    self.activation_function = activation_function()
    self.input = numpy.zeros(size)
    self.output = numpy.zeros(size)

  def apply_activation_function(self, input):
    self.input = input
    output = self.activation_function(self.input)
    self.output = output

  def delta(self, deeper_layer_delta):
    return self.activation_function.delta(self.input, self.output, deeper_layer_delta)

class Network:

  def __init__(self, layers):
    self.layers = layers
    self.layer_sizes = tuple(layer.size for layer in self.layers)
    self.weight_matrices_dimensions = [(m+1, n) for m, n in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
    self.weight_matrices = []
    for weight_matrix_dimension in self.weight_matrices_dimensions:
      weight_matrix = WeightMatrix(weight_matrix_dimension)
      self.weight_matrices.append(weight_matrix)

  def feed(self, data):
    input_layer = self.layers[0].apply_activation_function(data)
    for previous_layer, current_layer, weight_matrix in zip(self.layers[:-1], self.layers[1:], self.weight_matrices):
      current_input = self.forward(weight_matrix.matrix, previous_layer.output)
      current_layer.apply_activation_function(current_input)
    return self.layers[-1].output

  def forward(self, weights, activations):
    activations = numpy.insert(activations, 0, 1)
    return activations.dot(weights)

  def backward(self, weights, activations):
    change = activations.dot(weights.transpose())
    return change[1:]
