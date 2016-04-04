import numpy

class BackPropagation:

  def __init__(self, network, cost_function):
    self.network = network
    self.cost_function = cost_function()

  def __call__(self, input_data, expected_output):
    predicted_value = self.network.feed(input_data)
    gradient_output_layer = self._delta_output_layer(predicted_value, expected_output)
    layers_gradient = self._delta_hidden_layer(gradient_output_layer)
    return self._compute_weight_updates(layers_gradient)

  def _delta_output_layer(self, predicted_value, expected_value):
    delta_cost_function = self.cost_function.delta(predicted_value, expected_value)
    delta_output_layer = self.network.layers[-1].delta(delta_cost_function)
    return delta_output_layer

  def _delta_hidden_layer(self, gradient_output_layer):
    gradients = [gradient_output_layer]
    hidden_layers_and_weight_matrices = list(zip(self.network.layers[1:-1], self.network.weight_matrices[1:]))
    for layer, weight_matrix in reversed(hidden_layers_and_weight_matrices):
      deeper_layer_delta = self.network.backward(weight_matrix.matrix, gradients[-1])
      current_layer_gradient = layer.delta(deeper_layer_delta)
      gradients.append(current_layer_gradient)
    return reversed(gradients)

  def _compute_weight_updates(self, layers_gradient):
    weight_updates_list = []
    source_layers_and_layers_gradient = zip(self.network.layers[:-1], layers_gradient)
    for source_layer, target_layer_gradient in source_layers_and_layers_gradient:
      activations = numpy.insert(source_layer.output, 0, 1)
      weight_updates_list.append(numpy.outer(activations, target_layer_gradient))
    return weight_updates_list
