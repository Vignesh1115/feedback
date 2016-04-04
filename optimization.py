class GradientDescent:

  def __call__(self, weight_matrix, weight_updates, learning_rate=0.1):
    return weight_matrix - weight_updates * learning_rate
