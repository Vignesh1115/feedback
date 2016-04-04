class CostFunction:

    def __call__(self, predicted_value, expected_value):
        raise NotImplementedError

    def delta(self, predicted_value, expected_value):
        raise NotImplementedError

class SquaredErrorCostFunction(CostFunction):

  def __call__(self, predicted_value, expected_value):
    return (predicted_value - expected_value) ** 2 / 2

  def delta(self, predicted_value, expected_value):
        return predicted_value - expected_value
