import numpy

class ActivationFunction:

    def __call__(self, inputs):
        raise NotImplementedError

    def delta(self, inputs, outputs, deeper_layers_delta):
        raise NotImplementedError


class SigmodActivationFunction(ActivationFunction):

  def __call__(self, inputs):
    return 1/(1+numpy.exp(-inputs))

  def delta(self, inputs, outputs, deeper_layers_delta):
    derivative = outputs * (1 - outputs)
    return derivative * deeper_layers_delta

class IdentityActivationFunction(ActivationFunction):

  def __call__(self, inputs):
    return inputs

  def delta(self, inputs, outputs, deeper_layers_delta):
        derivative = numpy.ones(inputs.shape).astype(float)
        return derivative * deeper_layers_delta
