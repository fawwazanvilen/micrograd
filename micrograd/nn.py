# imports
import random # need this here
from micrograd.engine import Value # this is cool lmao

# define a parent class for inheritance
# of some common methods
class Module:
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0
    
  def parameters(self):
    # to be overrided
    # default implementation so that child classes that don't have parameters
    # don't return an error
    return []

# make the Neuron class
class Neuron(Module):
  def __init__(self, n_in: int, nonlin: bool = True) -> None:
    self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
    self.b = Value(random.uniform(-1, 1))
    self.nonlin = nonlin

  def __call__(self, x):
    # w*x + b
    activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
    # output = activation.tanh() # if using tanh
    # output = activation.relu() # default will be relu
    return activation.relu() if self.nonlin else activation

  def parameters(self):
    return self.w + [self.b]

  def __repr__(self):
    return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

# make the Layer class (which is just a list of neurons)
class Layer(Module):
  def __init__(self, n_in: int, n_out: int, **kwargs) -> None:
  # we use **kwargs here so that it can receive arguments like nonlin
  # and any other that we might add later here 
    self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]

  def __call__(self, x):
    output = [n(x) for n in self.neurons]
    return output[0] if len(output) == 1 else output

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]
  
  def __repr__(self):
    return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

# make the MLP class (which is just a list of layers)
class MLP(Module):
  def __init__(self, n_in: int, n_outs: list) -> None:
    size = [n_in] + n_outs
    self.layers = [Layer(size[i], size[i+1], nonlin=i!=len(size)-2) for i in range(len(size)-1)] # karpathy uses range(len(n_outs)) here
    # nonlin=i!len(size)-2 this means that we want the last layer to be
    # linear, ie. not activated with relu/tanh

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

  def __repr__(self):
    return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"