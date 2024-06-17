# let's code a Value object
class Value():
  def __init__(self, data, _children=(), _op='', label=''): # this is to initialize, also called a constructor, the object with some initial attributes
    self.data = data
    self.grad = 0 # add a new attribute showing the grad of the end node with respect to this node
    self._prev = set(_children)
    self._backward = lambda: None # an anonymous fuction that returns none, as the default value of our backprop function
    # the backprop function here is changed to backward, to be consistent with pytorch's api
    self._op = _op
    self.label = label

  def __repr__(self): # returns  a string representation of the object
    return f"Value(label='{self.label}', data={self.data},  grad={self.grad})"

  def __add__(self, other):
    # define the add method, so that Python could interpret what happens when two values get added together
    # when we use the operator +
    other = other if isinstance(other, Value) else Value(other)
    output = Value(self.data + other.data, (self, other), '+')

    # we also need to set up what happens to the output's children's backprop functions when they add together to the output
    def _backward(): # impure function because this modifies state
      self.grad += 1 * output.grad # += because the grads stack together in the multivariate case (eg. when you have b = a + a; the grad is 2); see here https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/differentiating-vector-valued-functions/a/multivariable-chain-rule-simple-version
      other.grad += 1 * output.grad
    output._backward = _backward

    return output

  def __radd__(self, other):
    return self + other

  def __mul__(self, other):
    # define the mul method, for the * operator
    other = other if isinstance(other, Value) else Value(other)
    output = Value(self.data * other.data, (self, other), '*')

    # set up the backprop function when __mul__ happens
    def _backward():
      self.grad += other.data * output.grad # +=, same reasoning as for the add above
      other.grad += self.data * output.grad
    output._backward = _backward

    return output

  def __rmul__(self, other):
    return self * other

  def __pow__(self, other):
    # now for this implementation we'll only accept others that are float or ints
    assert isinstance(other, (int, float)), "only supports int or float powers/exponents for now"
    output = Value(self.data**other, (self, ), f'**{other}')

    def _backward():
      self.grad += other * (self.data**(other-1)) * output.grad
    output._backward = _backward

    return output

  def __truediv__(self, other):
    return self * other**-1

  def __rtruediv__(self, other):
    return self**-1 * other

  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)

  def __rsub__(self, other):
    return (-self) + other

  def exp(self):
    # define exponentiation
    output = Value(math.exp(self.data), (self, ), 'exp')

    # set up the backprop function
    def _backward():
      self.grad = output.data * output.grad
      # you shouldn't use += here; causes some bug where the backprop gets called before the output grad settles down to some value
      # the order of the topo sort is a bit random so you don't actually add the grads together, but you add an intermediate input which messes the calculation
      # self.grad += output.data * output.grad
    output._backward = _backward

    return output

  def tanh(self): # we can implement exponentiation and division methods here, but we're just going to use the math library
  # karpathy said that we could implement this however simple or complex or atomic we want; and the activation function could be simpler than this
  # what activation function simply does is *squashing* the input into some Range in the real numbers
    output = (math.exp(self.data*2) - 1) / (math.exp(self.data*2) + 1) # definition of tanh
    output = Value(output, (self, ), 'tanh') # wrap the output inside a Value object

    def _backward():
      self.grad += (1 - output.data**2) * output.grad # should be +=; was = before because of bad topo_sort implementation
    output._backward = _backward

    return output

  def relu(self):
    # define a relu activation function
    output = Value(0 if self.data<0 else self.data, (self), 'relu')

    def _backward():
      self.grad += (self.data>0) * output.grad # at 0 the grad is also 0, but why??
    output._backward = _backward

    return output

  def backward(self):
    # first we need to turn the graph into a topo sort
    # sorta flatten it into a list that can be traversed serially
    topo = []
    seen = set()

    def topo_sort(n):
      if n not in seen:
        seen.add(n)
        for c in n._prev:
          topo_sort(c)
        topo.append(n) # this should be in the if block; so that you don't have nodes that's backpropped out of order
        # easiest example of this going awry if topo.append(n) is not inside the if block is this
        # say you have a node that goes like this (sorta like MLP)
        #     O
        # O <   > O
        #     O
        # imagine how this would go if topo.append(n) is outside of the if block
        # it would be appended twice !!

    topo_sort(self)

    # then do the backprop along the topo sort thing that we have
    for n in reversed(topo):
      n._backward() # calls the backward function
    
    return None # doesn't return anything