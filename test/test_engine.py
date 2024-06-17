# we test our micrograd engine against pytorch here
# imports
import torch
from micrograd.engine import Value

# this will largely follow karpathy's implementation
def test_sanity_check():
  # micrograd
  x = Value(-4)
  z = 2*x + 2 + x
  q = z.relu() + z*x
  h = (z*z).relu()
  y = h + q + q*x
  y.backward()
  xmg, ymg = x, y

  # pytorch
  x = torch.Tensor([-4]).double()
  x.requires_grad = True
  z = 2*x + 2 + x
  q = z.relu() + z*x
  h = (z*z).relu()
  y = h + q + q*x
  y.backward()
  xpt, ypt = x, y

  # if forward pass went well
  assert ymg.data == ypt.data.item() # in pytorch you use .item() to get the element out of a single-element array

  # backward pass went well
  assert xmg.grad == xpt.grad.item()

def test_more_ops():
  # micrograd
  a = Value(-4)
  b = Value(2)
  c = a + b
  d = 