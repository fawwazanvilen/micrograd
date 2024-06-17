# we test our micrograd engine against pytorch here
# imports
import torch
from micrograd.engine import Value

# set PYTHONPATH=%CD%
# needed to use this at windows cmd to get python
# to recognize micrograd as a module

# this will largely follow karpathy's implementation
def test_sanity_check():
  # micrograd
  x = Value(-4.0)
  z = 2*x + 2 + x
  q = z.relu() + z*x
  # q = z + 1 # temp
  h = (z*z).relu()
  y = h + q + q*x
  y.backward()
  xmg, ymg = x, y

  # pytorch
  x = torch.Tensor([-4.0]).double()
  x.requires_grad = True
  z = 2*x + 2 + x
  q = z.relu() + z*x
  # q = z + 1 # temp
  h = (z*z).relu()
  y = h + q + q*x
  y.backward()
  xpt, ypt = x, y

  # if forward pass went well
  assert ymg.data == ypt.data.item(), f"Forward pass failed: ymg.data={ymg.data}, ypt.data={ypt.data.item()}"

  # backward pass went well
  assert xmg.grad == xpt.grad.item(), f"Backward pass failed: xmg.grad={xmg.grad}, xpt.grad={xpt.grad.item()}"


def test_more_ops():
  # micrograd
  a = Value(-4.0) # .0 so that it's float
  b = Value(2.0)
  c = a + b
  d = a*b + b**3
  c += c + 1
  c += 1 + c + (-a)
  d += d * 2 + (b + a).relu()
  d += 3 * d + (b - a).relu()
  e = c - d
  f = e**2
  g = f / 2.0
  g += 10.0 / f
  g.backward()
  amg, bmg, gmg = a, b, g
  
  # pytorch
  a = torch.Tensor([-4.0]).double()
  b = torch.Tensor([2.0]).double()
  a.requires_grad = True
  b.requires_grad = True
  c = a + b
  d = a*b + b**3
  c += c + 1
  c += 1 + c + (-a)
  d += d * 2 + (b + a).relu()
  d += 3 * d + (b - a).relu()
  e = c - d
  f = e**2
  g = f / 2.0
  g += 10.0 / f
  g.backward()
  apt, bpt, gpt = a, b, g

  # test
  # tolerance
  tolerance = 1e-6

  # if forward pass went well
  assert abs(gmg.data - gpt.data.item()) < tolerance, f"Forward pass failed: gmg.data={gmg.data}, gpt.data={gpt.data.item()}"

  # if backward pass went well
  assert abs(amg.grad - apt.grad.item()) < tolerance, f"Backward pass for a failed: amg.grad={amg.grad}, apt.grad={apt.grad.item()}"
  assert abs(bmg.grad - bpt.grad.item()) < tolerance, f"Backward pass for b failed: bmg.grad={bmg.grad}, bpt.grad={bpt.grad.item()}"

# do the tests
if __name__ == "__main__":
    test_sanity_check()
    test_more_ops()
    print("All tests passed!")