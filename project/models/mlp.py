
import haiku as hk
import jax 


class Mlp(hk.Module):

  def __init__(self, name, output_sizes=[]):

    super().__init__(name=name)

    self.layers = []

    for output_size in output_sizes:
      self.layers.append(hk.Linear(output_size=output_size))

  def __call__(self, x, output_intermediate=False):

    if output_intermediate:
      outputs = []

    x = hk.Flatten()(x)

    num_layers = len(self.layers)
    for i, layer in enumerate(self.layers):
      x = layer(x)
      if i < num_layers-1:
        x = jax.nn.relu(x)

      if output_intermediate:
        outputs.append(x)

    if output_intermediate:
      return outputs
    
    return x

def lenet_fn(x):
  layers = [300,100,10]
  net = Mlp('mlp',layers)
  return net(x,output_intermediate=False)