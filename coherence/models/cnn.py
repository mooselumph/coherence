
import haiku as hk
import jax 

class ConvModule(hk.Module):

    def __init__(self,name,output_channels,**bn_config):

        super().__init__(name=name)

        self.conv = hk.Conv2D(output_channels=output_channels,stride=1,kernel_shape=3,padding="SAME",with_bias=True)
        
        # bn_config = dict(bn_config)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)
        bn_config.setdefault("decay_rate", 0.999)

        self.bn = hk.BatchNorm(**bn_config)


    def __call__(self, x, is_training, test_local_stats):
        x = self.conv(x)
        x = self.bn(x,is_training,test_local_stats)
        x = jax.nn.relu(x)
        return x


class Cnn(hk.Module):

  def __init__(self, name, plan=[], output_size=10, bn_config=dict()):

    super().__init__(name=name)

    layers = []

    for i, spec in enumerate(plan):
        if spec == 'M':
            layers.append(hk.MaxPool(name=f"max_pool_{i}",window_shape=2,strides=2,padding="SAME"))
        else:
            layers.append(ConvModule(name=f"conv_{i}",output_channels=spec,**bn_config))

    self.plan = plan
    self.layers = layers

    self.fc = hk.Linear(output_size=output_size)
    

  def __call__(self, x, is_training=True, test_local_stats=False, output_intermediate=False):

    if output_intermediate:
      outputs = []

    for spec, layer in zip(self.plan,self.layers):
      if spec == 'M':
        x = layer(x)
      else:
        x = layer(x,is_training=is_training,test_local_stats=test_local_stats)

      if output_intermediate:
        outputs.append(x)

    x = hk.Flatten()(x)
    x = self.fc(x)

    if output_intermediate:
      outputs.append(x)
      return outputs
    
    return x

def cifar_vgg_11_fn(x):
  plan = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
  net = Cnn('mlp',plan,output_size=10)
  return net(x,output_intermediate=False)