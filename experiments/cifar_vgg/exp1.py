import jax
import optax

from coherence.data import get_data, decimate, normalize, sanitize
from coherence.train_with_state import network_and_loss, do_training, update_params, net_accuracy
from coherence.models.cnn import cifar_vgg_11_fn
from coherence.models.mlp import lenet_fn

# load mnist data
train, train_eval, test_eval = get_data("cifar10",batch_size=20,format_fun=sanitize)

# cnn, loss, params
net, xent_loss = network_and_loss(cifar_vgg_11_fn)
params, state = net.init(jax.random.PRNGKey(42), next(train)["image"])

# optimization of network
opt = optax.adam(1e-3)

accuracy_fn = net_accuracy(net)
update_fn = update_params(opt,xent_loss)
opt_state = opt.init(params)

# train cnn
def fun():
    final_params = do_training(update_fn, accuracy_fn, params, state, opt_state, train, train_eval, test_eval, epochs=1001)
    return final_params

fun()
