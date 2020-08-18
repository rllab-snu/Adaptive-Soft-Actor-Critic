import numpy as np
import tensorflow as tf
import scipy.signal
EPS = 1e-8

def scale_holder():
    return tf.placeholder(dtype=tf.float32, shape=())

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


"""
Policies
"""

LOG_STD_MAX = 2

LOG_STD_MIN = -20

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)

    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    logp_a = gaussian_likelihood(a, mu, log_std)
    return mu, pi, logp_pi, logp_a

def apply_squashing_func(mu, pi, logp_pi, a, logp_a):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    a = tf.tanh(a)
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    logp_a -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - a**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi, logp_a


"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(400,300), activation=tf.nn.relu,
                     output_activation=None, policy=mlp_gaussian_policy, action_space=None):
    action_scale = action_space.high[0]
    a_unsqueeze = a / action_scale
    a_unsqueeze = tf.atanh(a_unsqueeze)
    # policy
    with tf.variable_scope('pi'):
        mu, pi, logp_pi, logp_a = policy(x, a_unsqueeze, hidden_sizes, activation, output_activation)
        mu, pi, logp_pi, logp_a = apply_squashing_func(mu, pi, logp_pi, a_unsqueeze, logp_a)

    # make sure actions are in correct range

    mu *= action_scale
    pi *= action_scale

    # vfs
    vf_mlp = lambda x: tf.squeeze(mlp(x, list(hidden_sizes) + [1], activation, None), axis=1)
    with tf.variable_scope('q1'):
        q1 = vf_mlp(tf.concat([x, a], axis=-1))
    with tf.variable_scope('q1', reuse=True):
        q1_pi = vf_mlp(tf.concat([x, pi], axis=-1))
    with tf.variable_scope('q2'):
        q2 = vf_mlp(tf.concat([x, a], axis=-1))
    with tf.variable_scope('q2', reuse=True):
        q2_pi = vf_mlp(tf.concat([x, pi], axis=-1))
    with tf.variable_scope('v'):
        v = vf_mlp(x)
    with tf.variable_scope('Q'):
        Q = vf_mlp(tf.concat([x, a], axis=-1))
    with tf.variable_scope('Q', reuse=True):
        Q_pi = vf_mlp(tf.concat([x, pi], axis=-1))
    with tf.variable_scope('R'):
        R = vf_mlp(x)
    return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v, Q, Q_pi, R