import sys
import subprocess
repo_dir = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
sys.path.append(repo_dir)
import tensorflow as tf
import numpy as np
from scipy.stats import norm
from GeneralMMOT.Distributions import gen_margs, gen_theta, gen_comparison, gen_OT

DIM = 20
T = 2  # Here: Just so covariance is the same (random seed is used correctly)
GAMMA = 500*DIM
BATCH_MARGINAL =  2 ** 9
BATCH_SIZE = (2 ** 8) * (2 ** (int(round(np.log2(DIM)))))
N = 20000
N_FINE = 20000
FINE_S = 2500
MINMAX = 1  # Multiplier for objective function


# Objective Function
STRIKE = 0
def f(s):
    # s is input of shape [K, T, DIM], returns tensor of shape [BATCH_SIZE]
    # return MINMAX * (s[:, 1, 0] - s[:, 1, 1]) ** 2
    # return MINMAX * tf.reduce_sum(tf.nn.relu(s[:, 1:2, 0] - s[:, 0:1, 0]), axis=1)
    # return tf.reduce_sum(s[:, T-1:T, :], axis=1)
    return MINMAX * tf.nn.relu(tf.reduce_sum(s, axis=1) - STRIKE)


NCD = 10
skt = 0
skmt = 0
gen_c = gen_comparison(2 ** 17, T, DIM)
gen_m = gen_margs(2 ** 17, T, DIM)
for ci in range(NCD):
    s_m = next(gen_m)
    s = next(gen_c)
    sk = np.mean(np.maximum(np.sum(s, axis=1) - STRIKE, 0))
    skm = np.mean(np.maximum(np.sum(s_m[:, T-1, :], axis=1) - STRIKE, 0))
    skt += sk/NCD
    skmt += skm/NCD
print(skt)
print(skmt)

# feed forward network structure
def univ_approx(x, name, hidden_dim=64, input_dim=1, output_dim=1):
    # returns tensor of shape [BATCH_SIZE, output_dim]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        ua_w = tf.get_variable('ua_w', shape=[input_dim, hidden_dim],
                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        ua_b = tf.get_variable('ua_b', shape=[hidden_dim], initializer=tf.contrib.layers.xavier_initializer(),
                               dtype=tf.float32)
        z = tf.matmul(x, ua_w) + ua_b
        a = tf.nn.relu(z)
        ua_w2 = tf.get_variable('ua_w2', shape=[hidden_dim, hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        ua_b2 = tf.get_variable('ua_b2', shape=[hidden_dim], initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
        z2 = tf.matmul(a, ua_w2) + ua_b2
        a2 = tf.nn.relu(z2)
        ua_w3 = tf.get_variable('ua_w3', shape=[hidden_dim, hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        ua_b3 = tf.get_variable('ua_b3', shape=[hidden_dim], initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
        z3 = tf.matmul(a2, ua_w3) + ua_b3
        a3 = tf.nn.relu(z3)
        ua_w4 = tf.get_variable('ua_w4', shape=[hidden_dim, output_dim],
                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        ua_b4 = tf.get_variable('ua_b4', shape=[output_dim], initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
        z = tf.matmul(a3, ua_w4) + ua_b4
    return z


S_marg = tf.placeholder(dtype=tf.float32, shape=[None, DIM])
S_theta = tf.placeholder(dtype=tf.float32, shape=[None, DIM])
s1 = 0
for j in range(DIM):
    s1 += tf.reduce_sum(univ_approx(S_marg[:, j:j+1], str(j)), axis=1)

ints = tf.reduce_mean(s1)

s1_mu = 0
for j in range(DIM):
    s1_mu += tf.reduce_sum(univ_approx(S_theta[:, j:j+1], str(j)), axis=1)

s2_mu = 0

fvar = f(S_theta)
obj_fun = ints + GAMMA * tf.reduce_mean(tf.square(tf.nn.relu(fvar - s1_mu - s2_mu)))

global_step = tf.Variable(0, trainable=False)
train_op_fine = tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(0.0001, global_step, N_FINE/FINE_S, 0.995, staircase=False), beta1=0.99, beta2=0.995).minimize(obj_fun)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    vals = []
    gen_marginals = gen_OT(BATCH_MARGINAL, T, DIM, type='MultiNormal')
    gen_ref = gen_OT(BATCH_SIZE, T, DIM, type='MultiNormal')
    for t in range(1, N+1):
        sample_marginals = next(gen_marginals)
        sample_ref = next(gen_ref)

        (c, _) = sess.run([obj_fun, train_op_fine], feed_dict={S_marg: sample_marginals, S_theta: sample_ref, global_step: 1})
        vals.append(c)
        if t%100 == 0:
            print(t)
            print(np.mean(vals[t-2000:t]))
    for t in range(N+1, N+N_FINE+1):
        sample_marginals = next(gen_marginals)
        sample_ref = next(gen_ref)
        (c, _) = sess.run([obj_fun, train_op_fine], feed_dict={S_marg: sample_marginals, S_theta: sample_ref, global_step: t-N})
        vals.append(c)
        if t%100 == 0:
            print(t)
            print(np.mean(vals[t-2000:t]))
