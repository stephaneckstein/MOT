import sys
import subprocess
repo_dir = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
sys.path.append(repo_dir)
import tensorflow as tf
import numpy as np
from scipy.stats import norm
from GeneralMMOT.Distributions import gen_margs, gen_theta, gen_comparison, gen_OT
from sklearn.mixture import GaussianMixture
from NewTrainingTests.UpdatingSampler import Lap_update


DIM = 50
T = 2  # Here: Just so covariance is the same (random seed is used correctly)
GAMMA = 500
GAMMA_INC = 500  # increase GAMMA by this value each loop
PENALTY_POWER = 2
BATCH_MARGINAL = 2 ** 9
BATCH_SIZE = 2 ** 9
N_COMP = 25
# BATCH_SIZE = (2 ** 8) * (2 ** (int(round(np.log2(DIM)))))
N_0 = 5000
N = 5000
N_FINE = 5000
FINE_S = 1500
MINMAX = 1  # Multiplier for objective function
N_LOOP = 20

print(DIM)
print(GAMMA)
print(GAMMA_INC)
print(PENALTY_POWER)
print(BATCH_SIZE)
print(N_COMP)

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
gen_c = gen_comparison(2 ** 15, T, DIM)
gen_m = gen_margs(2 ** 15, T, DIM)
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
den = tf.nn.relu(fvar - s1_mu - s2_mu)
obj_fun = ints + GAMMA * tf.reduce_mean(tf.pow(den, PENALTY_POWER))

global_step = tf.Variable(0, trainable=False)
train_op_fine = tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(0.0001, global_step, N_FINE/FINE_S, 0.995, staircase=False), beta1=0.99, beta2=0.995).minimize(obj_fun)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    gen_marginals = gen_OT(BATCH_MARGINAL, T, DIM, type='MultiNormal')
    gen_functions = []

    for ind_loop in range(N_LOOP):
        gen_list = []

        n_gens = 1 + len(gen_functions)

        gen_list.append(gen_OT(int(round(BATCH_SIZE/n_gens)), T, DIM, type='MultiNormal'))

        for gen_f in gen_functions:
            gen_list.append(gen_f(int(round(BATCH_SIZE/n_gens))))

        def gen_ref():
            while True:
                sample = np.zeros([0, DIM])
                for gen in gen_list:
                    sample = np.append(sample, next(gen), axis=0)
                yield sample

        batch_loop = n_gens * int(round(BATCH_SIZE/n_gens))
        gen_r = gen_ref()

        good_samples = np.zeros([0, DIM])
        maxv = []
        vals = []

        for t in range(1, N_0+1):
            sample_marginals = next(gen_marginals)
            sample_ref = next(gen_r)

            (c, denv, _) = sess.run([obj_fun, den, train_op_fine],
                                    feed_dict={S_marg: sample_marginals, S_theta: sample_ref, global_step: 1})
            vals.append(c)
            maxv.append(np.max(denv))

            if t % 500 == 0:
                print(ind_loop)
                print(t)
                print(np.mean(vals[t - min(t, 1000):t]))

        for t in range(1, N+1):
            sample_marginals = next(gen_marginals)
            sample_ref = next(gen_r)
            (c, denv, _) = sess.run([obj_fun, den, train_op_fine],
                                    feed_dict={S_marg: sample_marginals, S_theta: sample_ref, global_step: 1})
            vals.append(c)
            maxv.append(np.max(denv))
            den_max = max(maxv[-1000:])
            u = np.random.random_sample([batch_loop])
            good_samples = np.append(good_samples, sample_ref[u * den_max <= denv, :], axis=0)

            if t % 500 == 0:
                print(ind_loop)
                print(t)
                print(np.mean(vals[N_0 + t - 1000:N_0 + t]))

        for t in range(1, N_FINE+1):
            sample_marginals = next(gen_marginals)
            sample_ref = next(gen_r)
            (c, denv, _) = sess.run([obj_fun, den, train_op_fine],
                                    feed_dict={S_marg: sample_marginals, S_theta: sample_ref, global_step: t})
            vals.append(c)
            maxv.append(np.max(denv))
            den_max = max(maxv[-1000:])
            u = np.random.random_sample([batch_loop])
            good_samples = np.append(good_samples, sample_ref[u * den_max <= denv, :], axis=0)

            if t % 500 == 0:
                print(ind_loop)
                print(t)
                print(np.mean(vals[N_0 + N + t - 1000:N_0 + N + t]))

        print('Primal value: ' + str(np.mean(np.maximum(0, np.sum(good_samples, axis=1) - STRIKE))))
        gen_functions.append(Lap_update(good_samples))
        GAMMA += GAMMA_INC
