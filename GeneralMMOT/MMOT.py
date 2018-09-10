import tensorflow as tf
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from GeneralMMOT.Distributions import gen_margs, gen_theta, up_sample

BATCH_SIZE = 2 ** 10
N = 15000
N_FINE = 15000
GAMMA = 1000
DIM = 5
T = 3
MINMAX = -1  # Multiplier for objective function
dist_type = 'MultiNormal'
ftype = 'basket0'

# Objective Function TODO: ADJUST FOR DIFFERENT FTYPES!
STRIKE = 0
def f(s):
    # s is input of shape [K, T, DIM], returns tensor of shape [BATCH_SIZE]
    # return MINMAX * (s[:, 1, 0] - s[:, 1, 1]) ** 2
    # return MINMAX * tf.reduce_sum(tf.nn.relu(s[:, 1:2, 0] - s[:, 0:1, 0]), axis=1)
    # return tf.reduce_sum(s[:, T-1:T, :], axis=1)
    return MINMAX * tf.nn.relu(tf.reduce_sum(s[:, T-1, :], axis=1) - STRIKE)


# feed forward network structure
def univ_approx(x, name, hidden_dim=32, input_dim=1, output_dim=1):
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


S_marg = tf.placeholder(dtype=tf.float32, shape=[None, T, DIM])
S_theta = tf.placeholder(dtype=tf.float32, shape=[None, T, DIM])
s1 = 0
for i in range(T):
    for j in range(DIM):
        s1 += tf.reduce_sum(univ_approx(tf.reduce_sum(S_marg[:, i:i+1, j:j+1], axis=1), str(i)+'_'+str(j)), axis=1)

ints = tf.reduce_mean(s1)

s1_theta = 0
for i in range(T):
    for j in range(DIM):
        s1_theta += tf.reduce_sum(univ_approx(tf.reduce_sum(S_theta[:, i:i+1, j:j+1], axis=1), str(i)+'_'+str(j)), axis=1)

s2_theta = 0
for i in range(T-1):
    # vi = tf.reshape(S[:, 0:i+1, :], [BATCH_SIZE, (i+1) * DIM])
    # the reshape results in [(t=1, d=1), (t=1, d=2), ..., (t=1, d=DIM), (t=2, d=1), ..., (t=(i+1), d=DIM)]
    s2_theta += tf.reduce_sum(univ_approx(tf.reshape(S_theta[:, 0:i+1, :], [BATCH_SIZE, (i+1) * DIM]), 'm'+str(i), input_dim=(i+1)*DIM, output_dim=DIM, hidden_dim=32+DIM*10) * (tf.reduce_sum(S_theta[:, i+1:i+2, :], axis=1) - tf.reduce_sum(S_theta[:, i:i+1, :], axis=1)), axis=1)

fvar = f(S_theta)
obj_fun = ints + GAMMA * tf.reduce_mean(tf.square(tf.nn.relu(fvar - s1_theta - s2_theta)))
train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.99, beta2=0.995).minimize(obj_fun)

global_step = tf.Variable(0, trainable=False)
train_op_fine = tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(0.0001, global_step, 8, 0.995, staircase=False), beta1=0.99, beta2=0.995).minimize(obj_fun)

density = tf.nn.relu(fvar - s1_theta - s2_theta)

vlist = tf.trainable_variables()
print([x.name for x in vlist])  # should be ordered 0_0, 0_1, ..., 0_DIM, 1_0, ..., 1_DIM, ..., (T-1)_DIM, m0, ..., m(T-2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    vals = []
    gen_marginals = gen_margs(BATCH_SIZE, T, DIM, type=dist_type)
    gen_ref = gen_theta(BATCH_SIZE, T, DIM, type=dist_type)
    for t in range(1, N+1):
        sample_marginals = next(gen_marginals)
        sample_ref = next(gen_ref)

        (c, _) = sess.run([obj_fun, train_op], feed_dict={S_marg: sample_marginals, S_theta: sample_ref})
        vals.append(c)
        if t%1000 == 0:
            print(t)
            print(np.mean(vals[t-1000:t]))
    for t in range(N+1, N+N_FINE+1):
        sample_marginals = next(gen_marginals)
        sample_ref = next(gen_ref)
        (c, _) = sess.run([obj_fun, train_op_fine], feed_dict={S_marg: sample_marginals, S_theta: sample_ref, global_step: t-N})
        vals.append(c)
        if t%1000 == 0:
            print(t)
            print(np.mean(vals[t-1000:t]))

    weights_tot = sess.run(vlist)
    weights = [[[] for i in range(T)], []]
    UNIV_VARS = 8
    for i in range(T):
        for j in range(DIM):
            weights[0][i].append(weights_tot[i*T*UNIV_VARS+j*UNIV_VARS:i*T*UNIV_VARS+j*UNIV_VARS+8])
    for i in range(T-1):
        weights[1].append(weights_tot[T*DIM*UNIV_VARS + i*UNIV_VARS:T*DIM*UNIV_VARS + i*UNIV_VARS + 8])

    gen_update = up_sample(BATCH_SIZE, T, DIM, weights, type=dist_type, ftype=ftype, MINMAX=MINMAX)
    sample_plot = np.zeros([0, T, DIM])
    sample_plot_m = np.zeros([0, T, DIM])
    for i in range(10):
        sample = next(gen_update)
        sample_m = next(gen_marginals)
        sample_plot = np.append(sample_plot, sample, axis=0)
        sample_plot_m = np.append(sample_plot_m, sample_m, axis=0)

    sk = np.sum(sample_plot[:, T-1, :], axis=1) - STRIKE
    print(np.mean(np.maximum(sk, 0)))

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.hist(sample_plot[:, 0, 0])
    ax2.hist(sample_plot_m[:, 0, 0])
    plt.show()

    yn = input('Continue with updating? Empty enter to exit')
    if not yn:
        exit()
    for t in range(N+N_FINE+1, N+N_FINE+N+1):
        sample_marginals = next(gen_marginals)
        sample_ref = next(gen_ref)
        sample_update = next(gen_update)

        # I split initial reference measure and updating measure
        actual_sample = np.zeros([BATCH_SIZE, T, DIM])
        actual_sample[:int(BATCH_SIZE/2), :, :] = sample_ref[:int(BATCH_SIZE/2), :, :]
        actual_sample[int(BATCH_SIZE/2):, :, :] = sample_update[:int(BATCH_SIZE/2), :, :]

        (c, _) = sess.run([obj_fun, train_op], feed_dict={S_marg: sample_marginals, S_theta: actual_sample})
        vals.append(c)

        sample_marginals = next(gen_marginals)
        actual_sample[:int(BATCH_SIZE/2), :, :] = sample_ref[int(BATCH_SIZE/2):, :, :]
        actual_sample[int(BATCH_SIZE/2):, :, :] = sample_update[int(BATCH_SIZE/2):, :, :]
        (c, _) = sess.run([obj_fun, train_op], feed_dict={S_marg: sample_marginals, S_theta: actual_sample})
        vals.append(c)

        if t%500 == 0:
            print(t)
            print(np.mean(vals[t-1000:t]))
