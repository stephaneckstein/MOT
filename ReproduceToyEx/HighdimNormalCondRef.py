import tensorflow as tf
import numpy as np
import time
from operator import add
import matplotlib.pyplot as plt

# This is a general implementation for two period, d-asset MMOT
d = 30
BATCH_SIZE = 2 ** 10
GAMMA = 200
N_TRAIN = 30000
morp = 1

STEPS_UPDATE = 10000
STEPS_AV_W = 1
n_updates = int(np.ceil(N_TRAIN/STEPS_UPDATE))

np.random.seed(0)
cov_maty = np.diag(np.random.random_sample(d)) * 2
downscale = 0.5 # affects how much the variance is scaled down from t=1 to t=0
scalemat = np.random.random_sample([d, d])*(1-downscale) + downscale
cov_matx = cov_maty * scalemat
mean = np.random.random_sample(d) * 2 - 1

print(mean)
np.random.seed(int(round(time.time())))  # don't want to fix randomization for Adam

K = sum(mean)
def cost_f(y):
    out = tf.nn.relu(tf.reduce_sum(y, axis=1) - K)
    return morp * out


def cost_f_rebuild(y):
    out = np.maximum(np.sum(y, axis=1) - K, 0)
    return morp * out


def gen_marginal(batch_size):
    # returns sampled marginals of S0, S1 as batch_size x d numpy arrays.
    while True:
        y = np.random.multivariate_normal(size=batch_size, cov=cov_maty, mean=mean)
        yield y


def gen_mu(batch_size):
    # returns two samples x, y as batch_size x d numpy arrays
    # can be same as gen_marginal
    while True:
        y = np.random.multivariate_normal(size=batch_size, cov=cov_maty, mean=mean)
        yield y


def univ_approx(x, name, hidden_dim=64):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        ua_w1 = tf.get_variable('ua_w1', shape=[1, hidden_dim],
                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        ua_b1 = tf.get_variable('ua_b1', shape=[hidden_dim], initializer=tf.contrib.layers.xavier_initializer(),
                               dtype=tf.float32)
        z = tf.matmul(x, ua_w1) + ua_b1
        # z = tf.layers.batch_normalization(z)
        a = tf.nn.relu(z)
        ua_w2 = tf.get_variable('ua_w2', shape=[hidden_dim, hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        ua_b2 = tf.get_variable('ua_b2', shape=[hidden_dim], initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
        z2 = tf.matmul(a, ua_w2) + ua_b2
        # z2 = tf.layers.batch_normalization(z2)
        a2 = tf.nn.relu(z2)
        ua_w3 = tf.get_variable('ua_w3', shape=[hidden_dim, hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        ua_b3 = tf.get_variable('ua_b3', shape=[hidden_dim], initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
        z3 = tf.matmul(a2, ua_w3) + ua_b3
        # z3 = tf.layers.batch_normalization(z3)
        a3 = tf.nn.relu(z3)
        ua_w4 = tf.get_variable('ua_w4', shape=[hidden_dim, 1],
                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        ua_b4 = tf.get_variable('ua_b4', shape=[hidden_dim], initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
        z = tf.matmul(a3, ua_w4) + ua_b4
    return tf.reduce_sum(z, 1)


def univ_convert(variablelist, names):
    mult = [[0 for i in range(4)] for lay in range(d)]
    const = [[0 for i in range(4)] for lay in range(d)]
    for idx, na in enumerate(names):
        if na[0] == 'o':
            lay = int(na[-3])-1
            i_d = na[1]
            lf = 2
            while na[lf] != '/':
                i_d = i_d + na[lf]
                lf += 1
            i_d = int(i_d)
            if na[-4] == 'w':
                mult[i_d][lay] = variablelist[idx]
            elif na[-4] == 'b':
                const[i_d][lay] = variablelist[idx]
    return mult, const


def rebuild_univ(x, multi, consti):
    for i in range(3):
        x = np.matmul(x, multi[i]) + consti[i]
        x = np.maximum(x, 0)
    x = np.matmul(x, multi[3]) + consti[3]
    return np.sum(x, axis=1)


def eval_density(x, mult, const):
    s_rebuild = 0
    for i in range(d):
        s_rebuild += rebuild_univ(x[:, i, np.newaxis], mult[i], const[i])
    landscape = cost_f_rebuild(x) - s_rebuild
    return 2 * GAMMA * np.maximum(landscape, 0), landscape  # just landscape for sampling


RADIUS = 0.025
BATCH2 = round(BATCH_SIZE/10)
CONVBATCH = 10
def gen_mu_cond(batch_size, mult, const):
    while True:
        ytotal = np.zeros([batch_size + BATCH2 * CONVBATCH, d])
        y = np.random.multivariate_normal(size=batch_size, cov=cov_maty, mean=mean)
        ytotal[:batch_size, :] = y
        conv = np.random.random_sample([BATCH2*CONVBATCH, d]) * 2 * RADIUS - RADIUS
        _, landscape = eval_density(y, mult, const)
        ind = np.argpartition(landscape, -BATCH2)[-BATCH2:]
        for i in range(BATCH2):
            conv[i * CONVBATCH: (i+1) * CONVBATCH, :] = conv[i * CONVBATCH: (i+1) * CONVBATCH, :] + y[ind[i], :]

        ytotal[:batch_size, :] = y
        ytotal[batch_size:, :] = conv
        yield ytotal
        # yield conv

        # _, landscape2 = eval_density(conv, mult, const)
        # ytotal[batch_size:, :] = conv
        # totalland = np.zeros(batch_size + BATCH2 * CONVBATCH)
        # totalland[:batch_size] = landscape
        # totalland[batch_size:] = landscape2
        # ind2 = np.argpartition(totalland, -batch_size)[-batch_size:]
        # out = ytotal[ind2]
        # yield out


n_steps = 2 ** 6
min_batch = 2 ** 4  # BATCH_SIZE = n_steps * min_batch
big_batch = 2 ** 10
def gen_mu_cond2(batch_size, mult, const):
    while True:
        ytotal = np.zeros([n_steps * min_batch, d])
        for j in range(n_steps):
            y = np.random.multivariate_normal(size=big_batch, cov=cov_maty, mean=mean)
            _, landscape = eval_density(y, mult, const)
            ind = np.argpartition(landscape, -min_batch)[-min_batch:]
            ytotal[j*min_batch:(j+1)*min_batch, :] = y[ind, :]
        yield ytotal


S1 = tf.placeholder(dtype=tf.float32, shape=[None, d])
mu1 = tf.placeholder(dtype=tf.float32, shape=[None, d])


sum_1 = 0
sum_mu1 = 0
for i in range(d):
    sum_1 += univ_approx(S1[:, i:i + 1], 'o'+str(i))
    sum_mu1 += univ_approx(mu1[:, i:i + 1], 'o'+str(i))
ints_1 = tf.reduce_mean(sum_1)


pen = GAMMA * tf.reduce_mean(tf.square(tf.nn.relu(cost_f(mu1) - sum_mu1)))
obj_fun = ints_1 + pen
train_op = tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.99, beta2=0.995).minimize(obj_fun)

weight_names = [v.name for v in tf.trainable_variables()]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gen_marg = gen_marginal(BATCH_SIZE)
    gen_pen = gen_mu(BATCH_SIZE)
    value_list = np.zeros(N_TRAIN)
    saved_weights = []

    for mt in range(n_updates):
        if mt > 0:
            gen_pen = gen_mu_cond2(BATCH_SIZE, mult, const)
        for t in range(STEPS_UPDATE):
            sam_s1 = next(gen_marg)
            sam_mu1 = next(gen_pen)
            (_, c) = sess.run([train_op, obj_fun], feed_dict={S1: sam_s1, mu1: sam_mu1})
            value_list[mt * STEPS_UPDATE + t] = c
            if t % 100 == 0 and (t > 0 or mt > 0):
                if mt > 0 and t == 100:
                    de, lc = eval_density(sam_mu1, mult, const)
                    print(de)
                    print(de.shape)
                    plt.hist(de)
                    plt.show()

                print(mt * STEPS_UPDATE + t)
                print(np.mean(value_list[mt * STEPS_UPDATE + t-100:mt * STEPS_UPDATE + t]))

        weight_val = sess.run(weight_names)

        # weight_val[:] = [w1x / (STEPS_AV_W + 1) for w1x in weight_val]
        # for t in range(STEPS_UPDATE-STEPS_AV_W, STEPS_UPDATE):
        #     sam_s1 = next(gen_marg)
        #     sam_mu1 = next(gen_pen)
        #     (_, c) = sess.run([train_op, obj_fun], feed_dict={S1: sam_s1, mu1: sam_mu1})
        #     value_list[mt * STEPS_UPDATE + t] = c
        #     w2_val = sess.run(weight_names)
        #     w2_val[:] = [w2x / (STEPS_AV_W + 1) for w2x in w2_val]
        #     weight_val[:] = map(add, weight_val, w2_val)
        #     if t % 100 == 0 and (t > 0 or mt > 0):
        #         print(mt *STEPS_UPDATE + t)
        #         print(np.mean(value_list[mt * STEPS_UPDATE + t-100:mt * STEPS_UPDATE + t]))
        mult, const = univ_convert(weight_val, weight_names)

    # print(weight_names)
    # print(weight_val)
    # print(weight_val[0].shape)
    # mult, const = univ_convert(weight_val, weight_names)
    # gen2 = gen_mu_cond(BATCH_SIZE, mult, const)
    # s2 = next(gen2)
    # print(s2)


    print('Final_value: ' + str(np.mean(value_list[N_TRAIN-2000:])))
