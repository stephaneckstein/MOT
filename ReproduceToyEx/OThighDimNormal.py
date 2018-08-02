import tensorflow as tf
import numpy as np
import time


# This is a general implementation for two period, d-asset MMOT
d = 5
BATCH_SIZE = 2 ** 10
GAMMA = 1000
N_TRAIN = 40000
morp = -1
K = 0


def cost_f(y):
    out = tf.nn.relu(tf.reduce_sum(y, axis=1)/d - K)
    return morp * out

np.random.seed(0)
cov_maty = np.diag(np.random.random_sample(d)) * 2
downscale = 0.5 # how much the variance is scaled down from t=1 to t=0
scalemat = np.random.random_sample([d, d])*(1-downscale) + downscale
cov_matx = cov_maty * scalemat
mean = np.random.random_sample(d) * 2 - 1

print(mean)
np.random.seed(int(round(time.time())))

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
        ua_w = tf.get_variable('ua_w', shape=[1, hidden_dim],
                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        ua_b = tf.get_variable('ua_b', shape=[hidden_dim], initializer=tf.contrib.layers.xavier_initializer(),
                               dtype=tf.float32)
        z = tf.matmul(x, ua_w) + ua_b
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
        ua_v = tf.get_variable('ua_v', shape=[hidden_dim, 1],
                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        z = tf.matmul(a3, ua_v)
    return tf.reduce_sum(z, 1)


S1 = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, d])
mu1 = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, d])


sum_1 = 0
sum_mu1 = 0
for i in range(d):
    sum_1 += univ_approx(S1[:, i:i + 1], 'o'+str(i))
    sum_mu1 += univ_approx(mu1[:, i:i + 1], 'o'+str(i))
ints_1 = tf.reduce_mean(sum_1)


pen = GAMMA * tf.reduce_mean(tf.square(tf.nn.relu(cost_f(mu1) - sum_mu1)))
obj_fun = ints_1 + pen
train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.99, beta2=0.995).minimize(obj_fun)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gen_marg = gen_marginal(BATCH_SIZE)
    gen_pen = gen_mu(BATCH_SIZE)

    value_list = np.zeros(N_TRAIN)
    for t in range(N_TRAIN):
        sam_s1 = next(gen_marg)
        sam_mu1 = next(gen_pen)
        (_, c) = sess.run([train_op, obj_fun], feed_dict={S1: sam_s1, mu1: sam_mu1})
        value_list[t] = c
        if t % 2000 == 0 and t > 0:
            print(t)
            print(np.mean(value_list[t-2000:t]))

    print('Final_value: ' + str(np.mean(value_list[N_TRAIN-5000:])))
