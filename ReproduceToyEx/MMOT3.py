import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# This is a general implementation for two period, d-asset MMOT
d = 2
BATCH_SIZE = 2 ** 11
GAMMA = 3000
N_TRAIN = 50000
P_COST = 2
morp = -1

print(P_COST)
print(morp)
print(GAMMA)

starttime = time.time()

def cost_f(y):
    # spread option for d = 2:
    out = tf.pow(tf.abs(y[:, 0] - y[:, 1]), P_COST)

    # K = 1
    # out = tf.nn.relu(y[:, 0] + y[:, 1] - K)

    return morp * out


def gen_marginal(batch_size):
    # returns sampled marginals of S0, S1 as batch_size x d numpy arrays.
    while True:
        x = np.random.random_sample([batch_size, 2])
        x = x * [2, 2] - [1, 1]

        y = np.random.random_sample([batch_size, 2])
        y = y * [6, 2] - [3, 1]

        z = np.random.random_sample([batch_size, 2])
        z = z * [8, 8] - [4, 4]

        yield x, y, z


def gen_mu(batch_size):
    # returns two samples x, y as batch_size x d numpy arrays
    # can be same as gen_marginal
    while True:
        x = np.random.random_sample([batch_size, 2])
        x = x * [2, 2] - [1, 1]

        y = np.random.random_sample([batch_size, 2])
        y = y * [6, 2] - [3, 1]

        z = np.random.random_sample([batch_size, 2])
        z = z * [8, 8] - [4, 4]

        yield x, y, z


def univ_approx(x, name, hidden_dim=32):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        ua_w = tf.get_variable('ua_w', shape=[1, hidden_dim],
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
        ua_v = tf.get_variable('ua_v', shape=[hidden_dim, 1],
                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        z = tf.matmul(a3, ua_v)
    return tf.reduce_sum(z, 1)


def multi_to_one_approx(x, name, hidden_dim=64):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        ua_w = tf.get_variable('ua_w', shape=[d, hidden_dim],
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
        ua_v = tf.get_variable('ua_v', shape=[hidden_dim, 1],
                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        z = tf.matmul(a3, ua_v)
    return tf.reduce_sum(z, 1)


def multi2_to_one_approx(x1, x2, name, hidden_dim=64):
    x = tf.concat([x1, x2], axis=1)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        ua_w = tf.get_variable('ua_w', shape=[2 * d, hidden_dim],
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
        ua_v = tf.get_variable('ua_v', shape=[hidden_dim, 1],
                               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        z = tf.matmul(a3, ua_v)
    return tf.reduce_sum(z, 1)


S0 = tf.placeholder(dtype=tf.float32, shape=[None, d])
S1 = tf.placeholder(dtype=tf.float32, shape=[None, d])
S2 = tf.placeholder(dtype=tf.float32, shape=[None, d])
mu0 = tf.placeholder(dtype=tf.float32, shape=[None, d])
mu1 = tf.placeholder(dtype=tf.float32, shape=[None, d])
mu2 = tf.placeholder(dtype=tf.float32, shape=[None, d])

sum_0 = 0
sum_mu0 = 0
for i in range(d):
    sum_0 += univ_approx(S0[:, i:i + 1], 'z'+str(i))
    sum_mu0 += univ_approx(mu0[:, i:i + 1], 'z'+str(i))
ints_0 = tf.reduce_mean(sum_0)

sum_1 = 0
sum_mu1 = 0
for i in range(d):
    sum_1 += univ_approx(S1[:, i:i + 1], 'o'+str(i))
    sum_mu1 += univ_approx(mu1[:, i:i + 1], 'o'+str(i))
ints_1 = tf.reduce_mean(sum_1)

sum_2 = 0
sum_mu2 = 0
for i in range(d):
    sum_2 += univ_approx(S2[:, i:i + 1], 't'+str(i))
    sum_mu2 += univ_approx(mu2[:, i:i + 1], 't'+str(i))
ints_2 = tf.reduce_mean(sum_2)

sum_mart_mu_01 = 0
for i in range(d):
    sum_mart_mu_01 += multi_to_one_approx(mu0, 'm'+str(i)) * tf.reduce_sum((mu1[:, i:i+1] - mu0[:, i:i+1]), 1)

sum_mart_mu_02 = 0
for i in range(d):
    sum_mart_mu_02 += multi2_to_one_approx(mu0, mu1, 'm2'+str(i)) * tf.reduce_sum((mu2[:, i:i+1] - mu1[:, i:i+1]), 1)


pen = GAMMA * tf.reduce_mean(tf.square(tf.nn.relu(cost_f(mu2) - sum_mu0 - sum_mu1 - sum_mu2 - sum_mart_mu_01 - sum_mart_mu_02)))
obj_fun = ints_0 + ints_1 + ints_2 + pen
train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.99, beta2=0.995).minimize(obj_fun)

### Objects to plot
phi_0 = univ_approx(mu1[:, 0:1], 'o'+str(0))
phi_1 = univ_approx(mu1[:, 1:2], 'o'+str(1))
psi_0 = univ_approx(mu0[:, 0:1], 'z'+str(0))
psi_1 = univ_approx(mu0[:, 1:2], 'z'+str(1))
h0 = multi_to_one_approx(mu0, 'm'+str(0))
h1 = multi_to_one_approx(mu0, 'm'+str(1))
den = 2 * GAMMA * tf.nn.relu(tf.nn.relu(cost_f(mu2) - sum_mu0 - sum_mu1 - sum_mu2 - sum_mart_mu_01 - sum_mart_mu_02))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gen_marg = gen_marginal(BATCH_SIZE)
    gen_pen = gen_mu(BATCH_SIZE)

    value_list = np.zeros(N_TRAIN)
    for t in range(N_TRAIN):
        sam_s0, sam_s1, sam_s2 = next(gen_marg)
        sam_mu0, sam_mu1, sam_mu2 = next(gen_pen)
        (_, c) = sess.run([train_op, obj_fun], feed_dict={S0: sam_s0, S1: sam_s1, S2: sam_s2, mu0: sam_mu0, mu1: sam_mu1, mu2: sam_mu2})
        # thv = sess.run(th, feed_dict={S0: sam_s0, S1: sam_s1, mu0: sam_mu0, mu1: sam_mu1})
        # print(thv)
        value_list[t] = c
        if t % 2000 == 0 and t > 0:
            print(t)
            print(np.mean(value_list[t-2000:t]))

    print('Final_value: ' + str(np.mean(value_list[N_TRAIN-5000:])))
    print(time.time() - starttime)
    exit()

    # sampling from optimal measure
    sample_opt = 1000
    batch_up = 2 ** 14
    y1y2 = np.zeros([sample_opt, d])
    x1x2 = np.zeros([sample_opt, d])
    x1y1 = np.zeros([sample_opt, d])
    x1y2 = np.zeros([sample_opt, d])
    x2y1 = np.zeros([sample_opt, d])
    x2y2 = np.zeros([sample_opt, d])
    gen_pen = gen_mu(batch_up)
    s_ind = 0
    while s_ind < sample_opt:
        sam_mu0, sam_mu1, sam_mu2 = next(gen_pen)
        denv = sess.run(den, feed_dict={mu0: sam_mu0, mu1: sam_mu1})
        d_max = np.max(denv)
        # d_max = np.quantile(denv, 0.99)  # make it quicker
        u_den = np.random.random_sample(batch_up)
        for i_den in range(batch_up):
            if denv[i_den] >= d_max * u_den[i_den]:
                y1y2[s_ind] = sam_mu1[i_den, :]
                x1x2[s_ind] = sam_mu0[i_den, :]
                x1y1[s_ind, 0] = sam_mu0[i_den, 0]
                x1y1[s_ind, 1] = sam_mu1[i_den, 0]

                x1y2[s_ind, 0] = sam_mu0[i_den, 0]
                x1y2[s_ind, 1] = sam_mu1[i_den, 1]

                x2y1[s_ind, 0] = sam_mu0[i_den, 1]
                x2y1[s_ind, 1] = sam_mu1[i_den, 0]

                x2y2[s_ind, 0] = sam_mu0[i_den, 1]
                x2y2[s_ind, 1] = sam_mu1[i_den, 1]

                s_ind +=1
                if s_ind == sample_opt:
                    break

    ssx = 100
    sseach = 20
    sschoice = 2 ** 10
    condsampx = np.zeros([ssx*sseach, d])
    condsampy = np.zeros([ssx*sseach, d])
    gen_pen = gen_mu(sschoice)
    for i_s in range(ssx):
        sam_mu0 = np.zeros([sschoice, d]) + x1x2[i_s, :]
        condsampx[i_s * sseach:(i_s+1) * sseach, :] += x1x2[i_s, :]
        _, sam_mu1 = next(gen_pen)
        denv = sess.run(den, feed_dict={mu0: sam_mu0, mu1: sam_mu1})
        ind_s = np.argpartition(denv, -sseach)[-sseach:]
        condsampy[i_s * sseach:(i_s+1) * sseach, :] = sam_mu1[ind_s, :]
    while 1:
        numpoints = int(input('Num points?'))
        if numpoints == 0:
            break
        inds_s = np.random.choice(ssx, numpoints)
        for i_s in range(numpoints):
            plt.scatter(condsampx[sseach * inds_s[i_s], 0], condsampx[sseach * inds_s[i_s], 1], color='red')
            for j_s in range(sseach):
                plt.scatter(condsampy[sseach * inds_s[i_s] + j_s, 0], condsampy[sseach * inds_s[i_s] + j_s, 1])
                plt.arrow(x=condsampx[sseach * inds_s[i_s], 0], y=condsampx[sseach * inds_s[i_s], 1], dx=condsampy[sseach * inds_s[i_s] + j_s, 0] - condsampx[sseach * inds_s[i_s], 0], dy=condsampy[sseach * inds_s[i_s] + j_s, 1] - condsampx[sseach * inds_s[i_s], 1],
                          width=0.0001)
        plt.show()


    plt.scatter(y1y2[:, 0], y1y2[:, 1], s=0.5)
    name = input('Name for plot?')
    savename = str(d)+'_'+str(BATCH_SIZE)+'_'+str(GAMMA)+'_'+str(N_TRAIN)+'_'+str(P_COST)+'_'+str(morp)+name+'.pdf'
    plt.savefig(savename, format='pdf', dpi=400)
    plt.title('y1y2')
    plt.show()

    plt.scatter(x1x2[:, 0], x1x2[:, 1], s=0.5)
    savename = str(d)+'_'+str(BATCH_SIZE)+'_'+str(GAMMA)+'_'+str(N_TRAIN)+'_'+str(P_COST)+'_'+str(morp)+'Xvar'+name+'.pdf'
    plt.savefig(savename, format='pdf', dpi=400)
    plt.title('x1x2')
    plt.show()

    while 1:
        numpoints = int(input('number points?'))
        if numpoints == 0:
            break
        plt.scatter(y1y2[:numpoints, 0], y1y2[:numpoints, 1], s=0.5)
        plt.scatter(x1x2[:numpoints, 0], x1x2[:numpoints, 1], s=0.5, color='red')
        for i in range(numpoints):
            plt.arrow(x=x1x2[i, 0], y=x1x2[i, 1], dx=y1y2[i, 0]-x1x2[i, 0], dy=y1y2[i, 1]-x1x2[i, 1], width=0.0001)
        plt.show()

    plt.scatter(x1y1[:, 0], x1y1[:, 1], s=0.5)
    savename = str(d)+'_'+str(BATCH_SIZE)+'_'+str(GAMMA)+'_'+str(N_TRAIN)+'_'+str(P_COST)+'_'+str(morp)+'x1y1'+name+'.pdf'
    plt.savefig(savename, format='pdf', dpi=400)
    plt.title('x1y1')
    plt.show()

    plt.scatter(x1y2[:, 0], x1y2[:, 1], s=0.5)
    savename = str(d)+'_'+str(BATCH_SIZE)+'_'+str(GAMMA)+'_'+str(N_TRAIN)+'_'+str(P_COST)+'_'+str(morp)+'x1y2'+name+'.pdf'
    plt.savefig(savename, format='pdf', dpi=400)
    plt.title('x1y2')
    plt.show()

    plt.scatter(x2y1[:, 0], x2y1[:, 1], s=0.5)
    savename = str(d)+'_'+str(BATCH_SIZE)+'_'+str(GAMMA)+'_'+str(N_TRAIN)+'_'+str(P_COST)+'_'+str(morp)+'x2y1'+name+'.pdf'
    plt.savefig(savename, format='pdf', dpi=400)
    plt.title('x2y1')
    plt.show()

    plt.scatter(x2y2[:, 0], x2y2[:, 1], s=0.5)
    savename = str(d)+'_'+str(BATCH_SIZE)+'_'+str(GAMMA)+'_'+str(N_TRAIN)+'_'+str(P_COST)+'_'+str(morp)+'x2y2'+name+'.pdf'
    plt.savefig(savename, format='pdf', dpi=400)
    plt.title('x2y2')
    plt.show()

    # Plot optimal functions (dual optimizers)
    g_size = 500
    sam_mu1 = np.zeros(shape=[g_size, 2])
    sam_mu0 = np.zeros(shape=[g_size, 2])
    for i in range(g_size):
        sam_mu1[i, 0] = -3 + 6 * i/(g_size-1)
        sam_mu1[i, 1] = -2 + 4 * i/(g_size-1)
        sam_mu0[i, 0] = -1 + 2 * i/(g_size-1)
        sam_mu0[i, 1] = -1 + 2 * i/(g_size-1)
    (p0, p1, ph0, ph1) = sess.run([psi_0, psi_1, phi_0, phi_1], feed_dict={mu0: sam_mu0, mu1: sam_mu1})

    plt.plot(sam_mu0[:, 0], p0)
    plt.show()
    plt.plot(sam_mu0[:, 1], p1)
    plt.show()

    plt.plot(sam_mu1[:, 0], ph0)
    plt.show()
    plt.plot(sam_mu1[:, 1], ph1)
    plt.show()


    g_size = 50
    sam_mu0 = np.zeros(shape=[g_size ** 2, 2])
    for i in range(g_size):
        for j in range(g_size):
            sam_mu0[i * g_size + j, 0] = -1 + 2 * i/(g_size-1)
            sam_mu0[i * g_size + j, 1] = -1 + 2 * j/(g_size-1)

    (h0v, h1v) = sess.run([h0, h1], feed_dict={mu0: sam_mu0})

    print([sam_mu0[0, 0], sam_mu0[0, 1], h0v[0]])
    print([sam_mu0[g_size * (g_size - 1), 0], sam_mu0[g_size * (g_size - 1), 1], h0v[g_size * (g_size - 1)]])
    print([sam_mu0[(g_size - 1), 0], sam_mu0[(g_size - 1), 1], h0v[(g_size - 1)]])
    print([sam_mu0[g_size * (g_size - 1) + (g_size - 1), 0], sam_mu0[g_size * (g_size - 1) + (g_size - 1), 1], h0v[g_size * (g_size - 1) + (g_size - 1)]])
    print([sam_mu0[0, 0], sam_mu0[0, 1], h1v[0]])
    print([sam_mu0[g_size * (g_size - 1), 0], sam_mu0[g_size * (g_size - 1), 1], h1v[g_size * (g_size - 1)]])
    print([sam_mu0[(g_size - 1), 0], sam_mu0[(g_size - 1), 1], h1v[(g_size - 1)]])
    print([sam_mu0[g_size * (g_size - 1) + (g_size - 1), 0], sam_mu0[g_size * (g_size - 1) + (g_size - 1), 1], h1v[g_size * (g_size - 1) + (g_size - 1)]])


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(sam_mu0[:, 0], sam_mu0[:, 1], h0v)
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(sam_mu0[:, 0], sam_mu0[:, 1], h1v)
    plt.show()


    # plot optimal density (primal optimizer)
    g_size = 25
    sam_mu1 = np.zeros(shape=[g_size, 2])
    sam_mu0 = np.zeros(shape=[g_size, 2])
    for i in range(g_size):
        sam_mu1[i, 0] = -3 + 6 * i/(g_size-1)
        sam_mu1[i, 1] = -2 + 4 * i/(g_size-1)
        sam_mu0[i, 0] = -1 + 2 * i/(g_size-1)
        sam_mu0[i, 1] = -1 + 2 * i/(g_size-1)
    (p0, p1, ph0, ph1) = sess.run([psi_0, psi_1, phi_0, phi_1], feed_dict={mu0: sam_mu0, mu1: sam_mu1})

    sam_mu0_big = np.zeros(shape=[g_size ** 2, 2])
    for i in range(g_size):
        for j in range(g_size):
            sam_mu0_big[i * g_size + j, 0] = -1 + 2 * i/(g_size-1)
            sam_mu0_big[i * g_size + j, 1] = -1 + 2 * j/(g_size-1)
    (h0v, h1v) = sess.run([h0, h1], feed_dict={mu0: sam_mu0_big})

    const = 1 / (g_size ** 2)
    z = np.zeros(g_size ** 2)
    sam_mu1_big = np.zeros(shape=[g_size ** 2, 2])
    for i in range(g_size):
        y0 = -3 + 6 * i / (g_size - 1)
        for j in range(g_size):
            y1 = -2 + 4 * j / (g_size - 1)
            sam_mu1_big[i * g_size + j, 0] = -3 + 6 * i/(g_size-1)
            sam_mu1_big[i * g_size + j, 1] = -2 + 4 * j/(g_size-1)
            for k in range(g_size):
                x0 = -1 + 2 * k / (g_size - 1)
                for l in range(g_size):
                    x1 = -1 + 2 * l / (g_size - 1)
                    z[i * g_size + j] += np.maximum(morp * np.float_power(np.abs(y1 - y0), P_COST) - h0v[k*g_size + l] * (y0 - x0) - h1v[k*g_size + l] * (y1 - x1) - p0[k] - p1[l] - ph0[i] - ph1[j], 0)
    z = z * const  # * 2 * GAMMA
    print(z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(sam_mu1_big[:, 0], sam_mu1_big[:, 1], z)
    plt.show()

