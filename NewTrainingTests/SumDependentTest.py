import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


DIM = 10
BATCH_SIZE = 2 ** 7
BSAVE = 2 ** 2
GAMMA = 200
N = 5000
N_FINE = 5000
steps = 10
CONV_MULT = 0.001 * np.arange(1, steps)[::-1]  # multiplier for the convolution


def gen_points(batch_size):
    while True:
        dataset = np.random.random_sample([batch_size, DIM])
        yield(dataset)


def gen_points_pareto(batch_size, par=2.3):
    while True:
        dataset = np.random.pareto(par, [batch_size, DIM])
        yield(dataset)


TH = DIM * (7/8)
LG = 1
def g(p):
    sgf = tf.reduce_sum(p, 1)
    cond = tf.greater(LG*sgf, LG*TH)
    return tf.cast(cond, dtype=tf.float32)


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


S_marg = tf.placeholder(dtype=tf.float32, shape=[None, DIM])
S_theta = tf.placeholder(dtype=tf.float32, shape=[None, DIM])

s0 = 0
for i in range(DIM):
    s0 += tf.reduce_mean(univ_approx(S_marg[:, i:i+1], str(i)), axis=1)
ints = tf.reduce_mean(s0)

s1 = 0
for i in range(DIM):
    s1 += tf.reduce_mean(univ_approx(S_theta[:, i:i+1], str(i)), axis=1)

fvar = g(S_theta)
diff = fvar - s1
obj_fun = ints + GAMMA * tf.reduce_mean(tf.square(tf.nn.relu(diff)))

train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.99, beta2=0.995).minimize(obj_fun)
global_step = tf.Variable(0, trainable=False)
train_op_fine = tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(0.0001, global_step, 5, 0.99, staircase=False), beta1=0.99, beta2=0.995).minimize(obj_fun)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    gen_marg = gen_points(BATCH_SIZE)
    vals = []
    good_samples = np.zeros([(N+N_FINE)*BSAVE, DIM])

    for t in range(1, N+1):
        sample = next(gen_marg)
        (c, diffv, _) = sess.run([obj_fun, diff, train_op], feed_dict={S_marg: sample, S_theta: sample})
        vals.append(c)
        ind = np.argpartition(diffv, -BSAVE)[-BSAVE:]
        good_samples[(t-1)*BSAVE:t*BSAVE, :] = sample[ind, :]
        if t % 1000 == 0:
            print(t)
            print(np.mean(vals[t-1000:t]))
            print(sample[ind, :])
    for t in range(N+1, N+1+N_FINE):
        sample = next(gen_marg)
        (c, diffv, _) = sess.run([obj_fun, diff, train_op_fine], feed_dict={S_marg: sample, S_theta: sample, global_step: t+1-N})
        vals.append(c)
        ind = np.argpartition(diffv, -BSAVE)[-BSAVE:]
        good_samples[(t-1)*BSAVE:t*BSAVE, :] = sample[ind, :]
        if t % 1000 == 0:
            print(t)
            print(np.mean(vals[t-1000:t]))
            print(sample[ind, :])

    old_good_samples = np.copy(good_samples)
    for s in range(steps-1):
        plt.scatter(good_samples[:, 0], good_samples[:, 1])
        plt.show()
        vals = []
        size_samples = len(good_samples)
        def new_gen():
            while True:
                dataset = np.zeros([BATCH_SIZE*3, DIM])
                dataset[:BATCH_SIZE, :] = next(gen_marg)
                dataset[BATCH_SIZE:2*BATCH_SIZE, :] = good_samples[np.random.choice(size_samples, BATCH_SIZE), :] + (np.random.random_sample([BATCH_SIZE, DIM])-0.5) * CONV_MULT[s]
                dataset[2*BATCH_SIZE:3*BATCH_SIZE, :] = good_samples[np.random.choice(size_samples, BATCH_SIZE), :]
                yield(dataset)


        new_good_samples = np.zeros([(N+N_FINE) * BSAVE, DIM])
        gen_theta = new_gen()

        for t in range(1, N+1):
            sample_marg = next(gen_marg)
            sample_theta = next(gen_theta)
            (c, diffv, _) = sess.run([obj_fun, diff, train_op], feed_dict={S_marg: sample_marg, S_theta: sample_theta})
            vals.append(c)
            ind = np.argpartition(diffv, -BSAVE)[-BSAVE:]
            new_good_samples[(t - 1) * BSAVE:t * BSAVE, :] = sample_theta[ind, :]
            if t % 1000 == 0:
                print(t)
                print(np.mean(vals[t - 1000:t]))
                print(sample_theta[ind, :])
        for t in range(N + 1, N + 1 + N_FINE):
            sample_marg = next(gen_marg)
            sample_theta = next(gen_theta)
            (c, diffv, _) = sess.run([obj_fun, diff, train_op_fine], feed_dict={S_marg: sample_marg, S_theta: sample_theta, global_step: t+1-N})
            vals.append(c)
            ind = np.argpartition(diffv, -BSAVE)[-BSAVE:]
            new_good_samples[(t - 1) * BSAVE:t * BSAVE, :] = sample_theta[ind, :]
            if t % 1000 == 0:
                print(t)
                print(np.mean(vals[t - 1000:t]))
                print(sample_theta[ind, :])


        # good_samples = np.append(good_samples, new_good_samples, axis=0)
        good_samples = np.append(old_good_samples, new_good_samples, axis=0)
        old_good_samples = np.copy(new_good_samples)