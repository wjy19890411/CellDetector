import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pdb

sns.set(color_codes=True)

tf.reset_default_graph()
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

caps1_num = 7
caps1_dim = 8
caps2_num = 3
caps2_dim = 6
seg_width = 255
seg_height = 255
origin_width = seg_width//2
origin_height = seg_height//2
batch_max = 10

def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)

def squash(s, axis=-2, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector

def tf_softmax(s, axis=-3, name=None):
    with tf.name_scope(name, default_name="softmax"):
        s_exp = tf.exp(s)
        s_exp_sum = tf.reduce_sum(s_exp, axis=axis, keep_dims=True)
        s_softmax = s_exp / s_exp_sum
        return s_softmax

#p.shape=[batc_size, 3*caps1_num, caps2_num, 1, caps2_dim], b.shape=[batch_size, 3*caps1_num, caps2_num, 1, 1]
def route(p, b):
    with tf.name_scope(name='route'):
        c = tf_softmax(b)
        s = tf.reduce_sum(tf.multiply(c, p), axis=-4, keep_dims=True) #s.shape=[batch_size, 1, caps2_num, 1, caps2_dim]
        v = squash(s) #v.shape=[batch_size, 1, caps2_num, 1, caps2_dim]
        v_tiled = tf.tile(v, [1, 3*caps1_num, 1, 1, 1])
        b = b + tf.matmul(p, v_tiled, transpose_b=True)
        return v, b

def get_feeder(i, j, X_feeder_raw, y_feeder_raw):
    return X_feeder_raw[i-seg_width//2 : i+seg_width//2+1, j-seg_height//2 : j+seg_height//2+1,
            :].reshape(1, seg_width, seg_height, 3), y_feeder_raw[i][j].reshape([1])


X = tf.placeholder(shape=[None, seg_width, seg_height, 3], dtype=tf.float32, name='input_area')
#input area with given size: seg_width*seg_height, and batch_size :None.

batch_size = tf.shape(X)[0]

weight_conv1 = []

for i in range(1, caps1_num+1):
    weight_conv1.append( tf.Variable(tf.random_normal(shape=[2**(i)+1, 2**(i)+1, 3, caps1_dim], mean=0., stddev=0.3),
        name='weight_'+str(i)+'_conv1') )

X_dim_extand = tf.expand_dims(X, -1)
X_tiled = tf.tile(X_dim_extand, [1, 1, 1, 1, caps1_dim], name='X_tiled')

weight_conv1_tiled = [tf.tile(tf.expand_dims(item, 0), [batch_size, 1, 1, 1, 1]) for item in weight_conv1]

output_caps1 = []

for i in range(caps1_num):
    output_caps1.append(tf.reduce_sum(tf.multiply(weight_conv1_tiled[i], X_tiled[:, origin_width-2**i: origin_width+2**i+1,
        origin_height-2**i: origin_height+2**i+1, :, :]), axis = [1, 2], keep_dims=True))

caps1_raw = tf.concat(output_caps1, -2)
caps1_raw_reshaped = squash(tf.reshape(caps1_raw, [-1, 3*caps1_num, 1, caps1_dim, 1], name='caps1_raw_reshaped'))
#caps1_raw_reshaped.shape = [batch_size, 3*caps1_num, 1, caps1_dim, 1]

caps1_tiled = tf.tile(caps1_raw_reshaped, [1, 1, caps2_num, 1, 1], name='caps1_tiled')

caps1_weight = tf.Variable(tf.random_normal(shape=[1, 3*caps1_num, caps2_num, caps1_dim, caps2_dim], mean = 0., stddev = 0.1), name='caps1_weight')
caps1_weight_tiled = tf.tile(caps1_weight, [batch_size, 1, 1, 1, 1], name='caps1_weight_tiled')

caps1_predict = tf.matmul(caps1_tiled, caps1_weight_tiled, transpose_a=True, name='caps1_predict')
caps2_raw = squash(caps1_predict, axis=-1)

weight_routing = tf.zeros((batch_size, 3*caps1_num, caps2_num, 1, 1), dtype=tf.float32)
for i in range(3):
    prediction, weight_routing = route(caps2_raw, weight_routing)

y = tf.placeholder(shape=[None], dtype=tf.int32, name='label')
y_onehot = tf.one_hot(y, caps2_num)
weight_label = tf.constant(np.array([0., 1., 4.]), dtype=tf.float32)
weight_label_softmax = tf_softmax(weight_label, axis=-1)
weight_label_dim_expanded = tf.expand_dims(weight_label_softmax, 0)
weight_label_tiled = tf.tile(weight_label_dim_expanded, [batch_size, 1])
#y_onehot = tf.multiply(weight_label_tiled, y_onehot)

prediction_squeezed = tf.squeeze(prediction, [1, 3], name='prediction_squeezed')
prediction_length = safe_norm(prediction_squeezed)

m_roof = 0.9
m_floor = 0.05
lam = 0.5

loss_right = tf.reduce_sum(tf.multiply(tf.multiply(tf.square( tf.maximum(0., m_roof-prediction_length) ), y_onehot),
    weight_label_tiled ), axis=-1, name='loss_right')
#loss_right.shape=[batch_size]

loss_miss = tf.reduce_sum(tf.multiply(tf.multiply(tf.square(tf.maximum(0., prediction_length-m_floor)), 1.-y_onehot),
    weight_label_tiled ), axis=-1, name='loss_miss')
#loss_miss.shape=[batch_size]

loss = tf.reduce_mean(loss_right + lam*loss_miss, name='name')

optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="training_op")

init = tf.global_variables_initializer()
saver = tf.train.Saver()

restore_checkpoint = True

checkpoint_path = "./saving_train/my_capsule_based_segmentation_with_weighted_loss"
best_loss_val = np.infty

with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()
    train_dir = os.listdir('stage1_train')
    loss_vals = []
    training_step = 0
    batch_box = 0
    for train_index in range(6, len(train_dir)):
        item = train_dir[train_index]
        X_feeder_raw = plt.imread('stage1_train/'+item+'/images/'+item+'.png')[:,:,:3]
        y_feeder_raw = np.load('stage1_train/'+item+'/'+item+'_with_boundary.npy')
        target = [seg_width//2+1, seg_height//2+1]
        X_feeders = []
        y_feeders = []
        for i in range(target[0], X_feeder_raw.shape[0]-target[0]-1):
            for j in range(target[1], X_feeder_raw.shape[1]-target[1]-1):
                if batch_box < batch_max:
                    batch_box = batch_box + 1
                    X_feeders.append(get_feeder(i,j, X_feeder_raw, y_feeder_raw)[0])
                    y_feeders.append(get_feeder(i,j, X_feeder_raw, y_feeder_raw)[1])
                else:
                    batch_box = 0
                    X_feeder = np.concatenate(X_feeders, axis=0)
                    y_feeder = np.concatenate(y_feeders, axis=0)
                    loss_val, _ = sess.run([loss, training_op], feed_dict={X: X_feeder, y: y_feeder})
                    loss_vals.append(loss_val)
                    if loss_val < best_loss_val or loss_val < 1e-7:
                        save_path = saver.save(sess, checkpoint_path)
                        best_loss_val = loss_val
                    training_step = training_step + 1
                    if training_step%100 == 0:
                        print('training ', train_index, ' th pic, and ',training_step,' th iteration loss is :', loss_val)
                    X_feeders = []
                    y_feeders = []



