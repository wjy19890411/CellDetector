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

# pic.shape = [width, height, 3]
def add_edge(pic):
    if len(pic.shape) == 3:
        pic_edged = np.zeros(shape=[pic.shape[0]+seg_width-1, pic.shape[1]+seg_height-1, pic.shape[2]], dtype=np.float32)
        pic_edged[seg_width//2:seg_width//2+pic.shape[0], seg_height//2: seg_height//2+pic.shape[1], :] = pic
        pic_edged[:, 0:seg_height//2, :] = pic_edged[:, seg_height-1:seg_height//2:-1, :]
        pic_edged[:, seg_height//2+pic.shape[1]:, :] = pic_edged[:, seg_height//2+pic.shape[1]-1:pic.shape[1]-1:-1, :]
        pic_edged[0:seg_width//2, :, :] = pic_edged[seg_width-1:seg_width//2:-1, :, :]
        pic_edged[seg_width//2+pic.shape[0]:, :, :] = pic_edged[seg_width//2+pic.shape[0]-1:pic.shape[0]-1:-1, :, :]
    if len(pic.shape) == 2:
        pic_edged = np.zeros(shape=[pic.shape[0]+seg_width-1, pic.shape[1]+seg_height-1], dtype=np.float32)
        pic_edged[seg_width//2:seg_width//2+pic.shape[0], seg_height//2: seg_height//2+pic.shape[1]] = pic
        pic_edged[:, 0:seg_height//2] = pic_edged[:, seg_height-1:seg_height//2:-1]
        pic_edged[:, seg_height//2+pic.shape[1]:] = pic_edged[:, seg_height//2+pic.shape[1]-1:pic.shape[1]-1:-1]
        pic_edged[0:seg_width//2, :] = pic_edged[seg_width-1:seg_width//2:-1, :]
        pic_edged[seg_width//2+pic.shape[0]:, :] = pic_edged[seg_width//2+pic.shape[0]-1:pic.shape[0]-1:-1, :]
    return pic_edged

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
        origin_height, :, :]), axis = [1, 2], keep_dims=True))

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

#y = tf.placeholder(shape=[None], dtype=tf.int32, name='label')
#y_onehot = tf.one_hot(y, caps2_num)
#weight_label = tf.constant(np.array([0., 1., 4.]), dtype=tf.float32)
#weight_label_softmax = tf_softmax(weight_label, axis=-1)
#weight_label_dim_expanded = tf.expand_dims(weight_label_softmax, 0)
#weight_label_tiled = tf.tile(weight_label_dim_expanded, [batch_size, 1])
#y_onehot = tf.multiply(weight_label_tiled, y_onehot)

prediction_squeezed = tf.squeeze(prediction, [1, 3], name='prediction_squeezed')
prediction_length = safe_norm(prediction_squeezed)

saver = tf.train.Saver()

restore_checkpoint = True

checkpoint_path = "./saving_train/my_capsule_based_segmentation_with_weighted_loss"

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    train_dir = os.listdir('stage1_test')
    item = train_dir[0]
    X_feeder_raw = plt.imread('stage1_test/'+item+'/images/'+item+'.png')[:,:,:3]
    #y_feeder_raw = np.load('stage1_train/'+item+'/'+item+'_with_boundary.npy')
    X_feeder_edged = add_edge(X_feeder_raw)
    target = [seg_width//2, seg_height//2]
    generating = np.zeros(shape=[X_feeder_raw.shape[0], X_feeder_raw.shape[1]], dtype=np.int32)
    for i in range(target[0], X_feeder_raw.shape[0] + target[0] ):
        for j in range(target[1], X_feeder_raw.shape[1] + target[1]):
            X_feeder = X_feeder_edged[i-seg_width//2 : i+seg_width//2+1, j-seg_height//2 : j+seg_height//2+1,
                    :].reshape(1, seg_width, seg_height, 3)
            prediction_length_val, = sess.run([prediction_length], feed_dict={X: X_feeder})
            generating[i-target[0]][j-target[1]] = prediction_length_val.argmax()

    np.save('test_prediction.npy', generating)
    plt.imshow(generating)
    plt.axis('off')
    plt.show()

