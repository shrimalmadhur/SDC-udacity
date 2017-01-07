import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import math
import numpy as np


nb_classes = 43
# TODO: Load traffic signs data.
training_file = './train.p'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)

X_train, Y_train = train['features'], train['labels']

# TODO: Split data into training and validation sets.
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.33, stratify=Y_train)
print(X_train.shape)
print(Y_train.shape)
# TODO: Define placeholders and resize operation.
image_shape = X_train[0].shape
x = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], image_shape[2]])
y = tf.placeholder(tf.int64, shape=[None])
resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
weight = tf.Variable(tf.truncated_normal(shape, stddev=1e-3))
bias = tf.Variable(tf.constant(0.05, shape=[nb_classes]))
logits = tf.nn.xw_plus_b(fc7, weight, bias)
probs = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
beta = 1e-3
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y)) + beta*(tf.nn.l2_loss(weight))
opt = tf.train.AdamOptimizer(1e-3)
train_op = opt.minimize(loss_op, var_list=[weight, bias])
# init_op = tf.initialize_all_variables()
y_pred = tf.arg_max(logits, 1)

# TODO: Train and evaluate the feature extraction model.
EPOCH = 1
BATCH_SIZE = 128

def data_iterator(x, y):
    """ A simple data iterator """
    while True:
        # shuffle labels and features
        idxs = np.arange(0, len(x))
        np.random.shuffle(idxs)
        shuf_features = x[idxs]
        shuf_labels = y[idxs]
        batch_size = BATCH_SIZE
        for batch_idx in range(0, len(x), batch_size):
            images_batch = shuf_features[batch_idx:batch_idx+batch_size]
            labels_batch = shuf_labels[batch_idx:batch_idx+batch_size]
            yield images_batch, labels_batch

def accuracy(predictions, labels):
  	return (100.0 * np.sum(predictions == labels) / predictions.shape[0])

session = tf.Session()
session.run(tf.initialize_all_variables())

num_iterations_per_epoch = math.ceil(X_train.shape[0]/BATCH_SIZE)

iter_ = data_iterator(X_train, Y_train)

for i in range(EPOCH):

    for j in range(num_iterations_per_epoch):
        # get a batch of data
        x_batch, y_batch = next(iter_)

        # Run the optimizer on the batch
        session.run([train_op], feed_dict={x: x_batch, y: y_batch})

    train_cost, train_predictions = session.run([loss_op, y_pred], feed_dict={x: X_train, y: Y_train})
    print("train accuracy %.1f%% with loss %0.1f in epoch %d" % (float(accuracy(train_predictions, Y_train)), train_cost, (i+1)))
    
print('Validation accuracy: %.1f%%' % accuracy(valid_pred.eval(session=session), Y_val)) 