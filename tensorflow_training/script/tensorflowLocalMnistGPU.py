import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime

now = datetime.now().strftime('%Y%m%d%H%M%S')

#root_logdir = 'tf_logs'
#logdir = '{}/run-{}'.format(root_logdir, now)

#Build a 3 layer DNN
print('***')
print('Starting to build the graph.')
n_input = 28*28
n_output = 10
n_hidden1 = 300
n_hidden2 = 100
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n_input), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

with tf.device('/device:GPU:0'):
    with tf.name_scope('dnn'):
        hidden_layer1 = tf.contrib.layers.fully_connected(X, n_hidden1)
        hidden_layer2 = tf.contrib.layers.fully_connected(hidden_layer1, n_hidden2)
        logits = tf.contrib.layers.fully_connected(hidden_layer2, n_output, activation_fn=None)
    
    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name='loss')
    
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
    
    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()

print('Graph built.')
print('***')

#accuracy_summary = tf.summary.scalar('Accuracy', accuracy)
#file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

#Load MNIST data
print('***')
print('Starting to load data.')
#mnist = input_data.read_data_sets("s3://TESTING/data/")
mnist = input_data.read_data_sets("/workdir/data/")

print('Data loaded.')
print('***')

#Training
print('***')
print('Starting to train model.')
n_epochs = 50
batch_size = 50
n_batches = mnist.train.num_examples//batch_size

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            step = epoch * n_batches + iteration
            #acc_train = accuracy_summary.eval(feed_dict={X:X_batch, y:y_batch})
            #file_writer.add_summary(acc_train, step) 
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
            
        acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
        acc_test = accuracy.eval(feed_dict={X:mnist.test.images, y:mnist.test.labels})
        print(epoch, "Train accuracy: ", acc_train, "Test accuracy: ", acc_test)

    print('Saving trained model.')   
    #save_path = saver.save(sess, "s3://TESTING/model/my_model_final.ckpt")
    save_path = saver.save(sess, "/workdir/output/my_model_final.ckpt")