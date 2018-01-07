import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import os
import argparse
import json
import logging
import json

now = datetime.now().strftime('%Y%m%d%H%M%S')

#Load MNIST data
print('***')
print('Starting to load data.')
mnist = input_data.read_data_sets("/workdir/data/")
print('Data loaded.')
print('***')

root_logdir = '/workdir/log/'
LOG_DIR = '{}/run-{}'.format(root_logdir, now)

#Create a cluster spec from environment variable TF_CONFIG
tf_config_json = os.environ.get("TF_CONFIG", "{}")
tf_config = json.loads(tf_config_json)
cluster_spec = tf_config.get("cluster", {})
cluster_spec_object = tf.train.ClusterSpec(cluster_spec)


#Create a server and start it for local task 
task = tf_config.get("task", {})
server_def = tf.train.ServerDef(
      cluster=cluster_spec_object.as_cluster_def(),
      protocol="grpc",
      job_name=task["type"],
      task_index=task["index"])
server = tf.train.Server(server_def)


#We use between-graph replication so the same model graph is built across all workers but not ps
job_type = task.get("type", "").lower()
if job_type == "ps":
    logging.info("Running PS code.")
    server.join()

elif job_type == "worker":
    logging.info("Running Worker code.")
    # The worker just blocks because we let the master assign all ops.
    server.join()

elif job_type == "master":
    logging.info("Running Master.")
    
    # By default, only Variable ops are placed on ps tasks, and the placement strategy is round-robin over all ps tasks.
    # Master assign ops to workers 
    for i in range(len(cluster_spec['worker'])):
        device_func = tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % i,
        cluster=server_def.cluster)
        with tf.device(device_func):
            print('***')
            print('Starting to build the graph.')
            n_input = 28*28
            n_output = 10
            n_hidden1 = 300
            n_hidden2 = 100
            learning_rate = 0.01



            X = tf.placeholder(tf.float32, shape=(None, n_input), name='X')
            y = tf.placeholder(tf.int64, shape=(None), name='y')



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

        #Training
        print('***')
        print('Starting to train model.')
        n_epochs = 50
        batch_size = 50
        n_batches = mnist.train.num_examples//batch_size
        total_steps = n_batches*n_epochs

        with tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(init)
            for epoch in range(n_epochs):
                for iteration in range(n_batches):
                    X_batch, y_batch = mnist.train.next_batch(batch_size)
                    sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
            
                acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
                acc_test = accuracy.eval(feed_dict={X:mnist.test.images, y:mnist.test.labels})
                print(epoch, "Train accuracy: ", acc_train, "Test accuracy: ", acc_test)

else:
    raise ValueError("invalid job_type %s" % (job_type,))
