import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import os
import json

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Create a cluster spec from environment variable TF_CONFIG
tf_config_json = os.environ.get("TF_CONFIG", "{}")
tf_config = json.loads(tf_config_json)
cluster_spec = tf_config.get("cluster", {})
cluster_spec_object = tf.train.ClusterSpec(cluster_spec)


#Create a server and start it for local task 
task = tf_config.get("task", {})
server = tf.train.Server(cluster_spec_object, job_name=task["type"],task_index=task["index"])


#Load MNIST data
print('***')
print('Starting to load data.')
mnist = input_data.read_data_sets("/workdir/data/")
print('Data loaded.')
print('***')

#We use between-graph replication so the same model graph is built across all workers but not ps
job_type = task.get("type", "").lower()
if job_type == "ps":
    print("Running PS code.")
    server.join()

elif job_type == "master":
    print("Running Master code.")
    server.join()

elif job_type == "worker":
    print("Running Worker code.")
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % task["index"],
        cluster=cluster_spec_object)):
        #Build a 3 layer DNN
        print('***')
        print('Starting to build the graph.')
        n_input = 28*28
        n_output = 10
        n_hidden1 = 300
        n_hidden2 = 100
        learning_rate = 0.01

        global_step = tf.train.get_or_create_global_step()


        X = tf.placeholder(tf.float32, shape=(None, n_input), name='X')
        y = tf.placeholder(tf.int64, shape=(None), name='y')


        hidden_layer1 = tf.contrib.layers.fully_connected(X, n_hidden1)
        hidden_layer2 = tf.contrib.layers.fully_connected(hidden_layer1, n_hidden2)
        logits = tf.contrib.layers.fully_connected(hidden_layer2, n_output, activation_fn=None)
    

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name='loss')
    
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss, global_step=global_step)
    

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
    print("n_batches is ", n_batches)
    total_steps = n_batches*n_epochs
    print("total_steps is ", total_steps)
    
    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=total_steps)]
    
    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,is_chief=(task["index"] == 0),checkpoint_dir="/workdir/log",
        hooks=hooks, config=tf.ConfigProto(log_device_placement=True)) as mon_sess:
        while not mon_sess.should_stop():
            for epoch in range(n_epochs):
                for iteration in range(n_batches):
                    X_batch, y_batch = mnist.train.next_batch(batch_size)
                    _,acc_train = mon_sess.run([training_op,accuracy], feed_dict={X:X_batch, y:y_batch})
                    #acc_test = accuracy.eval(session=mon_sess, feed_dict={X:mnist.test.images, y:mnist.test.labels})           
                #acc_train = accuracy.eval(session=mon_sess, feed_dict={X:X_batch, y:y_batch})
                #acc_test = accuracy.eval(session=mon_sess, feed_dict={X:mnist.test.images, y:mnist.test.labels})
                #print("Global Step: ", tf.train.global_step(mon_sess, global_step),"Epoch: ", epoch, "Train accuracy: ", acc_train, "Test accuracy: ", acc_test)
                print("Epoch: ", epoch, "Train accuracy: ", acc_train)
