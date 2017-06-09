import tensorflow as tf

# Flags for defining the tf.train.Server
from sklearn.datasets import load_iris

tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

print(FLAGS)

hiddenLayer = 7


def targetToY(target):
    items = []
    for singleTarget in target:
        if singleTarget == 0:
            items.append([0, 0, 1])
        elif singleTarget == 1:
            items.append([0, 1, 0])
        else:
            items.append([1, 0, 0])
    return items


def main(_):
    ps_hosts = ["localhost:2221"]
    workers = ["localhost:2222",
               "localhost:2223",
               "localhost:2224"]

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": workers})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        with tf.device(tf.train.replica_device_setter(worker_device="job:worker/task:%d/" % FLAGS.task_index,
                                                      cluster=cluster)):
            # Build model...
            x = tf.placeholder(tf.float32, [None, 4])
            w1 = tf.Variable(tf.random_uniform([4, hiddenLayer], -1, 1, seed=0))
            w2 = tf.Variable(tf.random_uniform([hiddenLayer, 3], -1, 1, seed=0))

            b1 = tf.Variable(tf.random_uniform([hiddenLayer], -1, 1, seed=0))
            b2 = tf.Variable(tf.random_uniform([3], -1, 1, seed=0))

            y1 = tf.nn.tanh(tf.matmul(x, w1) + b1)
            predicted = tf.nn.sigmoid(tf.matmul(y1, w2) + b2)
            correct_value = tf.placeholder(tf.float32, [None, 3], name="is_training")

            loss = tf.reduce_mean(tf.square(tf.subtract(correct_value, predicted)))
            global_step = tf.Variable(0)
            train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss, global_step=global_step)

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()

            init_op = tf.global_variables_initializer()

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/tmp/train_logs",
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)
    test = load_iris()
    x_in = test.data
    targets = targetToY(test.target)


    with sv.managed_session(master=server.target) as sess:
        step = 0
        while not sv.should_stop() and step < 10000:

            train_res, step, current_loss = sess.run([train_op, global_step, loss], feed_dict={x: x_in, correct_value: targets})
            # print(step, loss)
            print("Done step {} {}".format(step, current_loss))
        print("Test", sess.run(predicted, feed_dict={x: x_in}))
    # Ask for all the services to stop.
    sv.stop()


if __name__ == "__main__":
    tf.app.run()
