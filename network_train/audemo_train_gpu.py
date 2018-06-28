import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

import time
import os.path
from datetime import datetime

import audemo

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'audemo_train',
                           'Directory where to write event logs')

tf.app.flags.DEFINE_string('log_dir', 'audemo_log',
                           'Directory where to write event logs and checkpoint')

tf.app.flags.DEFINE_integer('max_steps', 300,
                            'Number of batches to run.')

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')

tf.app.flags.DEFINE_integer('log_frequency', 10,
                            'How often to log results to the console.')

NUM_EPOCHS = 1000

def fill_feed_dict(drop_out, rate):
    feed_dict = {
        drop_out: rate
    }
    return feed_dict


def run_train():
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # Get images and labels from dataset.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            images, labels = audemo.distorted_input(num_epochs=NUM_EPOCHS)
            keep_prob = tf.placeholder(tf.float32)

        logits = audemo.inference(images, keep_prob)
        total_loss = audemo.loss(logits, labels)
        train_op = audemo.train(total_loss, global_step)

        saver = tf.train.Saver(tf.global_variables())

        summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        with tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as sess:
            sess.run(init)

            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, graph=sess.graph)

            # Set keeping rate for dropout
            feed_dict = fill_feed_dict(keep_prob, 0.7)

            for step in xrange(FLAGS.max_steps):
                start_time = time.time()
                _, loss_value = sess.run([train_op, total_loss], feed_dict=feed_dict)
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

                if step % 20 == 0:
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)

                if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)


def main(_):
    if gfile.Exists(FLAGS.log_dir):
        gfile.DeleteRecursively(FLAGS.log_dir)
    gfile.MakeDirs(FLAGS.log_dir)
    run_train()


if __name__ == '__main__':
    tf.app.run()