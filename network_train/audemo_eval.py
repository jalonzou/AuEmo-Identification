import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

import time
import math
import os.path
from datetime import datetime

import audemo

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir_eval', 'audemo_eval_log',
                           'Directory where to write event logs and checkpoint')

tf.app.flags.DEFINE_string('checkpoint_dir', 'audemo_log',
                           'Directory where to read model checkpoints.')

tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            'How often to run the eval.')

tf.app.flags.DEFINE_integer('num_examples', 2000,
                            'Number of examples to run.')

tf.app.flags.DEFINE_boolean('run_once', True,
                            'Where to run eval only once.')

NUM_EPOCHS = 100

def fill_feed_dict(drop_out, rate):
    feed_dict = {
        drop_out: rate
    }
    return feed_dict


def eval_once(saver, top_k_op, summary_op, total_loss, keep_prob):
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir_eval, graph=sess.graph)

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print ckpt.model_checkpoint_path
            # saver.restore(sess, ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
            # for path in ckpt.all_model_checkpoint_paths:
                # print(path)

            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
        true_count = 0
        total_sample_count = num_iter * FLAGS.batch_size

        step = 0
        feed_dict = fill_feed_dict(keep_prob, 1)

        while step < num_iter:
            predictions, loss = sess.run([top_k_op, total_loss], feed_dict=feed_dict)
            true_count += np.sum(predictions)
            print loss
            step += 1

        print true_count
        print total_sample_count
        # print aa
        precision = float(true_count) / float(total_sample_count)

        print('%s: precision = %.3f' % (datetime.now(), precision))

        # summary = tf.Summary()
        # # feed_dict = fill_feed_dict(keep_prob, 1)
        # summary.ParseFromString(sess.run(summary_op, feed_dict=feed_dict))
        # summary.value.add(tag='Precision @ 1', simple_value=precision)
        # summary_writer.add_summary(summary, global_step)


def evaluate():
    with tf.Graph().as_default():
        images, labels = audemo.train_input(num_epochs=NUM_EPOCHS, is_augment=True)
        # images, labels = audemo.eval_input(num_epochs=NUM_EPOCHS)
        keep_prob = tf.placeholder(tf.float32)

        logits = audemo.inference(inputs=images, keep_prob=keep_prob)
        total_loss = audemo.loss(logits, labels)

        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        variable_averages_calculator = tf.train.ExponentialMovingAverage(
            audemo.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages_calculator.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        # saver = tf.train.Saver(tf.global_variables())

        summary_op = tf.summary.merge_all()

        while True:
            eval_once(saver, top_k_op, summary_op, total_loss, keep_prob)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(_):
    if gfile.Exists(FLAGS.log_dir_eval):
        gfile.DeleteRecursively(FLAGS.log_dir_eval)
    gfile.MakeDirs(FLAGS.log_dir_eval)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
