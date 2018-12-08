# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Build and train mobilenet_v1 with options for quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from deployment import model_deploy
slim = tf.contrib.slim

flags = tf.app.flags

flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('task', 0, 'Task')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('num_classes', 1001, 'Number of classes to distinguish')
flags.DEFINE_integer('number_of_steps', None,
                     'Number of training steps to perform before stopping')
flags.DEFINE_integer('eval_image_size', 224, 'Input image resolution')
flags.DEFINE_float('Temperature', 1.0, 'Temperature for distillation')
#flags.DEFINE_float('depth_multiplier', 1.0, 'Depth multiplier for mobilenet')
flags.DEFINE_float('label_smoothing', 0.0, 'Label smoothing in softmax')

flags.DEFINE_bool('quantize', False, 'Quantize training')
flags.DEFINE_string('fine_tune_checkpoint', '',
                    'Checkpoint from which to start finetuning.')
flags.DEFINE_string('checkpoint_dir', '',
                    'Directory for writing training checkpoints and logs')
flags.DEFINE_string('dataset_dir', '', 'Location of dataset')
flags.DEFINE_integer('log_every_n_steps', 100, 'Number of steps per log')
flags.DEFINE_integer('save_summaries_secs', 100,
                     'How often to save summaries, secs')
flags.DEFINE_integer('save_interval_secs', 100,
                     'How often to save checkpoints, secs')
flags.DEFINE_string('teacher_ckpt', '', 'Location of teacher ckpt')
flags.DEFINE_string('student_ckpt', '', 'Location of student ckpt')
flags.DEFINE_string('student_network', '', 'Student network name')
flags.DEFINE_string('student_scope', '', 'Student network scope name')
flags.DEFINE_string('teacher_network', '', 'Teacher network name')
flags.DEFINE_integer('labels_offset', 0, 'Input image resolution')
flags.DEFINE_string('preprocessing_name', '', 'Preprocessing name')
flags.DEFINE_integer('num_preprocessing_threads', 4, 'Preprocessing threads count')
flags.DEFINE_float('adam_beta1',0.9,'The decay rate of momentum')
flags.DEFINE_float('adam_beta2',0.999,'The decay rate of accumulated gradient')
flags.DEFINE_float('opt_epsilon',1.0,'Epsilon term for the optimiser')
flags.DEFINE_string('optimiser','sgd','Optimiser to use')

flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

flags.DEFINE_boolean('clone_on_cpu', False,
	                    'Use CPUs to deploy clones.')

flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

FLAGS = flags.FLAGS

_LEARNING_RATE_DECAY_FACTOR = 0.94


def get_learning_rate():
  if FLAGS.fine_tune_checkpoint:
    # If we are fine tuning a checkpoint we need to start at a lower learning
    # rate since we are farther along on training.
    return 1e-4
  else:
    return 0.045


def get_quant_delay():
  if FLAGS.fine_tune_checkpoint:
    # We can start quantizing immediately if we are finetuning.
    return 0
  else:
    # We need to wait for the model to train a bit before we quantize if we are
    # training from scratch.
    return 250000

def build_model():
    """Builds graph for model to train with rewrites for quantization.
    Returns:
      g: Graph with fake quantization ops and batch norm folding suitable for
      training quantized weights.
      train_tensor: Train op for execution during training.
    """
    is_training = True
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.worker_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)

    # Create global_step
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    # Training so read train split of the dataset  
    dataset = dataset_factory.get_dataset('imagenet', 'train',
                                          FLAGS.dataset_dir)
    
    network_fn_student = nets_factory.get_network_fn(
        FLAGS.student_network,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=True)

    network_fn_teacher = nets_factory.get_network_fn(
        FLAGS.teacher_network,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    preprocessing_name = FLAGS.preprocessing_name or FLAGS.student_network
      
    image_preprocessing_fn_student = preprocessing_factory.get_preprocessing(
      preprocessing_name, is_training=is_training)
  
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.teacher_network
    image_preprocessing_fn_teacher = preprocessing_factory.get_preprocessing(
      preprocessing_name, is_training=False)

    
    with tf.device(deploy_config.inputs_device()):
      provider = slim.dataset_data_provider.DatasetDataProvider(
          dataset,
          num_readers=FLAGS.num_readers,
          common_queue_capacity=20 * FLAGS.batch_size,
          common_queue_min=10 * FLAGS.batch_size)
      [image, label] = provider.get(['image', 'label'])
      label -= FLAGS.labels_offset
      
      eval_image_size = FLAGS.eval_image_size or network_fn_student.default_image_size
      image_student = image_preprocessing_fn_student(image, eval_image_size, eval_image_size)

      eval_image_size = FLAGS.eval_image_size or network_fn_teacher.default_image_size
      image_teacher = image_preprocessing_fn_teacher(image, eval_image_size, eval_image_size)
  
      images_student, images_teacher, labels = tf.train.batch(
        [image_student, image_teacher, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)
      labels = slim.one_hot_encoding(
            labels, dataset.num_classes - FLAGS.labels_offset)
      batch_queue = slim.prefetch_queue.prefetch_queue(
            [images_student, images_teacher, labels], capacity=2 * deploy_config.num_clones)


    def clone_fn(batch_queue):
      """Allows data parallelism by creating multiple clones of network_fn."""
      images_student, images_teacher ,labels = batch_queue.dequeue()
      logits, _, _ = network_fn_teacher(images_teacher,create_aux_logits=False)
      logits = logits/FLAGS.Temperature
      predictions = tf.nn.softmax(logits)
  
      (logits_student,end_points), _ = network_fn_student(images_student)
      logits_student_teacher = logits_student/FLAGS.Temperature
      #############################
      # Specify the loss function #
      #############################
      tf.losses.softmax_cross_entropy(
           labels, logits_student, label_smoothing=FLAGS.label_smoothing, weights=1.0)
      tf.losses.softmax_cross_entropy(predictions, logits_student_teacher, weights=FLAGS.Temperature**2)
      return end_points

    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
    first_clone_scope = deploy_config.clone_scope(0)
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Add summaries for end_points.
    end_points = clones[0].outputs
    for end_point in end_points:
      x = end_points[end_point]
      summaries.add(tf.summary.histogram('activations/' + end_point, x))
      summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                      tf.nn.zero_fraction(x)))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))

    if FLAGS.quantize:
      tf.contrib.quantize.create_training_graph(quant_delay=get_quant_delay())

    # Configure the learning rate using an exponential decay.
    num_epochs_per_decay = 2.5
    imagenet_size = 1271167
    decay_steps = int(imagenet_size / FLAGS.batch_size * num_epochs_per_decay)

    if FLAGS.moving_average_decay:
      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, global_step)
    else:
      moving_average_variables, variable_averages = None, None

    with tf.device(deploy_config.optimizer_device()):
      learning_rate = tf.train.exponential_decay(
          get_learning_rate(),
          global_step,
          decay_steps,
          _LEARNING_RATE_DECAY_FACTOR,
          staircase=True)
      if FLAGS.optimiser == 'sgd':
      	opt = tf.train.GradientDescentOptimizer(learning_rate)
      elif FLAGS.optimiser == 'adam':
      	opt = tf.train.AdamOptimizer(learning_rate,beta1=FLAGS.adam_beta1,beta2=FLAGS.adam_beta2,epsilon=FLAGS.opt_epsilon)
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    if FLAGS.sync_replicas:
      # If sync_replicas is enabled, the averaging will be done in the chief
      # queue runner.
      opt = tf.train.SyncReplicasOptimizer(
          opt=opt,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          total_num_replicas=FLAGS.worker_replicas,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables)
    elif FLAGS.moving_average_decay:
      # Update ops executed locally by trainer.
      update_ops.append(variable_averages.apply(moving_average_variables))
    
    total_loss, clones_gradients = model_deploy.optimize_clones(
        clones,
        opt,
        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=FLAGS.student_scope))
    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss', total_loss))

    # Create gradient updates.
    grad_updates = opt.apply_gradients(clones_gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
      train_tensor = tf.identity(total_loss, name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    return train_tensor,summary_op, opt, global_step

def get_checkpoint_init_fn(student_scope):
    """Returns the checkpoint init_fn if the checkpoint is provided."""

    variables_to_restore = slim.get_variables_to_restore()
    global_step_reset = tf.assign(tf.train.get_or_create_global_step(), 0)
    
    teacher_init_fn = slim.assign_from_checkpoint_fn(
            FLAGS.teacher_ckpt,
            variables_to_restore,
            ignore_missing_vars=True)

    student_init_fn = slim.assign_from_checkpoint_fn(
             FLAGS.student_ckpt,
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=student_scope),
             ignore_missing_vars=True)
               
    def init_fn(sess):

        teacher_init_fn(sess)
        if FLAGS.student_ckpt != '':
          student_init_fn(sess)
        # If we are restoring from a floating point model, we need to initialize
        # the global step to zero for the exponential decay to result in
        # reasonable learning rates.
        sess.run(global_step_reset)
    return init_fn
  

def train_model():
    """Trains student network"""
    tf.logging.set_verbosity(tf.logging.INFO)
    train_step, summary_op, optimizer, global_step = build_model()
    slim.learning.train(
        train_step,
        FLAGS.checkpoint_dir,
        is_chief=(FLAGS.task == 0),
        master=FLAGS.master,
        summary_op=summary_op,
        log_every_n_steps=FLAGS.log_every_n_steps,
        number_of_steps=FLAGS.number_of_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        init_fn=get_checkpoint_init_fn(FLAGS.student_scope),
        global_step=global_step,
        sync_optimizer=optimizer if FLAGS.sync_replicas else None)

def main(unused_arg):
  train_model()


if __name__ == '__main__':
  tf.app.run(main)
