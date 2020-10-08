import logging
import os
import numpy as np
import functools
from collections import namedtuple
import random
from sklearn.metrics import pairwise
import sys
# np.set_printoptions(threshold=sys.maxsize)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.contrib import metrics, slim
from tensorflow.contrib.metrics import streaming_mean

from . import nn
from . import weight_norm as wn
from .framework import assert_shape, HyperparamVariables
from . import string_utils


LOG = logging.getLogger('main')


class Model:
    DEFAULT_HYPERPARAMS = {
        # loss weight hyper-parameters
        # *** wd_coefficient is coeffs without multiplying lr
        'max_consistency_cost': 1.365,
        'wd_coefficient': .00005,

        # Consistency hyper-parameters
        'ema_consistency': True,
        'ema_decay_during_rampup': 0.99,
        'ema_decay_after_rampup': 0.999,

        # Optimizer hyper-parameters
        'max_learning_rate': 0.003,
        'adam_beta_1_before_rampdown': 0.9,
        'adam_beta_1_after_rampdown': 0.5,
        'adam_beta_2_during_rampup': 0.99,
        'adam_beta_2_after_rampup': 0.999,
        'adam_epsilon': 1e-8,

        # Architecture hyper-parameters
        'input_noise': 0.15,
        'student_dropout_probability': 0.5,
        'teacher_dropout_probability': 0.5,

        # Training schedule
        'rampup_length': 58606,
        'rampdown_length': 36629,
        'training_length': 219771, #219771

        # Input augmentation
        'flip_horizontally': False,
        'translate': True,

        # Whether to scale each input image to mean=0 and std=1 per channel
        # Use False if input is already normalized in some other way
        'normalize_input': True,

        # Output schedule
        'print_span': 100,
        'evaluation_span': 2000,
    }

    # pylint: disable=too-many-instance-attributes
    def __init__(self, batch_size, context_size, n_labeled_per_batch, run_context=None):
        if run_context is not None:
            self.training_log = run_context.create_train_log('training')
            self.validation_log = run_context.create_train_log('validation')
            self.checkpoint_path = os.path.join(run_context.transient_dir, 'checkpoint')
            self.tensorboard_path = os.path.join(run_context.result_dir, 'tensorboard')

        with tf.name_scope("placeholders"):
            self.images = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='images')
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
            self.context_image = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='context_image')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')
            self.graph_A = tf.placeholder(dtype=tf.float32, shape=(None, context_size), name='graph_A')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        tf.add_to_collection("init_in_init", self.global_step)
        self.hyper = HyperparamVariables(self.DEFAULT_HYPERPARAMS)
        for var in self.hyper.variables.values():
            tf.add_to_collection("init_in_init", var)

        self.batch_size = batch_size
        self.context_size = context_size
        self.d_size = 128 #128
        self.n_labeled_per_batch = n_labeled_per_batch
        # *** new hyper-parameters
        self.wd_coefficient = self.hyper['wd_coefficient']

        with tf.name_scope("ramps"):
            sigmoid_rampup_value = sigmoid_rampup(self.global_step, self.hyper['rampup_length'])
            sigmoid_rampdown_value = sigmoid_rampdown(self.global_step,
                                                      self.hyper['rampdown_length'],
                                                      self.hyper['training_length'])
            self.learning_rate = tf.multiply(sigmoid_rampup_value * sigmoid_rampdown_value,
                                             self.hyper['max_learning_rate'],
                                             name='learning_rate')
            self.adam_beta_1 = tf.add(sigmoid_rampdown_value * self.hyper['adam_beta_1_before_rampdown'],
                                      (1 - sigmoid_rampdown_value) * self.hyper['adam_beta_1_after_rampdown'],
                                      name='adam_beta_1')

            self.cons_coefficient = tf.multiply(sigmoid_rampup_value,
                                                self.hyper['max_consistency_cost'],
                                                name='consistency_coefficient')

            step_rampup_value = step_rampup(self.global_step, self.hyper['rampup_length'])
            self.adam_beta_2 = tf.add((1 - step_rampup_value) * self.hyper['adam_beta_2_during_rampup'],
                                      step_rampup_value * self.hyper['adam_beta_2_after_rampup'],
                                      name='adam_beta_2')
            self.ema_decay = tf.add((1 - step_rampup_value) * self.hyper['ema_decay_during_rampup'],
                                    step_rampup_value * self.hyper['ema_decay_after_rampup'],
                                    name='ema_decay')

        tower_args = dict(is_training=self.is_training,
                          input_noise=self.hyper['input_noise'],
                          normalize_input=self.hyper['normalize_input'],
                          flip_horizontally=self.hyper['flip_horizontally'],
                          context_size = self.context_size,
                          output_dim = self.d_size,
                          translate=self.hyper['translate'])

        self.f_logits_1, self.student_bx128, self.embedding_bxm = \
            tower(**tower_args, inputs=self.images, dropout_probability=self.hyper['student_dropout_probability'])
        post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Take only first call to update batch norm.

        # self.f_logits_2,_ ,_ = \
        #     tower(**tower_args, inputs=self.images, dropout_probability=self.hyper['student_dropout_probability'])

        ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)
        ema_op = ema.apply(model_vars())
        ema_getter = functools.partial(getter_ema, ema)
        self.f_logits_ema, self.teacher_bx128, _ = \
            tower(**tower_args, inputs=self.images, dropout_probability=self.hyper['teacher_dropout_probability'], getter=ema_getter)
        self.f_logits_ema = tf.stop_gradient(self.f_logits_ema)
        self.teacher_bx128 = tf.stop_gradient(self.teacher_bx128)


        self.mx10, self.mx128, _ = \
            tower(**tower_args, inputs=self.context_image, dropout_probability=self.hyper['teacher_dropout_probability'], getter=ema_getter) #mx128 fix samples
        self.mx10 = tf.stop_gradient(self.mx10)
        self.mx128 = tf.stop_gradient(self.mx128)

        self.embed_logits, self.embed_labels = get_embed_logits_labels(self.graph_A, self.embedding_bxm, self.mx10, self.f_logits_ema)

        # Weight Decay for Adam Optimizer
        post_ops.append(ema_op)
        # post_ops.extend([tf.assign(v, v * (1 - self.wd_coefficient)) for v in model_vars('classify') if 'kernel' in v.name])
        # Weight Decay for SGD Optimizer L2 regularization
        # self.wd_cost = sum(tf.nn.l2_loss(v) for v in utils.model_vars('classify') if 'kernel' in v.name)

        with tf.name_scope("objectives"):
            # error of labeled samples
            _, self.errors_f_mt = errors(self.f_logits_ema, self.labels) #eval
            self.mean_error_f, self.errors_f = errors(self.f_logits_1, self.labels) #eval #train

            self.mean_class_cost_f, self.class_eval_costs_f = classification_costs_f(self.f_logits_1, self.labels) #eval #train
            self.mean_cons_cost_mt_f = \
                consistency_costs_f(self.f_logits_1, self.f_logits_ema, self.cons_coefficient) #train

            self.mean_embed_cost_f = embedding_costs_f(self.embed_logits, self.embed_labels)

            # initial_weight = tf.cond(tf.to_float(self.global_step) < 50000, lambda:10e-8, lambda:10e-7)
            initial_weight = 10e-7
            embed_cost_weight = tf.multiply(initial_weight, tf.to_float(self.global_step)) # weight= 10e-8*epoch
            # max_weight = tf.cond(tf.to_float(self.global_step) < 100000, lambda:0.01, lambda:0.001) # max weight=0.001 when global_step >= 1000000
            # embed_cost_weight = tf.minimum(0.01, embed_cost_weight) # avoid embed weight higher than 0.01
            self.mean_embed_cost_f = tf.multiply(embed_cost_weight, self.mean_embed_cost_f)


            # self.mean_total_cost_pi = self.mean_class_cost_f + self.mean_cons_cost_mt_f
            # self.mean_total_cost_mt = self.mean_class_cost_f + self.mean_cons_cost_mt_f #train

            self.mean_total_cost_pi = self.mean_class_cost_f + self.mean_cons_cost_mt_f+ self.mean_embed_cost_f
            self.mean_total_cost_mt = self.mean_class_cost_f + self.mean_cons_cost_mt_f+ self.mean_embed_cost_f#train

            self.cost_to_be_minimized = tf.cond(self.hyper['ema_consistency'],
                                                lambda: self.mean_total_cost_mt,
                                                lambda: self.mean_total_cost_pi)



        with tf.name_scope("train_step"):
            # post_ops.append(ema_op)
            self.train_step_op = nn.adam_optimizer(self.cost_to_be_minimized,
                                                   self.global_step,
                                                   learning_rate=self.learning_rate,
                                                   beta1=self.adam_beta_1,
                                                   beta2=self.adam_beta_2,
                                                   epsilon=self.hyper['adam_epsilon'])
            with tf.control_dependencies([self.train_step_op]):
                self.train_step_op = tf.group(*post_ops)

        # return a dict
        self.training_control = training_control(self.global_step,
                                                 self.hyper['print_span'],
                                                 self.hyper['evaluation_span'],
                                                 self.hyper['training_length'])

        # *** need to be fixed
        self.training_metrics = {
            "learning_rate": self.learning_rate,
            "adam_beta_1": self.adam_beta_1,
            "adam_beta_2": self.adam_beta_2,
            "ema_decay": self.ema_decay,
            "cons_coefficient_f": self.cons_coefficient,
            "train/error/f": self.mean_error_f,
            "train/class_cost/f": self.mean_class_cost_f,
            "train/cons_cost/f": self.mean_cons_cost_mt_f,
            # "train/total_cost/pi": self.mean_total_cost_pi,
            "train/embed_cost/f": self.mean_embed_cost_f,
            "train/total_cost/mt": self.mean_total_cost_mt
        }

        self.features = {
            "mx128": self.mx128,
            "teacher_bx128": self.teacher_bx128
        }

        # *** need to be fixed
        with tf.variable_scope("validation_metrics") as metrics_scope:
            self.metric_values, self.metric_update_ops = metrics.aggregate_metric_map({
                "eval/error/f_mt": streaming_mean(self.errors_f_mt),
                "eval/error/f": streaming_mean(self.errors_f),
                "eval/class_cost/f": streaming_mean(self.class_eval_costs_f),
                "eval/embed_cost/f": streaming_mean(self.mean_embed_cost_f),
            })
            metric_variables = slim.get_local_variables(scope=metrics_scope.name)
            self.metric_init_op = tf.variables_initializer(metric_variables)

        self.result_formatter = string_utils.DictFormatter(
            order=["error/f", "class_cost/f", "cons_cost/f", "embed_cost/f"],
            default_format='{name}: {value:>10.6f}',
            separator=",  ")
        self.result_formatter.add_format('error', '{name}: {value:>6.2%}')

        with tf.name_scope("initializers"):
            init_init_variables = tf.get_collection("init_in_init")
            train_init_variables = [
                var for var in tf.global_variables() if var not in init_init_variables
            ]
            self.init_init_op = tf.variables_initializer(init_init_variables)
            self.train_init_op = tf.variables_initializer(train_init_variables)

        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.run(self.init_init_op)

        # add collection to restore model
        tf.add_to_collection('images', self.images)
        tf.add_to_collection('is_training', self.is_training)
        tf.add_to_collection('output_f', self.f_logits_1)
        tf.add_to_collection('output_f_mt', self.f_logits_ema)

    def __setitem__(self, key, value):
        self.hyper.assign(self.session, key, value)

    def __getitem__(self, key):
        return self.hyper.get(self.session, key)

    def train(self, training_batches, evaluation_batches_fn, context_training):
        self.run(self.train_init_op)
        LOG.info("Model variables initialized")
        print("sigma=1")
        mx128 = self.evaluate(evaluation_batches_fn, context_training)
        self.save_checkpoint()
        for batch in training_batches:
            teacher_bx128 = self.run(self.teacher_bx128, self.feed_dict(batch, context_training, is_training=False))
            A = adj_matrix(mx128, teacher_bx128, sigma=1) # A.shape = (b,m)
            results, _ = self.run([self.training_metrics, self.train_step_op],
                                  self.feed_dict(batch, context_training, A))
            step_control = self.get_training_control()
            self.training_log.record(step_control['step'], {**results, **step_control})
            if step_control['time_to_print']:
                LOG.info("step %5d:   %s", step_control['step'], self.result_formatter.format_dict(results))
            if step_control['time_to_stop']:
                break
            if step_control['time_to_evaluate']:
                mx128 = self.evaluate(evaluation_batches_fn, context_training)
                self.save_checkpoint()
        mx128 = self.evaluate(evaluation_batches_fn, context_training)
        self.save_checkpoint()

    def evaluate(self, evaluation_batches_fn, context_evaluation):
        self.run(self.metric_init_op)
        for batch in evaluation_batches_fn():
            feat = self.run(self.features, self.feed_dict(batch, context_evaluation, is_training=False))
            A = adj_matrix(feat['mx128'], feat['teacher_bx128'], sigma=1) # A is a matrix with dimension b*m
            self.run(self.metric_update_ops,
                     feed_dict=self.feed_dict(batch, context_evaluation, A, is_training=False))
        step = self.run(self.global_step)
        results = self.run(self.metric_values)
        self.validation_log.record(step, results)
        LOG.info("step %5d:   %s", step, self.result_formatter.format_dict(results))
        return feat['mx128']

    def get_training_control(self):
        return self.session.run(self.training_control)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def feed_dict(self, batch, context_image, A=None, is_training=True):
        if A is None:
            A = np.zeros(shape=(batch['x'].shape[0], context_image.shape[0]), dtype=float)

        return {
            self.images: batch['x'],
            self.labels: batch['y'],
            self.context_image: context_image,
            self.graph_A: A,
            self.is_training: is_training
        }

    def save_checkpoint(self):
        path = self.saver.save(self.session, self.checkpoint_path, global_step=self.global_step)
        LOG.info("Saved checkpoint: %r", path)

    def save_tensorboard_graph(self):
        writer = tf.summary.FileWriter(self.tensorboard_path)
        writer.add_graph(self.session.graph)
        return writer.get_logdir()


Hyperparam = namedtuple("Hyperparam", ['tensor', 'getter', 'setter'])


def model_vars(scope=None):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)


def getter_ema(ema, getter, name, *args, **kwargs):
    """Exponential moving average getter for variable scopes.
    Args:
        ema: ExponentialMovingAverage object, where to get variable moving averages.
        getter: default variable scope getter.
        name: variable name.
        *args: extra args passed to default getter.
        **kwargs: extra args passed to default getter.
    Returns:
        If found the moving average variable, otherwise the default variable.
    """
    var = getter(name, *args, **kwargs)
    ema_var = ema.average(var)
    return ema_var if ema_var else var


def training_control(global_step, print_span, evaluation_span, max_step, name=None):
    with tf.name_scope(name, "training_control"):
        return {
            "step": global_step,
            "time_to_print": tf.equal(tf.mod(global_step, print_span), 0),
            "time_to_evaluate": tf.equal(tf.mod(global_step, evaluation_span), 0),
            "time_to_stop": tf.greater_equal(global_step, max_step),
        }


def step_rampup(global_step, rampup_length):
    result = tf.cond(global_step < rampup_length,
                     lambda: tf.constant(0.0),
                     lambda: tf.constant(1.0))
    return tf.identity(result, name="step_rampup")


def sigmoid_rampup(global_step, rampup_length):
    global_step = tf.to_float(global_step)
    rampup_length = tf.to_float(rampup_length)
    def ramp():
        phase = 1.0 - tf.maximum(0.0, global_step) / rampup_length
        return tf.exp(-5.0 * phase * phase)

    result = tf.cond(global_step < rampup_length, ramp, lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampup")


def sigmoid_rampdown(global_step, rampdown_length, training_length):
    global_step = tf.to_float(global_step)
    rampdown_length = tf.to_float(rampdown_length)
    training_length = tf.to_float(training_length)
    def ramp():
        phase = 1.0 - tf.maximum(0.0, training_length - global_step) / rampdown_length
        return tf.exp(-12.5 * phase * phase)

    result = tf.cond(global_step >= training_length - rampdown_length,
                     ramp,
                     lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampdown")

def adj_matrix(mx128, bx128, sigma=1):
    gamma = 1/(2*sigma**2)
    dist = pairwise.rbf_kernel(bx128, mx128, gamma=gamma)
    return dist

def is_empty(input):
    return tf.equal(tf.size(input), 0)

def random_walk(to_sample_randomwalk, path_len):
    log_prob = tf.log(to_sample_randomwalk)
    after_random_walk = tf.random.categorical(log_prob, path_len) # return shape = (row number of to_sample_randomwalk, path_len)

    # get column index for each row of after_random_walk
    equal_prob = 0.5*tf.ones((tf.shape(after_random_walk)[1],))
    log_prob = tf.log([equal_prob])
    column_indices = tf.random.categorical(log_prob, tf.shape(after_random_walk)[0]) # return shape = (1, n_samples)
    column_indices = tf.squeeze(column_indices, 0)  # (n_samples,)

    #get context indices
    row_indices = tf.range(tf.shape(after_random_walk, out_type=tf.dtypes.int64)[0]) # select each row
    full_indices = tf.stack([row_indices, column_indices], axis=1) # shape = (n_samples, 2)
    context_ind = tf.gather_nd(after_random_walk, full_indices)

    return context_ind

def sample_uniform(to_uniform_sample):
    # get column index for each row of to_uniform_sample
    equal_prob = 0.5*tf.ones((tf.shape(to_uniform_sample)[1],))
    log_prob = tf.log([equal_prob])
    column_indices = tf.random.categorical(log_prob, tf.shape(to_uniform_sample)[0]) # return shape = (1, n_samples)
    column_indices = tf.squeeze(column_indices, 0)  # (n_samples,)

    return column_indices

def sample_graph_A(to_sample_randomwalk, to_uniform_sample, path_len):
    # labels from random walk
    labels_rw = tf.cond(is_empty(to_sample_randomwalk), lambda: tf.constant([], dtype=tf.int64), lambda: random_walk(to_sample_randomwalk, path_len))  # return [] if is_empty
    # labels from uniform sample
    labels_uniform = tf.cond(is_empty(to_uniform_sample), lambda: tf.constant([], dtype=tf.int64), lambda: sample_uniform(to_uniform_sample)) # return [] if is_empty

    return labels_rw, labels_uniform

def return_context_ind(bx10_label_x_ind, mx10_label_x_ind):
    num_row_bx10_labelx = tf.shape(bx10_label_x_ind, out_type=tf.dtypes.int64)[1]
    random_ind_x = tf.random.uniform((num_row_bx10_labelx,), maxval= tf.shape(mx10_label_x_ind, out_type=tf.dtypes.int64)[0], dtype=tf.dtypes.int64)
    context_ind_x = tf.gather(mx10_label_x_ind, random_ind_x)
    return context_ind_x

def return_neg1(bx10_label_x_ind):
    return -tf.ones(tf.shape(bx10_label_x_ind)[1], dtype = tf.dtypes.int64)

def exclude_label(mx10_label, num_labels):
  lis = []
  for i in range (num_labels):
    mask = [True] * num_labels
    mask[i] = False
    after_mask = [[i] for (i, v) in zip(mx10_label, mask) if v]
    after_mask = tf.concat(after_mask, axis=1)
    after_mask = tf.squeeze(after_mask, axis=0)
    lis.append(after_mask)
  return lis

def sample_equal_diff_pseudolabel(to_be_sampled, mx10, action):
    assert (action == 'equal' or action == 'diff')
    num_labels = 10
    int64 = tf.dtypes.int64
    bx10_pseudo_label = tf.argmax(to_be_sampled, axis=1)
    mx10_pseudo_label = tf.argmax(mx10, axis=1)

    # index of each label in bx10 pseudolabel list
    constant_list = [tf.constant(i, dtype=int64) for i in range(num_labels)]
    bx10_label = [[ tf.where(tf.equal(bx10_pseudo_label, constant_list[i]))[:,-1] ] for i in range(num_labels)]
    bx10_label_total_ind = tf.concat(bx10_label, axis=1)
    bx10_label_total_ind = tf.squeeze(bx10_label_total_ind, axis=0)

    # index of each label in mx10 pseudolabel list
    mx10_equal_label = [tf.where(tf.equal(mx10_pseudo_label, constant_list[i]))[:,-1] for i in range(num_labels)]

    if action == 'equal':
        mx10label = mx10_equal_label
    else:
        mx10label = exclude_label(mx10_equal_label, num_labels)

    # random sample
    context_ind_each_label = [[tf.cond(is_empty(mx10label[i]),
                                lambda: return_neg1(bx10_label[i]),
                                lambda: return_context_ind(bx10_label[i], mx10label[i]))]
                                for i in range(num_labels)]

    context_ind_total = tf.concat(context_ind_each_label, axis=1)
    context_ind_total = tf.squeeze(context_ind_total, axis=0)
    context_ind_total = tf.gather(context_ind_total, tf.argsort(bx10_label_total_ind, axis=-1))

    return context_ind_total

def sample_pseudo_label(to_sample_equal_label, to_sample_diff_label, mx10):

    # labels from sampling equal pseudolabel
    labels_equal = tf.cond(is_empty(to_sample_equal_label), lambda: tf.constant([], dtype=tf.int64), lambda: sample_equal_diff_pseudolabel(to_sample_equal_label, mx10, 'equal'))  # return [] if is_empty
    # labels from sampling different pseudolabel
    labels_different = tf.cond(is_empty(to_sample_diff_label), lambda: tf.constant([], dtype=tf.int64), lambda: sample_equal_diff_pseudolabel(to_sample_diff_label, mx10, 'diff')) # return [] if is_empty

    return labels_equal, labels_different


def predict_context(bxm, gamma):
    bxm = bxm*gamma
    bxm = tf.sigmoid(bxm)
    return bxm

def filter(input, mask):
    mask = tf.squeeze(mask, axis=1) #(b,1) -> (b,)

    mask_pos = tf.math.equal(mask, tf.constant(1.))
    mask_neg = tf.math.equal(mask, tf.constant(-1.))

    pos = tf.boolean_mask(input, mask_pos)
    neg = tf.boolean_mask(input, mask_neg)

    return pos, neg

def get_embed_logits_labels(A, embedding_bxm, mx10, teacher_bx10):
    r1 = 0.8 #0.5
    r2 = 0.2

    print("r1:", r1)
    print("r2:", r2)


    constant_one = tf.ones((tf.shape(A)[0],1))
    random_r1 = tf.random.uniform((tf.shape(A)[0],1)) #dim = bx1
    gamma = tf.where(random_r1 < r1, constant_one, -constant_one)

    random_r2 = tf.random.uniform((tf.shape(A)[0],1)) #dim = bx1
    beta = tf.where(random_r2 < r2, constant_one, -constant_one) # beta =+1 sample from graph, beta=-1 sample from pseudolabel

    embed_logits = predict_context(embedding_bxm, gamma)

    # divide graph A
    A_r2pos, _ = filter(A, beta)
    gamma_pos, gamma_neg = filter(gamma, beta)
    A_r2pos_r1pos, A_r2pos_r1neg = filter(A_r2pos, gamma_pos) # A_r2pos_r1pos do random walk, A_r2pos_r1neg do uniform sample

    # divide teacher_bx10
    _, bx10_r2neg = filter(teacher_bx10, beta)
    bx10_r2neg_r1pos, bx10_r2neg_r1neg = filter(bx10_r2neg, gamma_neg) # bx10_r2neg_r1pos sample from yi=yc, bx10_r2neg_r1neg sample from yi!=yc

    # divide logits
    logits_r2pos, logits_r2neg = filter(embed_logits, beta)
    logits_r2pos_r1pos, logits_r2pos_r1neg = filter(logits_r2pos, gamma_pos) # logits_r2pos_r1pos -> random walk, logits_r2pos_r1neg -> uniform sample
    logits_r2neg_r1pos, logits_r2neg_r1neg = filter(logits_r2neg, gamma_neg) # logits_r2neg_r1pos -> equal pseudolabel, logits_r2neg_r1neg -> diff pseudolabel

    # get one_hot label
    labels_rw, labels_uniform = sample_graph_A(A_r2pos_r1pos, A_r2pos_r1neg, path_len=10)
    labels_equal, labels_different = sample_pseudo_label(bx10_r2neg_r1pos, bx10_r2neg_r1neg, mx10)

    # concatenate one_hot labels and logits
    embed_labels = tf.concat([[labels_rw], [labels_uniform], [labels_equal], [labels_different]], axis=1) #concat vertically
    one_hot = tf.one_hot(tf.squeeze(embed_labels, axis=0), tf.shape(mx10)[0])
    embed_logits = tf.concat([logits_r2pos_r1pos, logits_r2pos_r1neg, logits_r2neg_r1pos, logits_r2neg_r1neg], axis=0) #concat vertically

    # return embed_logits, embed_labels
    return embed_logits, one_hot


def tower(inputs,
          is_training,
          dropout_probability,
          input_noise,
          normalize_input,
          flip_horizontally,
          translate,
          context_size,
          output_dim,
          getter=None):
    conv_args = dict(kernel_size=3, padding='same')
    bn_args = dict(training=is_training, momentum=0.999)
    with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
        net = inputs
        assert_shape(net, [None, 32, 32, 3])

        net = tf.cond(normalize_input,
                      lambda: slim.layer_norm(net,
                                              scale=False,
                                              center=False,
                                              scope='normalize_inputs'),
                      lambda: net)
        assert_shape(net, [None, 32, 32, 3])

        # Data Augmentation
        net = nn.flip_randomly(net,
                               horizontally=flip_horizontally,
                               vertically=False,
                               is_training=is_training,
                               name='random_flip')
        net = tf.cond(translate,
                      lambda: nn.random_translate(net, scale=2, is_training=is_training, name='random_translate'),
                      lambda: net)
        net = nn.gaussian_noise(net, scale=input_noise, is_training=is_training, name='gaussian_noise')

        net = tf.layers.conv2d(net, 128, **conv_args)
        net = tf.layers.batch_normalization(net, **bn_args)
        net = nn.lrelu(net)
        net = tf.layers.conv2d(net, 128, **conv_args)
        net = tf.layers.batch_normalization(net, **bn_args)
        net = nn.lrelu(net)
        net = tf.layers.conv2d(net, 128, **conv_args)
        net = tf.layers.batch_normalization(net, **bn_args)
        net = nn.lrelu(net)
        net = tf.layers.max_pooling2d(net, 2, 2)
        net = tf.layers.dropout(net, dropout_probability, training=is_training)
        assert_shape(net, [None, 16, 16, 128])

        net = tf.layers.conv2d(net, 256, **conv_args)
        net = tf.layers.batch_normalization(net, **bn_args)
        net = nn.lrelu(net)
        net = tf.layers.conv2d(net, 256, **conv_args)
        net = tf.layers.batch_normalization(net, **bn_args)
        net = nn.lrelu(net)
        net = tf.layers.conv2d(net, 256, **conv_args)
        net = tf.layers.batch_normalization(net, **bn_args)
        net = nn.lrelu(net)
        net = tf.layers.max_pooling2d(net, 2, 2)
        net = tf.layers.dropout(net, dropout_probability, training=is_training)
        assert_shape(net, [None, 8, 8, 256])

        net = tf.layers.conv2d(net, 512, kernel_size=3, padding='valid')
        net = tf.layers.batch_normalization(net, **bn_args)
        net = nn.lrelu(net)
        assert_shape(net, [None, 6, 6, 512])
        net = tf.layers.conv2d(net, 256, kernel_size=1, padding='same')
        net = tf.layers.batch_normalization(net, **bn_args)
        net = nn.lrelu(net)
        net = tf.layers.conv2d(net, 128, kernel_size=1, padding='same')
        assert_shape(net, [None, 6, 6, 128])

        # net = tf.layers.batch_normalization(net, **bn_args)             # test on or off this layer
        # net = nn.lrelu(net)                                             # test on or off this layer

        # store b_6x6x128
        b_6x6x128 = tf.reshape(net, [tf.shape(net)[0], 6*6*128])  # (bs, 6, 6, 128) -> (bs, 6x6x128)
        assert_shape(b_6x6x128, [None, 6*6*128])

        # store bx128
        net = tf.reduce_mean(net, [1, 2])  # (bs, 6, 6, 128) -> (bs, 128)
        bx128= net
        assert_shape(net, [None, 128])

        # to get bxd and embedding bxm
        net2 = tf.layers.dense(b_6x6x128, 1024)
        net2 = tf.layers.batch_normalization(net2, **bn_args)
        # net2 = nn.lrelu(net2)
        assert_shape(net2, [None, 1024])

        net2 = tf.layers.dense(net2, context_size)
        embedding_bxm = net2                                        # to predict graph context
        # net2 = tf.layers.batch_normalization(net2, **bn_args)       # test on or off this layer
        # net2 = nn.lrelu(net2)                                       # test on or off this layer
        assert_shape(net2, [None, context_size])

        net2 = tf.layers.dense(net2, output_dim)
        # net2 = tf.layers.batch_normalization(net2, **bn_args)       # test on or off this layer
        # net2 = nn.lrelu(net2)                                       # test on or off this layer
        to_concat_bxd = net2                                          # to concat with bx128 and predict class label
        assert_shape(net2, [None, output_dim])

        # concat and predict f_logits
        net = tf.concat([bx128, to_concat_bxd], 1)                  # concat horizontally
        net = tf.layers.batch_normalization(net, **bn_args)         # test on or off this layer
        net = nn.lrelu(net)                                         # test on or off this layer
        assert_shape(net, [None, 128+output_dim])

        f_logits = tf.layers.dense(net, 10)
        assert_shape(f_logits, [None, 10])

        return f_logits, bx128, embedding_bxm # return (bx10), (bx128), (b,m)


def errors(logits, labels, name=None):
    """Compute error mean and whether each labeled example is erroneous
    Assume unlabeled examples have label == -1.
    Compute the mean error over labeled examples.
    Mean error is NaN if there are no labeled examples.
    Note that unlabeled examples are treated differently in cost calculation.
    """
    with tf.name_scope(name, "errors") as scope:
        applicable = tf.not_equal(labels, -1)
        labels = tf.boolean_mask(labels, applicable)
        logits = tf.boolean_mask(logits, applicable)
        predictions = tf.argmax(logits, -1)
        labels = tf.cast(labels, tf.int64)
        per_sample = tf.to_float(tf.not_equal(predictions, labels))
        mean = tf.reduce_mean(per_sample, name=scope)
        return mean, per_sample


def classification_costs_f(logits, labels, name=None):
    """Compute classification cost mean and classification cost per sample
    Assume unlabeled examples have label == -1. For unlabeled examples, cost == 0.
    Compute the mean over all examples.
    Note that unlabeled examples are treated differently in error calculation.
    """
    with tf.name_scope(name, "classification_costs_f") as scope:
        applicable = tf.not_equal(labels, -1)

        # Change -1s to zeros to make cross-entropy computable
        labels = tf.where(applicable, labels, tf.zeros_like(labels))

        # This will now have incorrect values for unlabeled examples
        per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        # Retain costs only for labeled
        per_sample = tf.where(applicable, per_sample, tf.zeros_like(per_sample))

        # Take mean over all examples, not just labeled examples.
        labeled_sum = tf.reduce_sum(per_sample)
        total_count = tf.to_float(tf.shape(per_sample)[0])
        mean = tf.div(labeled_sum, total_count, name=scope)
        return mean, per_sample


def consistency_costs_f(logits1, logits2, cons_coefficient, name=None):
    """Calculate the consistency loss between class prediction f.
    """
    with tf.name_scope(name, "consistency_costs_f") as scope:
        assert_shape(logits1, [None, 10])
        assert_shape(logits2, [None, 10])
        assert_shape(cons_coefficient, [])

        logits1 = tf.nn.softmax(logits1)
        logits2 = tf.nn.softmax(logits2)

        per_sample = tf.reduce_mean((logits1 - logits2) ** 2, axis=-1)
        mean_cost = tf.multiply(tf.reduce_mean(per_sample), cons_coefficient, name=scope)
        assert_shape(mean_cost, [])
        return mean_cost

def embedding_costs_f(logits, labels, name=None): #dim of logits and labels = (b,m)
    with tf.name_scope(name, "embedding_costs_f") as scope:
        loss = -tf.reduce_sum(labels * tf.log(tf.add(logits, 1.0e-10)))
        total_count = tf.shape(logits)[0] #count number of samples in both logits
        total_count = tf.to_float(total_count)
        mean_cost = tf.div(loss, total_count)

        # labels = tf.argmax(labels, axis=1)
        # per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        # labeled_sum = tf.reduce_sum(per_sample)
        # total_count = tf.to_float(tf.shape(per_sample)[0])
        # mean_cost = tf.div(labeled_sum, total_count, name=scope)
        #

    return mean_cost
