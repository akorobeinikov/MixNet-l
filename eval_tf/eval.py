import json
import os
import sys

from absl import logging
import numpy as np
import preprocessing
import tensorflow as t
import tensorflow.compat.v1 as tf
import mixnet_model
import cv2
import mixnet_builder

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


class EvalCkptDriver(object):
    """A driver for running eval inference.
  Attributes:
    model_name: str. Model name to eval.
    batch_size: int. Eval batch size.
    image_size: int. Input image size, determined by model name.
    num_classes: int. Number of classes, default to 1000 for ImageNet.
    include_background_label: whether to include extra background label.
    advprop_preprocessing: whether to use advprop preprocessing.
  """

    def __init__(self,
                 model_name,
                 batch_size=1,
                 image_size=224,
                 num_classes=1000,
                 include_background_label=False,
                 advprop_preprocessing=False):
        """Initialize internal variables."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.include_background_label = include_background_label
        self.image_size = image_size
        self.advprop_preprocessing = advprop_preprocessing

    def get_ema_vars(self):
        """Get all exponential moving average (ema) variables."""
        ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
        for v in tf.global_variables():
            # We maintain mva for batch norm moving mean and variance as well.
            if 'moving_mean' in v.name or 'moving_variance' in v.name:
                ema_vars.append(v)
        return list(set(ema_vars))

    def restore_model(self, sess, ckpt_dir, enable_ema=True, export_ckpt=None):
        """Restore variables from checkpoint dir."""
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        if enable_ema:
            ema = tf.train.ExponentialMovingAverage(decay=0.0)
            ema_vars = self.get_ema_vars()
            var_dict = ema.variables_to_restore(ema_vars)
            ema_assign_op = ema.apply(ema_vars)
        else:
            var_dict = self.get_ema_vars()
            ema_assign_op = None

        tf.train.get_or_create_global_step()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_dict, max_to_keep=1)
        saver.restore(sess, checkpoint)

        if export_ckpt:
            if ema_assign_op is not None:
                sess.run(ema_assign_op)
            saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
            saver.save(sess, export_ckpt)

    def build_model(self, features, is_training):
        """Build model with input features."""
        features -= tf.constant(
            MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
        features /= tf.constant(
            STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)
        logits, _ = mixnet_builder.build_model(features, self.model_name,
                                               is_training)
        probs = tf.nn.softmax(logits)
        probs = tf.squeeze(probs)
        return probs

    def get_preprocess_fn(self):
        """Build input dataset."""
        return preprocessing.preprocess_image

    def build_dataset(self, filenames, labels, is_training):
        """Build input dataset."""
        batch_drop_remainder = False
        filenames = tf.constant(filenames)
        labels = tf.constant(labels)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

        def _parse_function(filename, label):
            image_string = tf.read_file(filename)
            preprocess_fn = self.get_preprocess_fn()
            image_decoded = preprocess_fn(
                image_string, is_training, image_size=self.image_size)
            image = tf.cast(image_decoded, tf.float32)
            return image, label

        dataset = dataset.map(_parse_function)
        dataset = dataset.batch(self.batch_size,
                                drop_remainder=batch_drop_remainder)

        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()
        return images, labels

    def run_inference(self,
                      ckpt_dir,
                      image_files,
                      labels,
                      enable_ema=True,
                      export_ckpt=None):
        """Build and run inference on the target images and labels."""
        label_offset = 1 if self.include_background_label else 0
        with tf.Graph().as_default(), tf.Session() as sess:
            images, labels = self.build_dataset(image_files, labels, False)
            print(images)
            probs = self.build_model(images, is_training=False)
            if isinstance(probs, tuple):
                probs = probs[0]

            self.restore_model(sess, ckpt_dir, enable_ema, export_ckpt)

            prediction_idx = []
            prediction_prob = []
            writer = tf.summary.FileWriter("C:/Users/aalle/OneDrive/Рабочий стол/wrok/MixNet/eval_tf/log", sess.graph)

            for _ in range(len(image_files) // self.batch_size):
                out_probs = sess.run(probs)
                idx = np.argsort(out_probs)[::-1]
                prediction_idx.append(idx[:5] - label_offset)
                prediction_prob.append([out_probs[pid] for pid in idx[:5]])
            writer.close()
            # Return the top 5 predictions (idx and prob) for each image.
            return prediction_idx, prediction_prob

    def eval_example_images(self,
                            ckpt_dir,
                            image_files,
                            labels_map_file,
                            enable_ema=True,
                            export_ckpt=None):
        """Eval a list of example images.
    Args:
      ckpt_dir: str. Checkpoint directory path.
      image_files: List[str]. A list of image file paths.
      labels_map_file: str. The labels map file path.
      enable_ema: enable expotential moving average.
      export_ckpt: export ckpt folder.
    Returns:
      A tuple (pred_idx, and pred_prob), where pred_idx is the top 5 prediction
      index and pred_prob is the top 5 prediction probability.
    """
        # load labels
        classes = json.loads(tf.gfile.Open(labels_map_file).read())
        print([0] * len(image_files))

        # inference for images
        pred_idx, pred_prob = self.run_inference(
            ckpt_dir, image_files, [0] * len(image_files), enable_ema, export_ckpt)

        # print the results
        for i in range(len(image_files)):
            print('predicted class for image {}: '.format(image_files[i]))
            for j, idx in enumerate(pred_idx[i]):
                print('  -> top_{} ({:4.2f}%): {}  '.format(j, pred_prob[i][j] * 100,
                                                            classes[str(idx)]))
        return pred_idx, pred_prob
