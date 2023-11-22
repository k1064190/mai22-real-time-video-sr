"""Define the learner."""

import pathlib
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay

from learner.saver import Saver
from learner.metric import PSNR, SSIM

import logging
import os
import cv2 as cv


class StandardLearner():
    """Implement the standard learner.

    Attributes:
        config: A `dict` contains the configuration of the learner.
        model: A list of `tf.keras.Model` objects which generate predictions.
        dataset: A dataset `dict` contains dataloader for different split.
        step: An `int` represents the current step. Initialize to 0.
        optimizer: A `tf.keras.optimizers` is used to optimize the model. Initialize to None.
        lr_scheduler: A `tf.keras.optimizers.schedules.LearningRateSchedule` is used to schedule
            the leaning rate. Initialize to None.
        metric_functions: A `dict` contains one or multiple functions which are used to
            metric the results. Initialize to {}.
        saver: A `Saver` is used to save checkpoints. Initialize to None.
        summary: A `TensorboardSummary` is used to save eventfiles. Initialize to None.
        log_dir: A `str` represents the directory which records experiments.
        steps: An `int` represents the number of train steps.
        log_train_info_steps: An `int` represents frequency of logging training information.
        keep_ckpt_steps: An `int` represents frequency of saving checkpoint.
        valid_steps: An `int` represents frequency  of validation.
    """

    def __init__(self, restore_ckpt, model, dataset, log_dir, steps=100):
        """Initialize the learner and attributes.

        Args:
            model: Please refer to Attributes.
            dataset: Please refer to Attributes.
            log_dir: Please refer to Attributes.
        """
        super().__init__()

        self.total_steps = steps
        self.model = model
        self.dataset = dataset
        self.log_dir = log_dir
        self.log_txt = os.path.join(log_dir, 'logs.txt')
        self.restore_ckpt = restore_ckpt

        self.log_train_info_steps = 100
        self.keep_ckpt_steps = 5000
        self.valid_steps = 1000

        self.step = 0
        self.optimizer = None
        self.lr_scheduler = None
        self.metric_functions = {}
        self.saver = None
        self.summary = None

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        fh = logging.FileHandler(self.log_txt)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def register_training(self):
        """Prepare for training."""
        # prepare learning rate scheduler for training
        self.lr_scheduler = ExponentialDecay(
            initial_learning_rate=0.0001,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True,
        )

        # prepare optimizer for training
        self.optimizer = keras.optimizers.Adam(
            learning_rate=self.lr_scheduler, beta_1=0.9, beta_2=0.999, epsilon=1e-8
        )

        # prepare saver to save and load checkpoints
        self.saver = Saver(self.restore_ckpt, self, is_train=True, log_dir=self.log_dir)

        # prepare metric functions
        self.metric_functions['psnr'] = PSNR(data_range=255)
        self.metric_functions['ssim'] = SSIM(data_range=255)

    def register_evaluation(self):
        """Prepare for evaluation."""
        # prepare saver to save and load checkpoints
        self.saver = Saver(self.restore_ckpt, self, is_train=False, log_dir=self.log_dir)

        # prepare metric functions
        self.metric_functions['psnr'] = PSNR(data_range=255)
        self.metric_functions['ssim'] = SSIM(data_range=255)

    def loss_fn(self, pred_tensor, target_tensor):
        """Define the objective function and prepare loss for backward.

        Args:
            pred_tensor: A `torch.Tensor` represents the prediction.
            target_tensor: A `torch.Tensor` represents the target.
        """
        # l1 charbonnier loss
        epsilon = 1e-6
        diff = pred_tensor - target_tensor
        loss = tf.math.sqrt(diff * diff + epsilon)
        return tf.reduce_mean(loss)

    def log_metric(self, prefix=''):
        """Log the metric values."""
        metric_dict = {}
        for metric_name in self.metric_functions:
            value = self.metric_functions[metric_name].get_result().numpy()
            self.metric_functions[metric_name].reset()

            self.logger.info(f'Step: {self.step}, {prefix}Metric: {metric_name}: {value}')
            metric_dict[metric_name] = value

    @tf.function
    def train_step(self, data):
        """Define one training step.

        Args:
            data: A `tuple` contains input and target tensor.
        """
        input_tensors, target_tensors = data
        recurrent_steps = target_tensors.shape[1]  # T

        l1_norm_loss = 0
        with tf.GradientTape() as tape:
            for i in range(recurrent_steps):
                if i == 0:
                    b, _, h, w, _ = input_tensors.shape.as_list()
                    input_tensor = tf.concat(
                        [input_tensors[:, 0, ...], input_tensors[:, 0, ...]], axis=-1
                    )
                    hidden_state = tf.zeros([b, h, w, self.model.base_channels])
                    pred_tensor, hidden_state = self.model([input_tensor, hidden_state], training=True)
                else:
                    input_tensor = tf.concat(
                        [input_tensors[:, i - 1, ...], input_tensors[:, i, ...]], axis=-1
                    )
                    pred_tensor, hidden_state = self.model([input_tensor, hidden_state], training=True)
                l1_norm_loss += self.loss_fn(pred_tensor, target_tensors[:, i, ...])
        # Calculate gradients and update.
        gradients = tape.gradient(l1_norm_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return pred_tensor, l1_norm_loss

    def test_step(self, data):
        """Define one testing step.

        Args:
            data: A `tuple` contains input and target tensor.
        """
        input_tensors, target_tensors = data
        recurrent_steps = target_tensors.shape[1]  # T

        pred_tensors = []
        for i in range(recurrent_steps):
            if i == 0:
                b, _, h, w, _ = input_tensors.shape.as_list()
                input_tensor = tf.concat(
                    [input_tensors[:, 0, ...], input_tensors[:, 0, ...]], axis=-1
                )
                hidden_state = tf.zeros([b, h, w, self.model.base_channels])
                pred_tensor, hidden_state = self.model([input_tensor, hidden_state], training=False)
            else:
                input_tensor = tf.concat(
                    [input_tensors[:, i - 1, ...], input_tensors[:, i, ...]], axis=-1
                )
                pred_tensor, hidden_state = self.model([input_tensor, hidden_state], training=False)

            for metric_name in self.metric_functions:
                self.metric_functions[metric_name].update(pred_tensor, target_tensors[:, i, ...])

            pred_tensors.append(pred_tensor)

        return pred_tensors

    def train(self):
        """Train the model."""
        self.register_training()

        # restore checkpoint
        if self.saver.restore_ckpt:
            self.logger.info(f'Restore from {self.saver.restore_ckpt}')
            self.saver.load_checkpoint()
        else:
            self.logger.info('Train from scratch')

        train_loader = self.dataset['train']
        train_iterator = iter(train_loader)
        val_loader = self.dataset['val']

        # train loop
        while self.step < self.total_steps:
            try:
                data_pair = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                data_pair = next(train_iterator)

            # training
            pred, loss = self.train_step(data_pair)
            self.step = self.optimizer.iterations.numpy()

            # log the training information every n steps
            if self.step % self.log_train_info_steps == 0:
                self.logger.info(f'Step {self.step} train loss: {loss}, lr: {self.optimizer.lr(self.step)}')

            # save checkpoint every n steps
            if self.step % self.keep_ckpt_steps == 0:
                self.saver.save_checkpoint()

            # validation and log the validation results n steps
            if self.step % self.valid_steps == 0:
                for metric_name in self.metric_functions:
                    self.metric_functions[metric_name].reset()

                for data_pair in val_loader:
                    self.test_step(data_pair)
                    break

                # log the validation results
                self.log_metric('Val_')

        # save the checkpoint after finishing training
        self.saver.save_checkpoint()

    def test(self):
        """Evaluate the model."""
        self.register_evaluation()

        # restore checkpoint
        self.logger.info(f'Restore from {self.saver.restore_ckpt}')
        self.saver.load_checkpoint()

        val_loader = self.dataset['val']

        save_path = pathlib.Path(self.log_dir) / 'output'
        save_path.mkdir(exist_ok=True)
        for i, data_pair in enumerate(val_loader):
            pred_tensors = self.test_step(data_pair)
            for j, pred_tensor in enumerate(pred_tensors):
                tf.keras.utils.save_img(
                    save_path / f'{str(i).zfill(3)}_{str(j).zfill(8)}.png', pred_tensor[0]
                )

        # log the evaluation results
        self.log_metric('Test_')
