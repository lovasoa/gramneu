#!/usr/bin/env python3
# coding=utf-8

from tensor2tensor.bin import t2t_trainer

import tensorflow as tf

FLAGS = tf.flags.FLAGS


def train(generate_data=True):
    FLAGS.problem = "fix_grammar_mistakes"
    FLAGS.model = "transformer"
    FLAGS.generate_data = True
    FLAGS.hparams_set = "transformer_base_small_gpu"
    FLAGS.t2t_usr_dir = "src"
    FLAGS.output_dir = "t2t_output"
    FLAGS.data_dir = "t2t_data"
    t2t_trainer.main(None)


if __name__ == "__main__":
    train()
