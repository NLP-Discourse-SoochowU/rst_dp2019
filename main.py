# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date: 2018/5/4
@Description:
"""
from sys import argv
from utils.rst_utils import *
# from parser_model.form_data import Builder
from parser_model.trainer import Trainer


if __name__ == "__main__":
    # builder = Builder()
    # builder.form_trees_type_(type_="train")
    # builder.form_trees_type_(type_="dev")
    # builder.form_trees_type_(type_="test")
    test_desc = argv[1] if len(argv) >= 2 else "no message."
    trainer = Trainer(train_trees_path=RST_TRAIN_ELMo_TREES, dev_trees_path=RST_DEV_ELMo_TREES,
                      test_trees_path=RST_TEST_ELMo_TREES)
    try:
        trainer.train_(test_desc)
    except KeyboardInterrupt:
        trainer.save_eval_data()
