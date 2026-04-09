#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definitions and trajectory post-processing for DIY agent.
DIY 智能体的数据结构与轨迹后处理。
"""

import numpy as np
from common_python.utils.common_func import create_cls

from agent_diy.conf.conf import Config


ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_action=None,
)


ActData = create_cls(
    "ActData",
    action=None,
    d_action=None,
    prob=None,
    value=None,
)


SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,
    reward=Config.VALUE_NUM,
    reward_sum=Config.VALUE_NUM,
    done=1,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,
    score_reward=Config.VALUE_NUM,
    shaping_reward=Config.VALUE_NUM,
    potential=Config.VALUE_NUM,
)


def sample_process(list_sample_data):
    """Fill next_value and compute GAE advantages."""
    for idx in range(len(list_sample_data) - 1):
        if float(list_sample_data[idx].done[0]) < 0.5:
            list_sample_data[idx].next_value = list_sample_data[idx + 1].value

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    gae = np.zeros((Config.VALUE_NUM,), dtype=np.float32)
    gamma = Config.GAMMA
    lamda = Config.LAMDA

    for sample in reversed(list_sample_data):
        reward = np.asarray(sample.reward, dtype=np.float32)
        value = np.asarray(sample.value, dtype=np.float32)
        next_value = np.asarray(sample.next_value, dtype=np.float32)
        done = float(np.asarray(sample.done, dtype=np.float32)[0])
        not_done = 1.0 - done

        delta = reward + gamma * not_done * next_value - value
        gae = delta + gamma * lamda * not_done * gae
        sample.advantage = gae.astype(np.float32)
        sample.reward_sum = (gae + value).astype(np.float32)
