#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase DIY agent.
峡谷追猎 DIY 智能体配置。
"""


class Config:
    MAP_SIZE = 128.0
    LOCAL_MAP_SIZE = 21
    LOCAL_MAP_RADIUS = 10
    MAP_CHANNELS = 4

    VECTOR_FEATURE_DIM = 60
    IMAGE_FEATURE_DIM = MAP_CHANNELS * LOCAL_MAP_SIZE * LOCAL_MAP_SIZE
    ACTION_NUM = 16
    ACTION_PRIOR_DIM = ACTION_NUM
    VALUE_NUM = 1

    FEATURE_SPLIT_SHAPE = [
        VECTOR_FEATURE_DIM,
        IMAGE_FEATURE_DIM,
        ACTION_PRIOR_DIM,
    ]
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    MAX_DISTANCE = 180.0
    MAX_FLASH_CD = 2000.0
    MAX_BUFF_DURATION = 50.0
    MAX_MONSTER_SPEED = 5.0
    MAX_TOTAL_SCORE = 2500.0

    SCORE_REWARD_SCALE = 50.0
    ACTION_PRIOR_SCALE = 0.75

    GAMMA = 0.999
    LAMDA = 0.97
    INIT_LEARNING_RATE_START = 3e-4
    CLIP_PARAM = 0.15
    ENTROPY_COEF = 0.01
    VALUE_LOSS_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5
    PPO_EPOCHS = 4
    MINI_BATCH_SIZE = 256

    MODEL_NAME = "gorge_chase_diy_risk_aware"
