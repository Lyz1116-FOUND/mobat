#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

CNN + MLP policy/value model for the DIY Gorge Chase agent.
DIY 峡谷追猎智能体的 CNN + MLP 策略价值网络。
"""

import torch
import torch.nn as nn

from agent_diy.conf.conf import Config


def make_linear(in_features, out_features):
    layer = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(layer.weight.data)
    nn.init.zeros_(layer.bias.data)
    return layer


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device
        self.model_name = Config.MODEL_NAME

        self.vector_dim = Config.VECTOR_FEATURE_DIM
        self.image_dim = Config.IMAGE_FEATURE_DIM
        self.action_prior_dim = Config.ACTION_PRIOR_DIM

        self.vector_encoder = nn.Sequential(
            make_linear(self.vector_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            make_linear(128, 64),
            nn.ReLU(),
        )

        self.map_encoder = nn.Sequential(
            nn.Conv2d(Config.MAP_CHANNELS, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            make_linear(32 * 4 * 4, 128),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            make_linear(64 + 128, 128),
            nn.ReLU(),
            make_linear(128, 64),
            nn.ReLU(),
        )

        self.actor_head = make_linear(64, Config.ACTION_NUM)
        self.critic_head = make_linear(64, Config.VALUE_NUM)

    def forward(self, obs, inference=False):
        vector_obs = obs[:, : self.vector_dim]
        image_obs = obs[:, self.vector_dim : self.vector_dim + self.image_dim]
        action_prior = obs[:, -self.action_prior_dim :]

        image_obs = image_obs.view(-1, Config.MAP_CHANNELS, Config.LOCAL_MAP_SIZE, Config.LOCAL_MAP_SIZE)

        vector_hidden = self.vector_encoder(vector_obs)
        map_hidden = self.map_encoder(image_obs)
        fused_hidden = self.fusion(torch.cat([vector_hidden, map_hidden], dim=1))

        logits = self.actor_head(fused_hidden) + Config.ACTION_PRIOR_SCALE * action_prior
        value = self.critic_head(fused_hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
