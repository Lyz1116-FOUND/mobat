#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Risk-aware PPO for the DIY Gorge Chase agent.
DIY 峡谷追猎智能体的风险感知 PPO 算法。
"""

import os
import time

import torch

from agent_diy.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.monitor = monitor
        self.parameters = [p for group in self.optimizer.param_groups for p in group["params"]]

        self.last_report_monitor_time = 0.0
        self.train_step = 0

    def learn(self, list_sample_data):
        obs = torch.stack([sample.obs for sample in list_sample_data]).to(self.device)
        legal_action = torch.stack([sample.legal_action for sample in list_sample_data]).to(self.device)
        act = torch.stack([sample.act for sample in list_sample_data]).to(self.device).long().view(-1, 1)
        old_prob = torch.stack([sample.prob for sample in list_sample_data]).to(self.device)
        reward = torch.stack([sample.reward for sample in list_sample_data]).to(self.device)
        advantage = torch.stack([sample.advantage for sample in list_sample_data]).to(self.device)
        old_value = torch.stack([sample.value for sample in list_sample_data]).to(self.device)
        reward_sum = torch.stack([sample.reward_sum for sample in list_sample_data]).to(self.device)
        score_reward = torch.stack([sample.score_reward for sample in list_sample_data]).to(self.device)
        shaping_reward = torch.stack([sample.shaping_reward for sample in list_sample_data]).to(self.device)
        potential = torch.stack([sample.potential for sample in list_sample_data]).to(self.device)

        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)

        batch_size = obs.shape[0]
        mb_size = min(Config.MINI_BATCH_SIZE, batch_size)

        metrics = {
            "total_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "reward": float(reward.mean().item()),
            "score_reward": float(score_reward.mean().item()),
            "shaping_reward": float(shaping_reward.mean().item()),
            "potential": float(potential.mean().item()),
            "prior_mean": float(obs[:, -Config.ACTION_NUM :].mean().item()),
        }
        update_count = 0

        self.model.set_train_mode()
        for _ in range(Config.PPO_EPOCHS):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mb_size):
                index = permutation[start : start + mb_size]

                logits, value_pred = self.model(obs[index])
                prob_dist = self._masked_softmax(logits, legal_action[index])

                new_act_prob = prob_dist.gather(1, act[index]).clamp_min(1e-8)
                old_act_prob = old_prob[index].gather(1, act[index]).clamp_min(1e-8)

                ratio = new_act_prob / old_act_prob
                surr1 = ratio * advantage[index]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - Config.CLIP_PARAM,
                    1.0 + Config.CLIP_PARAM,
                ) * advantage[index]
                policy_loss = -torch.min(surr1, surr2).mean()

                clipped_value = old_value[index] + (value_pred - old_value[index]).clamp(
                    -Config.CLIP_PARAM,
                    Config.CLIP_PARAM,
                )
                value_loss_unclipped = torch.square(value_pred - reward_sum[index])
                value_loss_clipped = torch.square(clipped_value - reward_sum[index])
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_loss = -(prob_dist * torch.log(prob_dist.clamp_min(1e-8))).sum(dim=1).mean()
                total_loss = (
                    policy_loss
                    + Config.VALUE_LOSS_COEF * value_loss
                    - Config.ENTROPY_COEF * entropy_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
                self.optimizer.step()

                metrics["total_loss"] += float(total_loss.item())
                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy_loss"] += float(entropy_loss.item())
                update_count += 1

        self.train_step += 1
        if update_count > 0:
            for key in ("total_loss", "policy_loss", "value_loss", "entropy_loss"):
                metrics[key] /= update_count

        self._report_metrics(metrics)
        return metrics

    def _masked_softmax(self, logits, legal_action):
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        label = logits - label_max
        label = label * legal_action
        label = label + 1e5 * (legal_action - 1.0)
        return torch.nn.functional.softmax(label, dim=1)

    def _report_metrics(self, metrics):
        now = time.time()
        if now - self.last_report_monitor_time < 30:
            return

        if self.logger:
            self.logger.info(
                "[DIY train] loss:%.4f policy:%.4f value:%.4f entropy:%.4f reward:%.4f score_r:%.4f shape_r:%.4f"
                % (
                    metrics["total_loss"],
                    metrics["policy_loss"],
                    metrics["value_loss"],
                    metrics["entropy_loss"],
                    metrics["reward"],
                    metrics["score_reward"],
                    metrics["shaping_reward"],
                )
            )

        if self.monitor:
            self.monitor.put_data({os.getpid(): metrics})
        self.last_report_monitor_time = now
