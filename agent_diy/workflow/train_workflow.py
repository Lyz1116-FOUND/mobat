#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for the DIY Gorge Chase agent.
DIY 峡谷追猎智能体训练工作流。
"""

import os
import time

import numpy as np

from agent_diy.feature.definition import SampleData, sample_process
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    env = envs[0]
    agent = agents[0]
    last_save_model_time = time.time()

    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_metrics_time = 0.0
        self.last_monitor_time = 0.0

    def run_episodes(self):
        while True:
            now = time.time()
            if now - self.last_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_metrics_time = now
                if training_metrics is not None and self.logger:
                    self.logger.info(f"training_metrics is {training_metrics}")

            env_obs = self.env.reset(self.usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            obs_data, _ = self.agent.observation_process(env_obs)
            collector = []
            done = False
            step = 0
            self.episode_cnt += 1

            total_reward = 0.0
            total_score_reward = 0.0
            total_shaping_reward = 0.0

            if self.logger:
                self.logger.info(f"[DIY] Episode {self.episode_cnt} start")

            while not done:
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                _, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = bool(env_obs.get("terminated", False))
                truncated = bool(env_obs.get("truncated", False))
                done = terminated or truncated
                step += 1

                next_obs_data, remain_info = self.agent.observation_process(env_obs)

                reward = np.array(remain_info.get("reward", [0.0]), dtype=np.float32)
                score_reward = np.array(remain_info.get("score_reward", [0.0]), dtype=np.float32)
                shaping_reward = np.array(remain_info.get("shaping_reward", [0.0]), dtype=np.float32)
                potential = np.array(remain_info.get("potential", [0.0]), dtype=np.float32)

                total_reward += float(reward[0])
                total_score_reward += float(score_reward[0])
                total_shaping_reward += float(shaping_reward[0])

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    reward_sum=np.zeros(1, dtype=np.float32),
                    done=np.array([float(done)], dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                    score_reward=score_reward,
                    shaping_reward=shaping_reward,
                    potential=potential,
                )
                collector.append(frame)

                if done:
                    env_info = env_obs["observation"]["env_info"]
                    if self.logger:
                        self.logger.info(
                            "[DIY GAMEOVER] episode:%s steps:%s total_score:%.1f treasures:%s flash:%s reward:%.3f"
                            % (
                                self.episode_cnt,
                                step,
                                float(env_info.get("total_score", 0.0)),
                                int(env_info.get("treasures_collected", 0)),
                                int(env_info.get("flash_count", 0)),
                                total_reward,
                            )
                        )

                    if time.time() - self.last_monitor_time >= 30 and self.monitor:
                        monitor_data = {
                            "reward": round(total_reward, 4),
                            "score_reward": round(total_score_reward, 4),
                            "shaping_reward": round(total_shaping_reward, 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                            "total_score": float(env_info.get("total_score", 0.0)),
                            "finished_steps": float(env_info.get("finished_steps", step)),
                            "treasures_collected": float(env_info.get("treasures_collected", 0.0)),
                            "flash_count": float(env_info.get("flash_count", 0.0)),
                            "collected_buff": float(env_info.get("collected_buff", 0.0)),
                            "monster_interval": float(env_info.get("monster_interval", 0.0)),
                            "monster_speed": float(env_info.get("monster_speed", 0.0)),
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_monitor_time = time.time()

                    if collector:
                        yield sample_process(collector)
                    break

                obs_data = next_obs_data
