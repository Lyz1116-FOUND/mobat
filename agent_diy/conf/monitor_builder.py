#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Monitor panel configuration for the DIY agent.
DIY 智能体的监控面板配置。
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("峡谷追猎 DIY")
        .add_group(group_name="算法指标", group_name_en="algorithm")
        .add_panel(name="训练奖励", name_en="reward", type="line")
        .add_metric(metrics_name="reward", expr="avg(reward{})")
        .add_metric(metrics_name="score_reward", expr="avg(score_reward{})")
        .add_metric(metrics_name="shaping_reward", expr="avg(shaping_reward{})")
        .end_panel()
        .add_panel(name="损失", name_en="loss", type="line")
        .add_metric(metrics_name="total_loss", expr="avg(total_loss{})")
        .add_metric(metrics_name="policy_loss", expr="avg(policy_loss{})")
        .add_metric(metrics_name="value_loss", expr="avg(value_loss{})")
        .add_metric(metrics_name="entropy_loss", expr="avg(entropy_loss{})")
        .end_panel()
        .add_panel(name="先验与势函数", name_en="priors", type="line")
        .add_metric(metrics_name="potential", expr="avg(potential{})")
        .add_metric(metrics_name="prior_mean", expr="avg(prior_mean{})")
        .end_panel()
        .end_group()
        .add_group(group_name="环境指标", group_name_en="environment")
        .add_panel(name="得分", name_en="score", type="line")
        .add_metric(metrics_name="total_score", expr="avg(total_score{})")
        .add_metric(metrics_name="finished_steps", expr="avg(finished_steps{})")
        .add_metric(metrics_name="treasures_collected", expr="avg(treasures_collected{})")
        .end_panel()
        .add_panel(name="技能与增益", name_en="skill_buff", type="line")
        .add_metric(metrics_name="flash_count", expr="avg(flash_count{})")
        .add_metric(metrics_name="collected_buff", expr="avg(collected_buff{})")
        .add_metric(metrics_name="monster_speed", expr="avg(monster_speed{})")
        .add_metric(metrics_name="monster_interval", expr="avg(monster_interval{})")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
