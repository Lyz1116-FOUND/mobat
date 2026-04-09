# gorge_chase

`gorge_chase` 是一个基于 KaiwuDRL 的《峡谷追猎》项目仓库，当前主线已经切到 `agent_diy/`，并通过 `conf/app_conf_gorge_chase.toml` 默认选择 `diy` 算法。

## 主要目录

- `agent_diy/`: 当前主训练实现，包含特征工程、模型、PPO 算法和训练 workflow。
- `agent_ppo/`: 保留的 PPO 基线实现。
- `conf/`: 应用配置、算法装配和训练系统参数。
- `docs/`: 设计说明、实现总结和训练指南。
- `tools/`: 训练环境相关辅助脚本。
- `train_test.py`: 本地或容器环境下的训练测试入口。

## 当前默认训练算法

当前配置默认使用 `diy`：

```toml
[gorge_chase.policies.train_one]
algo = "diy"
```

## 参考文档

- `docs/diy_agent_research_and_design.md`
- `docs/diy_implementation_summary.md`
- `docs/kaiwu_gorge_chase_training_guide.md`
