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

## 项目贡献与实现证据

本项目基于腾讯开悟 KaiwuDRL 环境与接口，框架、原有基线和第三方材料保留原署名及版权信息；这里展示的是在该基础上的 DIY Agent 实践，不代表独立开发了 KaiwuDRL。

- **动作与观测**：16 个动作（8 方向移动 + 8 方向闪现）；1840 维输入由 60 维向量、4 × 21 × 21 局部地图和 16 维动作先验组成。见 [配置](agent_diy/conf/conf.py) 与 [特征预处理](agent_diy/feature/preprocessor.py)。
- **策略与训练**：CNN + MLP 融合 Actor-Critic、GAE、合法动作掩码 PPO、potential-based reward shaping。见 [模型](agent_diy/model/model.py)、[算法](agent_diy/algorithm/algorithm.py) 与 [样本处理](agent_diy/feature/definition.py)。
- **运行链路**：通过 Actor / Aisrv / Learner 接入采样、训练和模型保存；[实现说明](docs/diy_implementation_summary.md) 保留了训练验证记录。
- **结果边界**：训练链路跑通不等于策略性能已优于基线；本仓库不提供系统性调参或对比胜率结论。运行依赖 KaiwuDRL 配套环境，单独克隆本仓库不足以直接启动完整训练服务。

## 文档

- [DIY Agent 设计](docs/diy_agent_research_and_design.md)
- [实现与验证记录](docs/diy_implementation_summary.md)
- [KaiwuDRL 训练指南](docs/kaiwu_gorge_chase_training_guide.md)
