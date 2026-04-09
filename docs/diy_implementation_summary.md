# DIY 实现说明

## 1. 这次到底实现了什么

这次工作的目标不是在 `agent_ppo/` 上做小修小补，而是**只在 `agent_diy/` 里重写一套更贴合《峡谷追猎》问题结构的可训练智能体**，并且把它真正接进当前 `podman` 环境下的 KaiwuDRL 训练链路。

最终落地结果可以概括成三句话：

1. `agent_diy/` 已经从“模板壳子”补成了**可训练、可推理、可上报监控**的一整套实现。
2. 这套实现不是简单复制 `agent_ppo/`，而是围绕题目本身重构了**动作空间、特征工程、奖励设计、模型结构、PPO 更新和工作流**。
3. 代码不只是“能 import”，而是已经在 `podman` 里的真实训练链路上跑到过**环境交互、episode 结束、Learner 更新和模型保存**。

---

## 2. 设计目标

这套 `diy` 版本的核心目标有四个：

1. **补齐环境真实能力**  
   不再只用 8 个移动动作，而是完整使用 16 个动作，把闪现纳入训练。

2. **让观测更像局部路径规划问题**  
   不再只看很粗的简化向量，而是让模型看到局部地图、宝箱、buff、怪物威胁和动作先验。

3. **让 reward 更对齐真实总分**  
   用环境总分增量做基础奖励，再叠加有理论依据的 potential-based shaping。

4. **保持和 KaiwuDRL 框架兼容**  
   不绕过框架，不自己造另一套训练器，而是按当前 `podman` 环境里的 `Actor / Aisrv / Learner` 运行方式接进去。

---

## 3. 改动范围

### 3.1 核心改动文件

| 文件 | 作用 |
| --- | --- |
| `agent_diy/conf/conf.py` | 重定义观测维度、动作维度、PPO 超参数 |
| `agent_diy/feature/definition.py` | 重写 `ObsData / ActData / SampleData`，加入 GAE 后处理 |
| `agent_diy/feature/preprocessor.py` | 新增完整预处理器，负责特征构造、动作先验和奖励塑形 |
| `agent_diy/model/model.py` | 新增 CNN + MLP 融合策略价值网络 |
| `agent_diy/algorithm/algorithm.py` | 重写 PPO 训练逻辑 |
| `agent_diy/agent.py` | 重写智能体主类，接通 predict / exploit / learn / load/save model |
| `agent_diy/workflow/train_workflow.py` | 重写训练工作流，负责 episode 收集与样本发送 |
| `agent_diy/conf/monitor_builder.py` | 重写 DIY 监控面板 |

### 3.2 为了打通 smoke test 额外补的兼容层

| 文件 | 作用 |
| --- | --- |
| `tools/__init__.py` | 让本仓库下的 `tools.*` 导入和容器内框架工具兼容 |
| `tools/change_sample_server.sh` | 代理到容器内官方脚本 |
| `tools/modelpool_start.sh` | 代理到容器内官方脚本 |
| `tools/stop.sh` | 代理到容器内官方脚本 |
| `tools/stop_sim_python.sh` | 做最佳努力清理，兼容部分镜像无此脚本的情况 |
| `train_test.py` | 增加容器内 learner 配置路径回退 |

### 3.3 没动的东西

- **没有改 `agent_ppo/` 的算法逻辑**
- **没有把项目主线强行切到 `diy`**
- **没有改比赛环境实现**

也就是说，这次改动是**增量式新增一套 `diy` 能力**，不是破坏当前 PPO 基线。

---

## 4. 在 `agent_diy/` 里具体实现了什么

## 4.1 动作空间：从模板状态升级到完整 16 动作

环境真实动作空间是：

- `0-7`：八方向移动
- `8-15`：八方向闪现

这次 `diy` 里：

- 把动作维度设为 `16`
- legal action mask 也扩成 `16`
- 策略网络输出 `16` 维 logits
- `SampleData.prob` 同步扩成 `16` 维

这意味着 `diy` 真正开始学习：

- 什么时候闪现
- 往哪个方向闪现
- 闪现和普通移动怎么配合

这点是它相对当前 `agent_ppo/` 的最大结构性提升之一。

---

## 4.2 特征工程：从小向量升级为“结构化向量 + 局部地图 + 动作先验”

### 当前 `diy` 观测总维度

总输入维度是 `1840`，由三部分构成：

| 部分 | 维度 | 含义 |
| --- | --- | --- |
| 向量特征 | `60` | 英雄、怪物、宝箱、buff、环境统计、局部几何摘要 |
| 局部地图特征 | `4 × 21 × 21 = 1764` | 可通行性、宝箱、buff、怪物威胁热力图 |
| 动作先验 | `16` | 面向每个动作的一步前瞻启发式评分 |

### 向量特征里包含了什么

主要包含：

- 英雄位置、步数进度、闪现冷却、buff 剩余时间
- 两只怪物的相对位置、距离、方向、速度、危险近距标记
- 最近宝箱的方向与距离
- 最近 buff 的方向与距离
- `total_score / step_score / treasure_score`
- `treasures_collected / collected_buff / flash_count`
- `monster_interval / monster_speed`
- 局部开阔度、可逃逸邻居数、八方向射线距离

### 局部地图通道是什么

四个通道分别是：

1. `occupancy`：可通行区域
2. `treasure_channel`：视野内宝箱位置
3. `buff_channel`：视野内 buff 位置
4. `monster_threat`：根据怪物位置和速度构建的局部威胁热力图

这让模型不只是“知道有怪物”，而是更像在做**局部栅格路径规划**。

### 动作先验是什么

这次额外实现了一个对 16 个动作的一步前瞻启发式 `action_priors`，它会综合评估：

- 该动作后的怪物安全边际
- 路径上是否能吃到宝箱
- 路径上是否能吃到 buff
- 落点附近是否开阔
- 是否会走进死角
- 是否原地不动
- 闪现是否值得付出冷却代价

然后把这个先验直接加到 actor logits 上，作为结构性 bias。

这不是硬编码动作，而是给策略网络一个更像“局部规划器”的 inductive bias。

---

## 4.3 奖励设计：改成“总分增量 + potential-based shaping”

### 基础奖励

基础奖励不是再拍脑袋给固定 bonus，而是直接用：

```text
score_reward = (total_score_t - total_score_{t-1}) / 50
```

这样 reward 主体就直接对齐环境总分：

- 活得更久有收益
- 拿到宝箱有大收益

### 塑形奖励

塑形项写成了理论上更稳的形式：

```text
shaping_reward = gamma * Phi(s') - Phi(s)
```

其中势函数 `Phi(s)` 综合了：

- 安全性 `Safe`
- 宝箱接近度 `Treasure`
- buff 紧急性与接近度 `Buff`
- 局部机动性 `Mobility`

因此总训练奖励是：

```text
reward = score_reward + shaping_reward
```

### 这样做的意义

它兼顾了两点：

1. 主目标仍然是环境真实总分
2. 训练时又比纯 sparse score 更容易学

---

## 4.4 模型结构：CNN + MLP 融合 Actor-Critic

这次没有沿用“只有小 MLP”的模板，而是实现了一个更适合局部地图任务的网络：

```text
向量特征 -> vector encoder (MLP)
局部地图 -> map encoder (CNN)
两路特征拼接 -> fusion MLP
                    -> actor head (16)
                    -> critic head (1)
```

同时：

- actor logits 上叠加了动作先验 `action_prior`
- critic 输出单头 value

这让模型同时具备：

- 结构化统计感知能力
- 局部空间几何理解能力

---

## 4.5 算法：重写成完整的 masked PPO

当前 `diy` 里的 PPO 包含这些核心部分：

- legal action masking
- GAE
- advantage normalization
- clipped surrogate objective
- clipped value loss
- entropy regularization
- gradient clipping
- 多轮 `PPO_EPOCHS`
- mini-batch 更新

也就是说，这不是“随便写个 learn 函数能跑就算了”，而是一套完整的 PPO 训练实现。

---

## 4.6 工作流：按当前分布式链路采样和送训

当前 `diy` 工作流做了这些事：

1. 读取 `agent_diy/conf/train_env_conf.toml`
2. reset 环境
3. reset agent
4. load 最新模型
5. 与环境循环交互
6. 每步构造 `SampleData`
7. episode 结束后做 `sample_process()`
8. 调用 `agent.send_sample_data(g_data)` 发给 Learner
9. 定期保存模型

这条链路是和当前 KaiwuDRL 的 `Aisrv -> Learner` 工作方式对齐的，不是绕过框架单独造了一套本地 trainer。

---

## 4.7 监控：补齐 DIY 视角下最有用的指标

除了常见的：

- `total_loss`
- `policy_loss`
- `value_loss`
- `entropy_loss`

还额外上报了：

- `reward`
- `score_reward`
- `shaping_reward`
- `potential`
- `prior_mean`
- `total_score`
- `finished_steps`
- `treasures_collected`
- `flash_count`
- `collected_buff`
- `monster_speed`
- `monster_interval`

这能帮助你区分：

- 是 loss 在变好
- 还是真实得分在上涨

---

## 5. 还额外修了什么环境兼容问题

最开始 `run_train_test('diy')` 并没有直接跑起来，先卡在了项目仓库级的 smoke-test 基础设施上。

### 问题 1：本仓库没有 `tools/change_sample_server.sh`

`run_train_test` 会在当前工作目录里执行相对路径脚本，例如：

- `tools/change_sample_server.sh`
- `tools/modelpool_start.sh`
- `tools/stop.sh`

但当前仓库原本没有这些脚本。

### 解决方式

我新增了一个最小 `tools/` 兼容层，把这些脚本代理到容器里已有的官方脚本路径：

```text
/data/projects/gorge_chase/tools/...
```

这样做的目的不是复制整套框架，而是补齐这几个 smoke test 需要的入口。

### 问题 2：`train_test.py` 默认的 `conf_path` 在当前仓库里不存在

默认逻辑会找：

```text
kaiwudrl/conf/kaiwudrl/learner.toml
```

但本仓库根目录下没有这一套框架配置目录。

### 解决方式

我给 [train_test.py](/Users/foundtt/my_cow_horse/code/train_test.py:20) 加了容器回退逻辑：

- 如果容器内存在 `/data/projects/gorge_chase/kaiwudrl/conf/kaiwudrl/learner.toml`
- 就优先用这个绝对路径

这样在 `podman` 里直接跑 `python train_test.py` 时就不会先死在配置解析上。

---

## 6. 具体验证了哪些东西

这次不是只做静态代码改动，也做了多层验证。

### 6.1 语法与导入验证

已通过：

- 宿主机 `py_compile`
- `podman` 容器内 `py_compile`

### 6.2 预处理维度验证

在容器里构造 fake obs 后确认：

- `feature_dim = 1840`
- `legal_action_dim = 16`

### 6.3 模型前向验证

在容器里直接前向确认：

- `logits.shape = (2, 16)`
- `value.shape = (2, 1)`

### 6.4 框架包装业务方法验证

由于 KaiwuDRL 会给 `Agent` 方法做包装，我额外验证了：

- `Agent._business_predict(...)`
- `Agent._business_learn(...)`

都能工作。

### 6.5 真实训练链路验证

在 `podman` 里执行 `run_train_test('diy')` 后，真实看到：

- `learner is alive`
- `aisrv is alive`
- `DIY Episode 1 start`
- `[DIY GAMEOVER] episode:1 steps:25 total_score:137.5 treasures:1 flash:1 reward:2.527`
- Learner 输出 `[DIY train] loss:...`
- `save model /data/ckpt/gorge_chase_diy/model.ckpt-*.pkl successfully`

这说明：

- 环境交互跑起来了
- episode 真的结束了
- 样本真的被送到了 Learner
- PPO 更新真的执行了
- 模型文件真的被保存了

### 6.6 模型产物验证

容器里 `/data/ckpt/gorge_chase_diy/` 下已经生成了大量 `model.ckpt-*.pkl` 文件，说明不是“日志打印像成功”，而是训练产物真的落地了。

---

## 7. 当前残余问题和风险

### 7.1 `run_train_test` 外层进程清理阶段有一次 `137`

在一次完整 smoke test 里，外层 `podman exec` 最后出现过 `137`。

从已经观察到的事实看：

- Learner 和 Aisrv 都跑起来了
- episode 和训练日志都出现了
- 模型文件大量生成了
- 事后进程已清理完

所以这更像是：

- 清理阶段 stop 脚本把外层命令也一起带走了

而不是 `diy` 算法本身在训练早期崩溃。

### 7.2 smoke test 环境参数会导致模型文件很多

为了尽快验证链路，当前 smoke test 用了：

- `train_batch_size = 2`
- `dump_model_freq = 1`

这会导致模型保存非常频繁，验证时很方便，但不适合作为长期训练配置。

### 7.3 现在还没有做系统性调参

当前实现重心是：

- 把结构搭起来
- 接通真实链路

还没进入真正的：

- 超参数搜索
- 奖励权重搜索
- 环境配置课程化训练

所以“已经比当前 PPO 基线更贴题”是成立的，但“已经是最优配置”还远远不是。

---

## 8. 你后面最应该从哪继续

如果你接下来自己接着做，我建议优先顺序如下：

1. 先把 `train_test.py` 里的算法切到 `diy`，做更长一点的真实训练。
2. 观察监控里 `total_score / finished_steps / treasures_collected / flash_count` 的走势。
3. 第一轮重点调：
   - `gamma`
   - `ENTROPY_COEF`
   - `ACTION_PRIOR_SCALE`
   - `SCORE_REWARD_SCALE`
4. 如果发现过度保守，只会逃命不拿箱子，就加大宝箱相关势函数权重。
5. 如果发现乱闪现或乱冲死角，就加大动作先验里的死角惩罚和威胁惩罚。

---

## 9. 配套文档

这次还额外整理了两份文档，后面都可以直接配合这份实现说明一起看：

- [docs/kaiwu_gorge_chase_training_guide.md](./kaiwu_gorge_chase_training_guide.md)  
  讲题目规则、KaiwuDRL 训练流程和当前仓库映射。

- [docs/diy_agent_research_and_design.md](./diy_agent_research_and_design.md)  
  讲为什么这么设计、参考了哪些论文、奖励和结构推导是什么。

---

## 10. 最后一句总结

这次实现的本质，不是“把 `diy` 目录补全到能跑”，而是：

> 在不动 `agent_ppo` 主线的前提下，给当前项目新增了一套更贴合《峡谷追猎》问题结构的 `diy` 智能体，它已经具备完整的 16 动作控制、局部地图建模、理论支撑的奖励塑形、风险感知 PPO 训练能力，并且已经在 `podman` 的真实训练链路里跑通了环境交互、Learner 更新和模型保存。
