# DIY 智能体研究笔记：为什么这套实现应当优于当前 PPO 基线

## 1. 目标

本文档回答四个问题：

1. 为什么不直接照抄当前 `agent_ppo/`。
2. 为什么 `diy` 版本仍然保留 PPO 主体，而不是贸然换成别的算法。
3. 数学上如何把“生存 + 宝箱得分”改写成更适合训练的奖励，同时尽量不偏离原任务目标。
4. 这次 `agent_diy/` 里具体做了哪些结构性增强。

---

## 2. 先说结论

相较当前 `agent_ppo/` 基线，这次 `agent_diy/` 的提升不是来自“神奇的新 RL 名词”，而是来自更贴近题目本身的三个设计：

1. **动作空间补齐到 16 维**：把闪现真正纳入策略学习。
2. **观测从 40 维简化向量升级为局部地图 + 结构化向量 + 动作先验**：让模型真正具备“局部路径规划”能力。
3. **奖励从纯生存塑形改为“得分增量 + potential-based shaping”**：既对齐真实目标，又保持足够稠密。

这三点加在一起，理论上比当前 PPO 基线更合理，工程上也更适合这道题。

---

## 3. 外部研究结论

### 3.1 PPO 仍然是合适的主体

PPO 的优点不是“最先进”，而是它对这种离散动作、局部观测、样本噪声较大的环境足够稳。

参考：

- Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347  
  链接：[arXiv](https://arxiv.org/abs/1707.06347)

这篇工作的核心点是：

- 用 clipped surrogate objective 限制每次策略更新幅度。
- 比 TRPO 更容易实现。
- 经验上有较好的 sample efficiency 和稳定性。

对当前题目来说，这意味着：

- 我们没必要为了“看起来更高级”而放弃 PPO。
- 更有效的路线是保留 PPO 主干，把观测、奖励和动作建模做对。

### 3.2 GAE 依然值得保留

参考：

- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, arXiv:1506.02438  
  链接：[arXiv](https://arxiv.org/abs/1506.02438)

GAE 的关键价值是：

- 降低策略梯度估计的方差。
- 用可控偏差换更稳定的训练。

当前题目有长时程目标：

- 最多 1000 步。
- 生存收益和宝箱收益混合。
- 怪物、buff、闪现会带来明显的 delayed effect。

因此，GAE 比简单 Monte-Carlo return 或一步 TD 更适合。

### 3.3 合法动作掩码是有理论依据的

参考：

- Huang and Ontañón, *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*, arXiv:2006.14171  
  链接：[arXiv](https://arxiv.org/abs/2006.14171)

这篇工作说明：

- 在策略梯度类算法中，对非法动作做 masking 是有理论依据的。
- 当无效动作比例较高时，masking 会显著影响训练效果。

这对当前题目非常关键，因为：

- 真实动作空间是 16 维。
- `8-15` 是闪现动作，冷却期里往往非法。
- 如果不做正确 masking，策略会浪费大量概率质量在当前不可执行动作上。

### 3.4 Reward shaping 必须尽量用 potential-based 形式

参考：

- Ng, Harada, Russell, *Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping*  
  链接：[Andrew Ng 官方页面](https://wordpress.andrewng.org/index.php/publication/policy-invariance-under-reward-transformations-theory-and-application-to-reward-shaping/)

核心结论：

- 只有形如  
  `F(s, s') = gamma * Phi(s') - Phi(s)`  
  的 shaping 项，才能在一般条件下保证不改变最优策略集合。

这给当前题目一个非常明确的实践建议：

- 可以加 shaping。
- 但不要乱加“经验奖励”把原目标带偏。
- 最好的做法是：先用环境真实目标构造 base reward，再叠加 potential-based shaping。

---

## 4. 问题诊断：为什么当前 PPO 基线还不够

当前 `agent_ppo/` 的主要不足：

### 4.1 没用闪现

环境真实动作空间是 16 维，但当前 PPO 基线只输出 8 个移动动作。

这直接丢掉了最关键的策略杠杆之一：

- 危险脱困
- 穿越复杂地形
- 路径收集宝箱 / buff

### 4.2 局部地图建模太弱

当前 PPO 只使用了从 `21x21` 视野中心裁出来的 `4x4` 通行性。

这会带来两个问题：

- 看不清局部地形拓扑。
- 很难分辨“安全通道”和“即将走进死角”。

而这个题本质上是局部路径规划问题，地图结构理解比普通低维控制更重要。

### 4.3 奖励没有直接对齐总分

当前 PPO 主要靠：

- 每步小生存奖励
- 与怪物距离塑形

这虽然能学会“别送死”，但还不够好地表达：

- 宝箱到底值不值得绕路拿
- buff 什么时候值钱
- 生存与拿分的权衡

---

## 5. 数学推导：奖励应该怎么改

### 5.1 先把真实任务目标写出来

环境总分定义为：

```text
total_score = step_score + treasure_score
step_score = finished_steps × 1.5
treasure_score = treasures_collected × 100
```

如果把每一步环境状态记作 `s_t`，当前总分记作 `S_t`，那么最自然的基础奖励应该是：

```text
r_base(t) = S_{t+1} - S_t
```

因为：

```text
sum_t r_base(t) = sum_t (S_{t+1} - S_t) = S_T - S_0
```

而任务开始时 `S_0 = 0`，所以：

```text
sum_t r_base(t) = S_T
```

也就是说：

- 若我们直接用总分增量做 reward，
- 那么累计 reward 和最终总分是一致的。

这是最干净的目标对齐方式。

### 5.2 为什么还需要 shaping

虽然 `r_base` 已经目标对齐，但它仍然有两个问题：

1. 每步只给固定的步数分，区分不了“安全存活”和“危险乱跑”。
2. 宝箱和 buff 的价值依赖局部几何关系，直接靠 sparse score 学会，样本效率会差。

所以需要一个状态势函数 `Phi(s)` 来补充局部密集信号。

### 5.3 Potential-based shaping

定义：

```text
F(s_t, s_{t+1}) = gamma * Phi(s_{t+1}) - Phi(s_t)
```

则总奖励可以写成：

```text
r'(t) = r_base(t) / c + F(s_t, s_{t+1})
```

其中 `c` 是缩放常数，用来把分数量级调到更适合 PPO 的范围。

这次实现里取：

```text
c = 50
```

于是：

- 每步生存分 `1.5` 变成 `0.03`
- 每拿一个宝箱 `100` 变成 `2.0`

这个量级对 PPO 更友好。

### 5.4 这次使用的势函数

定义：

```text
Phi(s) = 0.50 * Safe(s)
       + 0.20 * Treasure(s)
       + 0.10 * Buff(s)
       + 0.20 * Mobility(s)
```

其中：

- `Safe(s)`：离怪物越远越大。
- `Treasure(s)`：最近宝箱越近越大。
- `Buff(s)`：当英雄当前缺速且怪物威胁较高时，最近 buff 越近越大。
- `Mobility(s)`：当前位置周围可通行区域越宽阔越大。

于是 shaping 奖励是：

```text
r_shape(t) = gamma * Phi(s_{t+1}) - Phi(s_t)
```

总训练奖励：

```text
r_train(t) = (S_{t+1} - S_t) / 50 + gamma * Phi(s_{t+1}) - Phi(s_t)
```

### 5.5 为什么这比“手工加一堆 bonus”更稳

因为这种写法有两个优点：

1. 主目标仍然是环境真实总分。
2. shaping 主要负责加速学习，而不是偷偷改题目。

这比直接拍脑袋加：

- “离怪物远 +0.2”
- “看见宝箱 +0.3”
- “闪现就奖励”

要稳得多。

---

## 6. 数学推导：为什么把 gamma 提高到 0.999

这个任务的最大 horizon 大约是 `H = 1000`。

如果 `gamma` 太小，比如 `0.99`，那么第 1000 步的权重是：

```text
0.99^1000 ≈ 0.000043
```

这会严重低估后期生存收益。

而如果取：

```text
gamma = 0.999
```

那么：

```text
0.999^1000 ≈ 0.368
```

这意味着：

- 远期收益仍然会被折扣，
- 但不会被迅速压到几乎不可见。

对于“最长 1000 步”的生存任务，这比 `0.99` 更合理。

直觉上也可以用：

```text
gamma ≈ 1 - 1 / H
```

当 `H = 1000` 时：

```text
gamma ≈ 0.999
```

---

## 7. 数学推导：为什么要引入 action prior

### 7.1 题目本质上是局部路径规划

在这个环境里，很多优劣其实可以通过**一步前瞻**估出来：

- 这个动作会不会撞进死角？
- 会不会更接近怪物？
- 会不会踩到宝箱？
- 会不会顺路拿到 buff？
- 闪现落点是否更开阔？

因此，把“局部启发式一跳评估”作为 policy 的先验是合理的。

### 7.2 形式化写法

令神经网络输出的学习型 logits 为：

```text
f_theta(s, a)
```

令手工构造的一步启发式为：

```text
u(s, a)
```

则最终策略使用：

```text
pi_theta(a|s) ∝ exp( f_theta(s, a) + alpha * u(s, a) )
```

这次实现里：

```text
alpha = 0.75
```

`u(s, a)` 主要由下面几项组成：

```text
u(s, a)
= 1.8 * clearance_after(a)
+ 1.2 * treasure_on_path(a)
+ 0.8 * buff_on_path(a)
+ 0.6 * openness_after(a)
- 1.5 * threat_after(a)
- 1.0 * deadend_after(a)
- 1.2 * stay_penalty(a)
- flash_cost(a)
```

再用 `tanh` 压缩到 `[-1, 1]`。

### 7.3 为什么这不会破坏 PPO

因为：

- `u(s, a)` 只依赖当前状态，不依赖模型参数。
- 它被并入了 policy 的 logits 定义中。
- rollout 和 learner 前向都使用同一个策略表达式。

所以从优化角度看，这仍然是在优化一个定义良好的参数化策略，只是加入了结构性 inductive bias。

这和“推理时偷偷手工改动作、训练时却不用同一策略”不同；后者会造成 rollout policy 和 learner policy 不一致。

---

## 8. 为什么这次观测结构更适合题目

这次 `diy` 观测由三部分拼起来：

1. **结构化向量特征**
2. **局部四通道地图**
3. **16 维动作先验**

### 8.1 局部四通道地图

四个通道分别是：

1. 可通行性 `occupancy`
2. 宝箱位置 `treasure`
3. buff 位置 `buff`
4. 怪物威胁热力图 `monster_threat`

相较当前 PPO 的 `4x4` 通行块，这种表示更像一个真正的 egocentric local planner 输入。

### 8.2 为什么用 CNN

因为这个问题里：

- 障碍物布局有空间结构。
- 通道、死角、拐点、开阔地是局部几何模式。
- 这些模式对“往哪走”和“要不要闪现”都很关键。

单纯把少量几何摘要塞进 MLP，不如让 CNN 直接看局部地图。

---

## 9. 这次 DIY 实现的关键改动

### 9.1 动作空间

- 从模板状态升级到完整 `16` 动作。
- 正确支持 flash mask。
- 样本中的 `prob`、`legal_action`、模型输出同步扩到 16 维。

### 9.2 特征工程

新增了：

- 英雄状态特征
- 两只怪物的结构化相对特征
- 最近宝箱 / buff 特征
- 局部开阔度、射线距离等几何摘要
- `21x21x4` 局部地图通道
- 面向 16 个动作的一步前瞻启发式先验

### 9.3 奖励

改成：

```text
score_delta / 50 + potential_based_shaping
```

不再沿用当前 PPO 工作流里那种额外的终局 `+10 / -10` 人工奖励。

### 9.4 模型

改为：

- `vector encoder` 处理结构化特征
- `CNN map encoder` 处理局部地图
- `fusion MLP` 融合
- actor logits 再加上 `action prior bias`
- critic 输出状态价值

### 9.5 算法

相较当前 PPO 基线，`diy` 里做了这些增强：

- 完整 16 动作 masked PPO
- GAE
- advantage normalization
- clipped value loss
- multi-epoch mini-batch PPO update

### 9.6 监控

补了更贴近任务的指标：

- `score_reward`
- `shaping_reward`
- `potential`
- `prior_mean`
- `total_score`
- `finished_steps`
- `treasures_collected`
- `flash_count`
- `collected_buff`

---

## 10. 为什么这套设计预期会优于当前 PPO 基线

### 10.1 信息利用更完整

当前基线：

- 不学闪现
- 地图看得很粗
- 宝箱和 buff 信息没吃满

DIY 方案：

- 动作更完整
- 地图更完整
- 目标信息更完整

### 10.2 奖励更对齐最终分数

当前基线偏向“活着就行”。

DIY 方案直接围绕：

- 步数得分
- 宝箱得分
- 局部风险

展开。

### 10.3 策略先验更强

当前基线几乎全靠网络自己摸索“哪一步安全、哪一步危险”。

DIY 方案把一跳局部规划知识显式编码进 `action prior`，这会显著提升 sample efficiency。

### 10.4 对长时程生存更友好

`gamma = 0.999` 比当前更适合 1000 步 horizon。

---

## 11. 当前验证情况与残余风险

### 已验证

- `agent_diy/` 全量 Python 语法通过。
- 容器内语法编译通过。
- 预处理输出维度正确：`1840`。
- 模型前向输出维度正确：`logits -> 16`, `value -> 1`。
- 通过框架包装后的业务方法 `_business_predict` / `_business_learn` 可执行。

### 未完全验证

- `run_train_test('diy')` 没能跑到真实训练阶段，因为当前仓库 smoke-test 链路在更早的位置就失败了：
  `tools/change_sample_server.sh` 缺失。

这说明：

- `diy` 代码本身已经接上了框架接口。
- 但仓库当前的端到端 smoke-test 基础设施并不完整，无法据此完成最终端到端验收。

---

## 12. 后续最值得继续优化的地方

如果继续在 `diy` 上冲更高分，建议优先顺序如下：

1. 把 `action prior` 从“一步前瞻”升级到“两步前瞻”或轻量 search。
2. 对宝箱密集区和死角区域增加更强的局部拓扑特征。
3. 引入课程式环境配置，从易到难逐步增加第二怪物和复杂地形压力。
4. 针对 flash 单独增加更精细的风险收益估计。
5. 在 `diy` 上做更系统的超参数搜索，比如 `gamma / lambda / entropy_coef / prior_scale`。

---

## 13. 结论

这次 `agent_diy/` 的核心思想可以浓缩成一句话：

> 保留 PPO 这条最稳的优化主线，但把动作、观测、奖励和局部规划先验全部按《峡谷追猎》这个具体问题重写一遍，让模型真正学的是“局部地图生存规划 + 宝箱收益权衡”，而不是一个只会随机游走求生的简化 baseline。

因此，这套 DIY 方案预期优于当前 `agent_ppo/`，不是因为它名字更花，而是因为它更贴题。
