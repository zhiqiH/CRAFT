# CRAFT × Debate

一个把 [CRAFT](https://arxiv.org/abs/2603.25268)（arXiv:2603.25268v2）benchmark 与自定义
**7-agent 多轮 Multi-Agent Debate 拓扑（Debate）** 结合的实验框架。每轮 7 个子 agent 都使用
`gpt-4o-mini`，固定 25 轮，并按论文方法评估任务进度，最后把 25 轮的分数曲线画出来。
本仓库的**最终目标**是用价值回归器学习"何时该停止辩论"的早停策略（见下文）。

## 研究目标：用价值回归器学习早停策略

在多轮 Debate 中，每轮 Judge 执行完动作后都要决定"是否进入下一轮"。目标是在质量分与
token/latency 开销之间权衡：

```text
V = score − λ · token_cost
```

最终方案是训练一个**价值回归器**：给定第 t 轮的完整状态 `s_t`，预测继续辩论能带来的
**剩余最大分数增量 Δscore(s_t)** 与 **剩余花费 Δcost(s_t)**，决策规则为：

```text
stop  ⟺  Δscore(s_t) < λ · Δcost(s_t)
```

λ 是权衡权重，可以在推理时连续调节，无需为每个 λ 重新训练。状态特征全部来自 trajectory：
当前分与增量、最近 k 轮趋势、oracle 剩余动作数、失败/clarify 情况、累计 token 花费、轮数与
结构复杂度（十几维特征，用逻辑回归/梯度提升等小模型即可）。

学习流程（前四步完全离线，不产生新的 API 开销）：

1. **采集轨迹**：craft-80 × 3 runs × 25 轮 = 240 局，每局是一段完整的 25 轮轨迹，作为数据集；
2. **特征 + 回归目标**：按每条轨迹的事后最优停止点，算出每轮的 Δscore(s_t) 与 Δcost(s_t)
   作为价值回归器的监督目标（纯离线计算）；
3. **训练**：按**结构**切分 50/12/18（训练/验证/测试），训练价值回归器；
4. **离线回放评估**：在完整轨迹上回放停止策略，与"永远跑满 25 轮"基线比较最终分、节省轮数与
   token，画 score-cost Pareto 曲线；
5. **实机接入**：把回归器接到 `run_debate.py` 每轮结束处，用少量真实运行确认离线估计。

## Debate 拓扑

每轮的结构是 `3 → 3 → 1`：

1. **Proposers（P1/P2/P3，并行）**：即论文的 D1/D2/D3，各自持有一面墙的私有 2D 视图，同时看到
   题目（当前棋盘 + 自己的目标视图）和上一轮 Judge 的综合回答，输出 `<think>` + `<message>`。
2. **Critics（C1/C2/C3，并行）**：看到题目和 P1-P3 的完整回答，从三个角度辩论批判——
   spatial grounding（空间接地）、mind modeling（心智建模）、pragmatic sufficiency（语用充分性），
   分别输出 `<critique>` + `<message>`。
3. **Judge（J1）**：汇总 proposers 的回答和 critics 的批判，按论文 Builder 的动作格式综合出一个
   最终动作（PLACE / REMOVE / CLARIFY），该动作执行后的任务进度就是本轮分数。

下一轮的三个 proposers 会同时看到题目和上一轮 Judge 的综合回答，如此循环 25 轮。

## 哪一层最值得升级模型

（历史分析记录：当前实验已冻结为全部 `gpt-4o-mini`，以下内容仅作背景参考。）

优先升级 **proposers（P1/P2/P3）**，收益最大，原因有三：

1. **信息瓶颈在 proposers**：三个 proposers 是唯一能看到私有目标视图的 agent，critics 和
   Judge 都看不到目标。proposer 没有说出的信息，后面任何 agent 都无法凭空补出来。
2. **论文刻意弱化了 Builder**：CRAFT 用 oracle 候选动作限制了 Builder（对应这里的 Judge）的
   动作空间，官方 README 明确说这是为了"把 Director 通信作为性能瓶颈单独隔离出来"。
3. **产出比理解更难**：论文引用的研究表明 LLM 作为"听话者"明显强于"说话者"。proposers 是
   产出方（说话者），critics/Judge 是理解方（听话者），瓶颈在产出方。

次优先是 **Judge**：它负责把六份意见映射成最终动作，出错（层数、span、块编码）会直接扣分。
如果 `oracle.enabled=false`（Judge 必须自己做完整空间推理），Judge 的升级收益会明显上升。
Critics 能过滤冲突和冗余，但补不了缺失的私有信息，边际收益最低。

`judges`（SG/MM/PS）是离线评估器，不参与决策、不影响分数曲线，只为诊断用，不需要为了提分升级它。

## 目录结构

```text
CRAFT/
├── .secret/               # 你的 API key（不入库）
├── benchmark/             # CRAFT 官方 20 个结构 + 程序化生成的 craft-N 数据集
├── config/                # 实验配置
├── results/               # 汇总结果 + 分数曲线 PNG
├── src/craft_debate/      # 面向程序的核心代码
├── scripts/               # 面向用户的入口，你只需要运行这里
└── trajectories/          # 最重要的完整实验记录（每个 agent 的 prompt/回答/执行/分数）
```

## 快速开始

```bash
# 1. 安装依赖（也可以先建一个 venv）
python3 -m pip install -e .

# 2. 放入 API key（二选一）
cp .secret/openai_api_key.example .secret/openai_api_key   # 然后把文件里的 key 换成真实的
# 或者： export OPENAI_API_KEY=sk-...

# 3. 检查环境
python3 scripts/check_setup.py

# 4. 跑一次实验（默认 1 个结构 × 1 run × 25 轮）
python3 scripts/run_debate.py

# 5. 重画某个实验的分数曲线
python3 scripts/visualize_scores.py --latest

# 纵轴默认上限 0.8（更容易看出各轮差距）；想改可用 --ymax，恢复全量用 1.0
python3 scripts/visualize_scores.py --latest --ymax 0.7
```

没有 key 时也可以先用 mock 模型验证整条链路：

```bash
python3 scripts/run_debate.py --mock
```

## 常用参数

```bash
python3 scripts/run_debate.py \
  --config config/debate_config.json \
  --structures 0,1,2 \
  --runs 1,2,3
```

`--structures` 是所有 20 个结构的索引（`0-19`，7 simple / 8 medium / 5 complex）。
`--runs` 对应论文里的多 run 设置；每个 structure-run 组合都会完整跑 25 轮。

## 程序化生成数据集与选择

```bash
# 生成 80 个结构（默认写到 benchmark/craft-80.json，命名规则 craft-<数量>）
python3 benchmark/generate_benchmark.py --count 80

# 只保留理论天花板不低于 0.7 的结构（适合早停策略训练，避免不可达的低分结构）
python3 benchmark/generate_benchmark.py --count 80 --out benchmark/craft-80-ceil07.json --min-ceiling 0.7

# 运行时选择数据集：--benchmark 传名称（自动补 benchmark/ 与 .json）或直接传路径
python3 scripts/run_debate.py --benchmark craft-80
python3 scripts/run_debate.py --benchmark benchmark/craft-80.json --structures 0,1,2
```

生成器是官方 `structure_generator_v2.py` 的忠实移植（同样的层铺、domino 规则与校验），
结构 id 形如 `craft-80-001`，`--seed` 可复现，`--min-ceiling` 用本仓库的 oracle 计算
每个结构可达的理论上限并过滤。

## 已冻结的实验设置

从当前版本起冻结为：**25 轮、7 个 agent 全部使用 `gpt-4o-mini`、拓扑保持 `3 → 3 → 1` 不变**。
模型和轮数由 [config/debate_config.json](config/debate_config.json) 控制（`debate.max_rounds=25`），
运行脚本不再接受 `--rounds / --model / --temperature` 命令行覆盖；如需换模型或轮数，直接改该配置文件。
早停策略研究的所有数据采集都基于这套冻结设置（craft-80 × 3 runs × 25 轮）。

## 实验命名与产物

每次实验按 `YYYYMMDDHHMM-<模型名>` 命名，`trajectories/` 与 `results/` 同名一一对应：

```text
trajectories/202608172330-gpt-4o-mini.json   # 完整轨迹（每轮 7 个 agent 的 prompt/回答、动作、分数）
results/202608172330-gpt-4o-mini.json        # 汇总（每轮分数曲线、最终进度等）
results/202608172330-gpt-4o-mini.png         # 25 轮分数曲线可视化
```

## 复现论文的关键设置

- 数据：官方 `structures_dataset_20.json`（20 个结构，21-25 块，7/8/5 复杂度划分）。
- 引擎：3×3 网格、最多 3 层、5 色、small/large domino 的物理校验，与官方实现一致。
- Oracle：每轮枚举“朝目标前进且物理可执行”的正确动作，给 Judge 最多 5 个候选（论文主实验设置）。
- 分数：IoU、距离（1-归一化编辑距离）、完成率、位置准确率，`overall_progress` 为前三者平均。
- 采样：API 模型 temperature=0.7（论文原设置）；固定 25 轮；对话超过 50 条时截断到最近 40 条。
- 可选：论文的 SG / MM / PS LLM 评判器（`config` 里 `judges.enabled` 打开，会显著增加调用量）。
