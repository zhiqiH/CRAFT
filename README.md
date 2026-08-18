# CRAFT × Debate

一个把 [CRAFT](https://arxiv.org/abs/2603.25268)（arXiv:2603.25268v2）benchmark 与自定义
**7-agent 多轮 Multi-Agent Debate 拓扑（Debate）** 结合的实验框架。每轮 7 个子 agent 都使用
`gpt-4o-mini`，共 20 轮，并按论文方法评估任务进度，最后把 20 轮的分数曲线画出来。

## Debate 拓扑

每轮的结构是 `3 → 3 → 1`：

1. **Proposers（P1/P2/P3，并行）**：即论文的 D1/D2/D3，各自持有一面墙的私有 2D 视图，同时看到
   题目（当前棋盘 + 自己的目标视图）和上一轮 Judge 的综合回答，输出 `<think>` + `<message>`。
2. **Critics（C1/C2/C3，并行）**：看到题目和 P1-P3 的完整回答，从三个角度辩论批判——
   spatial grounding（空间接地）、mind modeling（心智建模）、pragmatic sufficiency（语用充分性），
   分别输出 `<critique>` + `<message>`。
3. **Judge（J1）**：汇总 proposers 的回答和 critics 的批判，按论文 Builder 的动作格式综合出一个
   最终动作（PLACE / REMOVE / CLARIFY），该动作执行后的任务进度就是本轮分数。

下一轮的三个 proposers 会同时看到题目和上一轮 Judge 的综合回答，如此循环 20 轮。

## 哪一层最值得升级模型

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
├── benchmark/             # CRAFT 官方 20 个结构数据
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

# 4. 跑一次实验（默认 1 个结构 × 1 run × 20 轮）
python3 scripts/run_debate.py

# 5. 重画某个实验的分数曲线
python3 scripts/visualize_scores.py --latest
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
  --runs 1,2,3 \
  --rounds 20 \
  --model gpt-4o-mini
```

`--structures` 是所有 20 个结构的索引（`0-19`，7 simple / 8 medium / 5 complex）。
`--runs` 对应论文里的多 run 设置；每个 structure-run 组合都会完整跑 20 轮。

## 分层指定模型（升级 proposers 到 DeepSeek Flash）

配置里 `model` 是全局默认值，`stages` 可以按 `proposers / critics / judge / judges` 分层覆盖。
例如只把 proposers 升级为 `deepseek-v4-flash`（DeepSeek 官方 OpenAI 兼容接口）：

```json
"stages": {
  "proposers": {
    "model": "deepseek-v4-flash",
    "provider": "deepseek",
    "api_key": "deepseek_api_key",
    "temperature": 0.7,
    "max_tokens": 2000,
    "thinking": "disabled"
  }
}
```

仓库里已经有一份现成配置，直接运行：

```bash
# DeepSeek key（在 platform.deepseek.com 申请）
cp .secret/deepseek_api_key.example .secret/deepseek_api_key   # 替换成真实 key
# 或者： export DEEPSEEK_API_KEY=sk-...

python3 scripts/run_debate.py --config config/debate_config_deepseek_proposers.json
```

关于 DeepSeek V4-Flash（官方文档，base_url `https://api.deepseek.com`，模型 `deepseek-v4-flash`）：

- 1M 上下文、384K 最大输出，OpenAI Chat Completions 兼容；
- 默认开启 thinking 模式，此时 `temperature` 不生效且链式思考按输出计费；本框架的提示词已经自带
  `<think>` 块，所以示例配置用 `thinking: "disabled"` 保持论文的 temperature=0.7 并省 token；
- 想开思考模式把 `"thinking"` 改成 `"enabled"` 即可（V4-Flash 离线价约 $0.22/1M 输入、
  $0.66/1M 输出，一个结构 × 20 轮 × 3 个 proposers 通常只需几美分）。

## 实验命名与产物

每次实验按 `YYYYMMDDHHMM-<模型名>` 命名，`trajectories/` 与 `results/` 同名一一对应：

```text
trajectories/202608172330-gpt-4o-mini.json   # 完整轨迹（每轮 7 个 agent 的 prompt/回答、动作、分数）
results/202608172330-gpt-4o-mini.json        # 汇总（每轮分数曲线、最终进度等）
results/202608172330-gpt-4o-mini.png         # 20 轮分数曲线可视化
```

分层混合模型时，模型名部分会展开成 `模型-层名+...`，例如
`202608172330-deepseek-v4-flash-proposers+gpt-4o-mini-critics+gpt-4o-mini-judge`，
完整分层配置仍然记录在 trajectory 的 `experiment.config` 里。

## 复现论文的关键设置

- 数据：官方 `structures_dataset_20.json`（20 个结构，21-25 块，7/8/5 复杂度划分）。
- 引擎：3×3 网格、最多 3 层、5 色、small/large domino 的物理校验，与官方实现一致。
- Oracle：每轮枚举“朝目标前进且物理可执行”的正确动作，给 Judge 最多 5 个候选（论文主实验设置）。
- 分数：IoU、距离（1-归一化编辑距离）、完成率、位置准确率，`overall_progress` 为前三者平均。
- 采样：API 模型 temperature=0.7（论文原设置）；20 轮；对话超过 50 条时截断到最近 40 条。
- 可选：论文的 SG / MM / PS LLM 评判器（`config` 里 `judges.enabled` 打开，会显著增加调用量）。
