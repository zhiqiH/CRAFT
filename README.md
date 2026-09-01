# CRAFT × 3+3+1 Generative Debate

本仓库的主实验实现固定 `3 + 3 + 1` 通信拓扑，同时让 Builder 面对真实的开放式
动作生成任务。每一轮恰好调用 7 次 LLM：

```text
D1 / D2 / D3  并行独立观察，各自形成自然语言公开消息
        ↓
同一 D1 / D2 / D3  并行 reconciliation，只交流公开消息
        ↓
Builder 根据三条最终公开消息生成完整 PLACE / REMOVE / CLARIFY
        ↓
公开物理规则事后校验 → 合法才执行 → 独立离线计分
```

这与论文设定保留相同的 Director/Builder 分工和视角映射，但有两处有意差别：

1. 通信流固定为三个 Director 独立观察、同三个身份 reconciliation、一个 Builder，
   即每轮 `3 + 3 + 1`，而不是随机抽取 1–3 个 Director 顺序发言。
2. Builder 不接收经过答案验证的候选动作，也不接收完整动作菜单或动作编号；它必须从
   Director 的自然语言交流中自行推导 block、position、layer 和 span。

## 信息边界

- Phase 1 的每个 Director 只看到自己的 private wall view、公开棋盘和公开历史。
- Reconciliation 仍使用同一 D1/D2/D3 身份。每个 Director 看到自己的 private wall
  view 和三条 Phase-1 公共 `<message>`，看不到其他人的 `<analysis>`、raw response 或
  private view。
- Builder 只看到公开棋盘、三条 reconciliation 公共 `<message>`、可用 block 类型、
  公开物理规则和上一条公开 Builder 结果。
- 只有 `<message>` 跨通信边；`<analysis>` 只保留在离线 trajectory 中。
- 完整隐藏结构及其分数只用于环境初始化、终止判断和动作执行后的离线评价，不进入任何
  agent prompt、调用 metadata 或公开历史。

## 通信与执行协议

Director 的两个阶段都只返回：

```xml
<analysis>私有推理</analysis>
<message>自然语言公开指令或澄清请求</message>
```

Director 使用自己视角下的 left/middle/right 和 bottom/middle/top，说明颜色、大小及
相对位置，不负责输出坐标、layer、span 或机器动作语法。这样可以避免把弱模型同时变成
空间观察者和动作编译器。

Builder 负责把三条最终消息映射成一条完整动作：

```text
PLACE:block_code:position:layer:CONFIRM:interpretation
PLACE:block_code:position:layer:span_to:CONFIRM:interpretation
REMOVE:position:layer:CONFIRM:interpretation
REMOVE:position:layer:span_to:CONFIRM:interpretation
CLARIFY:specific question
```

本地解析器把回复区分为 `exact`、`recovered` 和 `invalid`。唯一且明确的公开消息或动作
即使带代码围栏、额外说明或缺少闭合标签，也可安全恢复；缺失或多重冲突输出会被隔离。
解析器不会把错误动作吸附为另一条动作，也不会替 Builder 重新决策。

Builder 输出动作后，`validate_physical_action` 只依据当前公开棋盘和确定性物理规则检查：

- PLACE 是否位于下一空层；
- REMOVE 是否只移除顶层；
- large block 是否有两个相邻、等高且允许的端点；
- small block 是否错误携带 span；
- layer、block 和坐标是否合法。

`CLARIFY` 属于协议有效但不执行；语法错误、物理拒绝和执行成功分别记录。每次失败原因或
澄清问题都会进入下一轮公开历史，控制台同时显示连续未执行轮数，避免再次把长期停滞误判
成模型仍在有效推进。

## 使用 Mistral-7B 运行 craft-20-hollow

默认配置已将 Phase 1、reconciliation 和 Builder 三个阶段都设为
`mistral:7b-instruct-v0.3-q8_0`，benchmark 为 `craft-20-hollow`。

先完成本地模型与项目准备：

```bash
ollama pull mistral:7b-instruct-v0.3-q8_0
python3 -m pip install -e .
python3 scripts/check_setup.py
```

先跑结构 0、run 1 的 20 轮 canary：

```bash
python3 scripts/run_debate.py \
  --benchmark craft-20-hollow \
  --structures 0 \
  --runs 1 \
  --name craft20-hollow-mistral7b-generative-v3-s0-r1
```

确认每轮日志均为 `p1:3/3`、`rec:3/3`，且 Builder 的 `parse`、`physics`、`executed`
状态正常后，再跑完整 20 个结构：

```bash
python3 scripts/run_debate.py \
  --benchmark craft-20-hollow \
  --structures 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
  --runs 1 \
  --name craft20-hollow-mistral7b-generative-v3-r1
```

无 API key 时可用确定性 mock 验证完整链路：

```bash
python3 scripts/run_debate.py --mock --benchmark craft-20-hollow --structures 0 --runs 1
```

## 日志与产物

完整 trajectory 每轮包含：

- 三个 Phase-1 和三个 reconciliation 的 raw 输出、私有分析、公共消息、usage 和 latency；
- 两条显式的 sanitized public-message 边界记录；
- Builder 生成的完整动作、parse mode、usage 和 latency；
- 独立的 physical validation、execution 和离线 evaluation；
- `protocol_status`、连续未执行轮数及下一轮可见的公开历史。

汇总结果额外报告 Director/Builder 协议有效率、Builder clarification 数、物理合法率、执行率
和最大连续未执行轮数。产物写入：

```text
trajectories/<experiment>.json  # 完整推理与评价轨迹
results/<experiment>.json       # 分数和协议健康度汇总
results/<experiment>.png        # 分数曲线
```

运行测试：

```bash
PYTHONPYCACHEPREFIX=/tmp/craft-pycache python3 -m unittest discover -s tests -v
```

## 与论文复现 runner 的关系

`scripts/run_paper.py` / `paper_protocol.py` 仍是独立的论文 baseline 复现工具，并保留论文
中的顺序发言和候选动作机制。主实验 `scripts/run_debate.py` 不导入或调用该决策链。
