# CRAFT × Oracle-free Debate

本仓库在 CRAFT benchmark 上实现严格 partial-observation 的固定 `3 + 3 + 1`
推理拓扑。`scripts/run_debate.py` 的每个 decision step 最多调用 LLM 7 次：

```text
D1 / D2 / D3  并行独立观察
        ↓
D1 / D2 / D3  以相同身份并行 reconciliation
        ↓
Builder 从完整 legal-action mask 中选择一个 action ID
        ↓
公开状态物理校验 → 执行 → 独立离线计分
```

## 信息边界

- Phase 1 的每个 Director 只看到自己的 private view、公开棋盘和公开动作历史。
- Reconciliation 阶段仍是同三个 Director；每个 Director 看到自己的 private view 和三份
  Phase-1 结构化消息，但看不到其他 Director 的 raw private view。
- Builder 只看到公开棋盘、三份 reconciliation 输出和完整 legal-action mask。
- hidden target、oracle moves、ground-truth progress、delta progress 和 target-conditioned ranking
  不进入任何 Director 或 Builder prompt。
- target 只留在环境初始化与动作执行后的离线评价记录中。

## 通信协议

拓扑固定为每轮 `3 + 3 + 1`，协议校验不会增加 LLM 调用：

- Phase 1：每个 Director 返回且仅返回 observation、proposed_action、reasoning、
  confidence 四个 XML 元素；action 使用统一的 PLACE/REMOVE 语法。
- Reconciliation：同一批 Director 只接收三份经过本地校验的 Phase-1 消息；无效消息
  只传递 `protocol_valid=false` 和错误原因，不传递未经验证的语义内容。
- Builder：只接收公开棋盘、三份经过校验的 reconciliation 和完整 legal-action mask，
  且仅返回一个来自当前 mask 的 action ID。
- 本地校验严格检查标签数量、标签外文本、模板回显、action 语法和 confidence 范围。
  校验失败会被记录和隔离，不重试、不调用 oracle，也不改变 `3 + 3 + 1` 拓扑。

`get_all_physically_legal_actions(current_public_state)` 仅使用公开结构、公开 span 和确定性
物理规则枚举全部 PLACE/REMOVE primitive actions。它不会按正确性或进度筛选，Builder 选择的
动作可能对 hidden target 是错的，但一定会先经过 `validate_physical_action` 校验。

## 运行

```bash
python3 -m pip install -e .
python3 scripts/check_setup.py
python3 scripts/run_debate.py
```

无 API key 时可用确定性 mock 验证完整链路：

```bash
python3 scripts/run_debate.py --mock --structures 0 --runs 1
```

常用参数：

```bash
python3 scripts/run_debate.py \
  --config config/debate_config.json \
  --benchmark craft-80 \
  --structures 0,1,2 \
  --runs 1,2,3
```

模型、provider、轮数和角色在 `config/debate_config.json` 中配置。三个推理 stage 分别为
`phase1`、`reconciliation`、`builder`。

## 日志与产物

完整 trajectory 每轮包含：

- 三个 Phase-1 输出及每次调用的 token usage/latency；
- 三个 reconciliation 输出及每次调用的 token usage/latency；
- 完整序列化 legal-action mask；
- Builder 选择的 action ID、usage 和 latency；
- 独立的 deterministic physical validation 和 execution 结果；
- `evaluation` 下的 target-derived score，以及三阶段 wall-clock latency。
- `protocol_status` 下三阶段的消息有效数；汇总结果另含 protocol validity rate。

产物写入：

```text
trajectories/<experiment>.json  # 完整推理与评价轨迹
results/<experiment>.json       # 分数汇总
results/<experiment>.png        # 分数曲线
```

运行测试：

```bash
PYTHONPYCACHEPREFIX=/tmp/craft-pycache python3 -m unittest discover -s tests -v
```

## 与论文复现 runner 的关系

`scripts/run_paper.py` / `paper_protocol.py` 保留为独立的论文 oracle-assisted baseline
复现工具；它不被 `scripts/run_debate.py` 或 `Debate` 推理路径调用。当前主 Debate pipeline
始终使用完整的 oracle-free legal-action mask。
