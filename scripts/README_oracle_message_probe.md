# Oracle 消息依赖小实验

这个脚本回答一个窄问题：**在相同棋盘和相同 Oracle 候选下，Builder 没有正常 Director 消息时，还能否选择并执行有效动作？**

这是独立的一步诊断。它只读取现有轨迹，复用公开消息；不修改原来的 baseline、环境、提示词或配置，不重新生成 Director 消息，也不运行完整游戏。

## 默认规模与条件

- 来源：`trajectories/director-baselines-fixed-1.json`。
- 结构索引：`7,8`（从 0 开始，均已检查可达）。
- 运行编号：`1`；源轨迹轮次：`1,6,11`（从 1 开始）。
- 合计 6 个棋盘状态，每个状态测试 3 种消息条件，各请求 Builder 一次。
- **共 18 次新的 Builder 请求，0 次 Director 请求**。已有客户端的失败重试可能额外发出请求。
- 默认从源轨迹继承 Builder 配置，目前为 `gpt-4o-mini`、temperature `0.1`。不需要 Ollama。

| 条件 | Builder 收到的 Director discussion |
|---|---|
| `normal` | 该状态原有的一名 Director 的公开消息 |
| `no_message` | 没有 Director 消息的占位文本 |
| `unrelated` | 其他结构中同一 Director 身份的真实公开消息 |

无关消息优先取相同轮次，否则取最近轮次，使用固定随机种子挑选；不根据表现、正确性或内容选择。它可能偶然与当前目标相容，因此这是“异结构消息”而非保证错误的消息。

三个条件共享同一棋盘、同一有序候选列表、同一系统提示词和 Builder 参数，只有 discussion 段改变。每次执行使用独立状态副本，前一次响应不会影响后一次。源轨迹的跨度通过重放原动作恢复，脚本还会验证重放状态与当前 Oracle 实现一致。

Oracle 设置仍为最多 5 个候选；默认 6 个状态实际分别有 **5、5、3、5、4、2** 个候选。脚本拒绝空候选或只有一个候选的测试点，避免把“没有选择余地”当成结果。默认 6 个状态均衡覆盖三种条件的六种调用顺序。

## 在远程仓库根目录运行

使用运行原 baseline 的 Python 环境即可，无新增依赖。远程需要同步本脚本，并保留完整的 Fixed-1 轨迹文件。

先检查状态与请求数量，不访问模型、不读取 API key，也不写结果：

```bash
python scripts/probe_oracle_messages.py --dry-run
```

正式运行默认 18 次请求：

```bash
python scripts/probe_oracle_messages.py --name oracle-message-tiny
```

沿用现有 `OPENAI_API_KEY` 环境变量或 `.secret/openai_api_key` 文件。API key 不要写进命令行。所有相对输入路径都从仓库根目录解析。

若只想先跑 **6 次请求**（一个结构、两个状态）：

```bash
python scripts/probe_oracle_messages.py --structures 7 --turns 1,6 --name oracle-message-six
```

离线验证管线，不代表真实实验结果：

```bash
python scripts/probe_oracle_messages.py --mock --name oracle-message-mock
```

若远程使用不同配置，可传 `--config config/paper_config.json`；这只读取配置，不会修改它。`--repeats 2` 将默认真实请求数增加到 36，第一轮诊断不必开启。

## 中断后继续

每个成功响应都会立即保存。使用相同代码、源轨迹和所有参数，加上 `--resume`：

```bash
python scripts/probe_oracle_messages.py --name oracle-message-tiny --resume
```

已有同名结果不会被覆盖。参数、代码或源数据变化会拒绝续跑；用新名字开始新实验。已经保存的响应不会重新请求；如果网络在服务器处理请求后中断、响应尚未保存，这一次可能需要重新请求。

## 输出与解读

结果单独保存到 `results/oracle-message-probe/<name>/results.json`。包含三组汇总、逐状态配对差值、完整提示词与响应、所用候选、原棋盘与跨度、无关消息来源，以及 Builder token 用量。

控制台主要列：

- `oracle OK`：动作严格匹配候选且成功执行的比例。比较包含颜色和跨度，反向书写同一跨度视为等价。
- `positive`：本次执行使真实进度上升的比例。
- `mean delta`：平均一步进度变化，**不是整局终分**。
- `invalid` / `clarify`：执行或解析失败比例 / 澄清比例。

另提供不调用 API 的机械参考：逐一模拟每个候选，计算均匀随机选择的精确平均进度变化。它是自动执行器参考，不是 LLM 实验条件。

判断时同时看“能否前进”和“选了哪个动作”。不同候选都能前进，因此动作不相同不一定意味着质量不同。

- `no_message` 与 `normal` 接近且大部分能前进：初步支持 Oracle 候选对这些状态已足够，值得进一步检查信息替代。
- `no_message` 表现好、`unrelated` 下降：提示错误上下文可能造成干扰，不能据此认定正常消息提供了不可替代的信息。
- `normal` 明显更好：提示消息有帮助，但仍需更大样本区分模型随机性和稳定效果。
- 三组都差：先检查输出格式、跨度错误和提示词遵循问题。

这只有两个结构、六个固定状态，是筛查信号，不做显著性或全局等效结论；也不能据此证明整个 20 轮任务或其他模型不需要通信。状态来自 Fixed-1 的既有轨迹，不代表其他策略会到达相同状态。

`--mock` 模型按设计总选首个候选，不读取消息；三组 mock 表现一致只能说明管线正常。
