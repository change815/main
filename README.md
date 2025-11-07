# Autogate: Elastic Registration Driven Polygon Gating

Autogate 是一个基于 SimpleITK B-样条弹性配准的自动圈门工具。它面向 2D 多边形门场景，支持按训练样本自适应选择、配置化参数管理、完整 CLI、详细日志以及基础 F1 评估。项目完全由 Python 实现，默认使用 8-bit 灰度密度图驱动配准。

## 功能特性

- ✅ **MVP 流程**：读取训练 FCS 与对应多边形门 CSV，生成目标 FCS 的自动圈门结果。
- ✅ **点云 → 8-bit 图像**：对任意 `(channel_x, channel_y)` 组合生成固定分辨率密度图，可配置平滑。
- ✅ **B-样条弹性配准**：基于 SimpleITK，支持多级金字塔、网格间距与迭代次数配置。
- ✅ **多训练样本自动选择**：对每个 plot 使用 L2 距离挑选最相似的训练图像再做配准。
- ✅ **配置化管理**：通过 YAML 描述 I/O、通道量程、配准与日志参数，命令行可覆盖关键项。
- ✅ **CLI 全流程**：`autogate init-config / run / eval` 覆盖初始化、运行与评估。
- ✅ **日志追踪**：标准 logging 同时输出控制台与文件，记录耗时、样本选择、回退等信息。
- ✅ **基础 F1 评估**：如提供真值 CSV，则可输出 precision/recall/F1 的 evaluation.csv。

```
┌───────────────┐    ┌───────────────────────────┐    ┌─────────────────────┐
│ 训练 FCS + 门 │──▶│ 点云→密度图→样本选择 │──▶│ B-样条配准 + 变形 │
└──────┬────────┘    └──────────┬────────────────────┘    └──────────┬────────┘
       │                           │                                  │
       │                           ▼                                  ▼
       │                    目标 FCS 密度图                目标多边形门 CSV
       └───────────────────────────────────────────────────────────────┘
```

## 安装与环境

- Python ≥ 3.10
- 依赖见 [`requirements.txt`](requirements.txt)。建议创建虚拟环境后执行：

```bash
pip install -r requirements.txt
```

### SimpleITK 安装提示

SimpleITK 官方轮子覆盖主流平台 (`manylinux`, `win_amd64`, `macosx_arm64/x86_64`)。若网络访问受限，可从 [SimpleITK PyPI](https://pypi.org/project/SimpleITK/) 下载对应 whl 后使用 `pip install SimpleITK-<version>-cp310-cp310-manylinux2014_x86_64.whl` 等方式离线安装。

## 配置文件

执行 `autogate init-config` 会在 `./cfg/panel.example.yaml` 下生成示例配置，字段说明：

- `io.*`：训练/目标数据目录、训练门 CSV、输出目录及可选真值 CSV。
- `panel.channels`：全局需要的通道列表；所有 FCS 必须包含这些列。
- `panel.transform`：数值变换（目前支持 `asinh`、`log10`、或省略）。
- `panel.compensation`：`auto`（使用 FCS 元数据）或指定补偿矩阵路径，`null` 表示不补偿。
- `panel.ranges`：各通道统一量程，缺失会报错。
- `imaging.*`：密度图分辨率与 Gaussian 平滑 sigma。
- `registration.*`：B-样条网格间距、金字塔层数、最大迭代数。
- `selection.metric`：训练样本选择度量，目前仅支持 `l2`。
- `logging.*`：日志级别与目录。

### 训练门 CSV 约定

| 列名        | 说明                                                       |
|-------------|------------------------------------------------------------|
| `gate_id`   | 唯一标识                                                   |
| `parent_id` | 父级门（保留字段）                                         |
| `type`      | 固定为 `polygon`                                           |
| `channel_x` | X 轴通道                                                   |
| `channel_y` | Y 轴通道                                                   |
| `points`    | 形如 `[(x1,y1),(x2,y2),...]` 的 JSON 风格字符串            |
| `fcs_file`  | 训练门对应的 FCS 文件名（含或不含后缀皆可），缺省时取首个训练文件 |

## 快速开始

1. **初始化配置**
   ```bash
   autogate init-config --directory ./cfg
   cp cfg/panel.example.yaml cfg/panel.yaml
   ```
2. **编辑配置**：按实际通道、量程、路径修改 `cfg/panel.yaml`。
3. **运行自动圈门**
   ```bash
   autogate run --config cfg/panel.yaml --out-dir ./out/gates
   ```
   运行过程中会输出训练样本选择、配准状态、耗时等日志，并在 `out/gates/` 下生成 `*_gates.csv`。
4. **可选评估**（需要真值 CSV）
   ```bash
   autogate eval --config cfg/panel.yaml --truth ./data/eval/truth.csv
   ```
   结果存储于 `out/gates/evaluation.csv`，同时控制台打印宏/微平均指标。

## 测试与样例数据

项目提供基于高斯分布的合成数据单元测试，确保：

- 密度图生成、训练样本选择、B-样条配准、门形变均可运行。
- 端到端冒烟测试验证从训练到导出 CSV 的完整流程。

执行测试：

```bash
pytest
```

## 常见问题排查

- **缺少通道或量程**：配置中的 `panel.channels` 必须在所有 FCS/CSV 中存在；`panel.ranges` 需覆盖全部通道。
- **补偿矩阵缺失**：若 FCS 元数据无 `$SPILLOVER` 且配置为 `auto`，会提示未补偿，可在 `panel.compensation` 指定矩阵 CSV。
- **配准失败回退**：当 SimpleITK 优化异常时会记录 `WARNING` 并自动使用恒等变换继续流程。
- **CSV points 无效**：解析失败的门会被跳过并记录日志，请确保格式符合示例。

## 后续扩展建议

- **并行化**：可在 `pipeline.run()` 中按目标文件或 plot 级别引入并发（多进程/线程），注意 SimpleITK 线程安全性与日志同步。
- **结果缓存**：针对重复的目标文件或密度图，可缓存成像与配准结果以降低重复计算开销。

## 许可证

本项目以 MIT 许可证发布，可自由用于科研与工程场景。详见 `LICENSE`（如需）。
