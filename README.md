# Autogate: Elastic Registration Driven Polygon Gating

Autogate 是一个基于 SimpleITK B-样条弹性配准的自动圈门工具。它面向 2D 多边形门场景，支持按训练样本自适应选择、配置化参数管理、完整 CLI、详细日志以及基础 F1 评估。项目完全由 Python 实现，默认使用 8-bit 灰度密度图驱动配准。

## 功能特性

- ✅ **MVP 流程**：读取训练 FCS 与对应多边形门 CSV，生成目标 FCS 的自动圈门结果。
- ✅ **点云 → 8-bit 图像**：对任意 `(channel_x, channel_y)` 组合生成固定分辨率密度图，可配置平滑。
- ✅ **B-样条弹性配准**：基于 SimpleITK，支持多级金字塔、网格间距与迭代次数配置。
- ✅ **多训练样本自动选择**：对每个 plot 使用 L2 距离挑选最相似的训练图像再做配准。
- ✅ **配置化管理**：通过 YAML 描述 I/O、通道量程、配准与日志参数，命令行可覆盖关键项。
- ✅ **CLI 全流程**：`autogate init-config / run / eval` 覆盖初始化、运行与评估。
- ✅ **事件级分类与可视化**：自动生成每个目标细胞的预测类别表与按坐标轴配色的散点图。
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

## 安装与环境（保姆级）

1. **准备 Python 3.10+**：
   - Windows：前往 [python.org](https://www.python.org/downloads/) 下载安装包，勾选 *Add python.exe to PATH*。
   - macOS/Linux：使用系统包管理器或 `pyenv` 安装（例如 `brew install python@3.11`）。
2. **创建独立虚拟环境**（推荐，避免污染系统包）：
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
   python -m pip install --upgrade pip
   ```
3. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```
4. **验证安装**：
   ```bash
   autogate --help
   ```
   若命令找不到，请确认虚拟环境已激活并执行 `pip install -e .` 以可编辑模式安装本项目。

### SimpleITK 安装提示

SimpleITK 官方轮子覆盖主流平台 (`manylinux`, `win_amd64`, `macosx_arm64/x86_64`)。若网络访问受限，可从 [SimpleITK PyPI](https://pypi.org/project/SimpleITK/) 下载对应 whl 后使用 `pip install SimpleITK-<version>-cp310-cp310-manylinux2014_x86_64.whl` 等方式离线安装。若公司内网需要代理，请在安装前设置 `PIP_INDEX_URL`。安装后可通过 `python -c "import SimpleITK as sitk; print(sitk.Version_VersionString())"` 验证。

## 配置文件

执行 `autogate init-config` 会在 `./cfg/panel.example.yaml` 下生成示例配置，字段说明：

- `io.*`：训练/目标数据目录、训练门 CSV、门输出目录以及可选的真值 CSV、事件级标签目录与可视化目录。
- `panel.channels`：全局需要的通道列表；所有 FCS 必须包含这些列。
- `panel.transform`：数值变换（目前支持 `asinh`、`log10`、或省略）。
- `panel.compensation`：`auto`（使用 FCS 元数据）或指定补偿矩阵路径，`null` 表示不补偿。
- `panel.ranges`：各通道统一量程，缺失会报错。
- `imaging.*`：密度图分辨率与 Gaussian 平滑 sigma。
- `registration.*`：B-样条网格间距、金字塔层数、最大迭代数。
- `selection.metric`：训练样本选择度量，目前仅支持 `l2`。
- `logging.*`：日志级别与目录。
- `visualization.*`：控制散点图配色；示例配置已针对 `CD34/CD117/SSC` 三个坐标系给出专属颜色，`UNGATED` 会使用 `ungated_color` 指定的灰色。
- `evaluation.*`：评估模式及真值列名；若提供事件级真值标签，保持 `mode: auto` 即可自动检测。

### 训练门 CSV 约定

| 列名          | 说明                                                       |
|---------------|------------------------------------------------------------|
| `gate_id`     | 唯一标识                                                   |
| `parent_id`   | 父级门（保留字段）                                         |
| `type`        | 固定为 `polygon`                                           |
| `channel_x`   | X 轴通道                                                   |
| `channel_y`   | Y 轴通道                                                   |
| `points`      | 形如 `[(x1,y1),(x2,y2),...]` 的 JSON 风格字符串            |
| `population`  | （推荐）该门对应的细胞类别，例如 `H`、`G`、`Mono` 等；若缺失则默认等同于 `gate_id` |
| `fcs_file`    | 训练门对应的 FCS 文件名（含或不含后缀皆可），缺省时取首个训练文件 |

## 保姆级使用流程

> 下列命令均假设你在项目根目录执行，并已激活虚拟环境。

### 1. 初始化目录结构

```bash
autogate init-config --directory ./cfg
cp cfg/panel.example.yaml cfg/panel.yaml
mkdir -p data/train data/target out/gates out/logs
```

### 2. 准备训练数据

1. **训练 FCS** 放入 `data/train/`，文件名示例 `donor01.fcs`、`donor02.fcs`。
2. **训练门 CSV**：
   - 必备字段：`gate_id,parent_id,type,channel_x,channel_y,points,fcs_file`。
   - `population` 建议填写细胞类别（如 `H`、`G`、`C`/`Mono`/`Lym`/`Gra`/`F`），这样事件级标签将直接使用这些名称；若缺失则默认使用 `gate_id`。
   - `points` 是字符串形式的顶点序列，如 `[(123.4,56.7),(140.2,80.1),...]`，首尾会自动闭合。
   - `fcs_file` 用于指明该门来自哪一个训练 FCS（必须能在 `data/train/` 找到同名文件）。如果所有门都来自同一文件也可以只填写一次并向下填充。
   - 若你现在的数据是事件级 CSV（如截图中包含 `FSC-A, SSC-A, CD3 APC-A750, cell_type` 等列），需要先在第三方工具（FlowJo、Cytobank、Kaluza 等）或自编脚本中**画出多边形门并导出顶点**，再整理成上述格式。事件级 CSV 可辅助你确定坐标范围，但不能直接作为门 CSV 使用。
3. `cfg/panel.yaml` 中的 `io.train_fcs_dir` 指向训练 FCS 目录，`io.train_gates_csv` 指向训练门 CSV。

### 2.1 准备真值标签 CSV（用于 F1）

若你已经为目标样本准备了事件级的“细胞类型标注 CSV”（例如 `sample003.csv` 包含 `CD34 ECD-A, CD45 APC-A750-A, cell_type` 等列），需按如下格式整理以便评估：

1. **统一列名**：创建一个新的 CSV 或目录，至少包含三列：
   - `sample_id`：与目标 FCS 文件名（不含扩展名）一致，例如 `003`、`006`、`007`；
   - `event_index`：对应 FCS 行号，从 0 开始递增。若原始 CSV 保持与 FCS 相同顺序，可直接添加一列 `event_index = range(len(df))`；
   - `cell_type`：事件所属的真实细胞类别（`H`、`G`、`C`、`Mono`、`Lym`、`Gra`、`F` 等）。
2. **保存位置**：可以将全部样本合并为一个 CSV（推荐命名为 `data/eval/labels.csv`），也可以一个样本一份放在目录 `data/eval/labels/` 下，文件名任意但需保持 `.csv` 后缀。
3. **配置修改**：在 `cfg/panel.yaml` 中设置 `io.eval_truth_csv` 指向该 CSV 或目录；`evaluation.label_column/sample_id_column/event_index_column` 可按需要调整（默认即 `cell_type/sample_id/event_index`）。

> 提示：若原始标注按坐标组合分别保存（类似截图中的三份 `annotation_*.csv`），请先合并为一个文件，并确保 `sample_id` 与 FCS 对应，否则无法对齐事件级预测。

### 3. 准备目标数据

将待处理的目标 FCS 放入 `data/target/`。程序会自动遍历目录下的所有 `.fcs` 文件。若未来需要评估，提前准备好对应的真值门 CSV（结构与训练门一致）。

### 4. 配置参数

打开 `cfg/panel.yaml`，逐项确认：

- `panel.channels` 必须覆盖训练/目标 FCS 中会用到的所有通道。
- `panel.transform` 建议保持 `asinh`，除非已有预处理。
- `panel.ranges` 中的每个通道范围可以参考事件级 CSV 的最小/最大值或经验量程。例如截图中的 `FSC-A`、`SSC-A` 列可计算 0.1% 与 99.9% 分位后填入。
- `imaging.bins` 决定密度图分辨率，默认 256；`smooth_sigma=1` 可在点较少时平滑噪声。
- `registration` 参数若首次使用可保留默认。
- `logging.log_dir` 建议保持在 `out/logs`。

### 5. 运行自动圈门

```bash
autogate run --config cfg/panel.yaml --out-dir ./out/gates
```

运行日志会在控制台滚动，同时写入 `out/logs/<timestamp>.log`。对于每个 `(channel_x, channel_y)` 组合，日志中会显示自动选择的训练样本及其 L2 距离。如果 SimpleITK 配准失败，会出现 `WARNING` 并自动回退为恒等变换。

运行完成后会生成三类结果：

- `out/gates/<sample>_gates.csv`：目标样本的形变后多边形门。可用于后续评估或导入流式软件。 
- `out/gates/event_labels/<sample>_labels.csv`：逐细胞的预测类别，字段包括 `event_index`（原始顺序）、`population`（最终细胞类型，如 `H`/`G`/`C`/`Mono` 等）、`gate_id`（叶子门名称）、`depth` 和 `gate_path`（从根到该门的路径）。
- `out/gates/plots/<sample>__<channel_x>__<channel_y>.png`：按配置通道组合绘制的散点图，不同颜色对应预测类别，可快速肉眼核对圈门效果。示例配置中已经指定：`CD34/CD45` 图仅显示 `H` 与未分类，`CD117/CD45` 图显示 `G` 与未分类，`SSC/CD45` 图分别给 `C`、`Mono`、`Lym`、`Gra`、`F` 与 `UNGATED` 不同颜色。

### 6. （可选）评估 F1

根据真值类型选择评估模式：

- **多边形门真值**（`truth` 含 `points` 列）：
  ```bash
  autogate eval --config cfg/panel.yaml --mode gates --truth ./data/eval/truth.csv
  ```
  输出的 `evaluation.csv` 会列出每个门的 Precision/Recall/F1，并在日志中给出宏/微平均成绩。
- **事件级标签真值**（第 2.1 步整理的 `labels.csv` 或目录）：
  ```bash
  autogate eval --config cfg/panel.yaml --mode labels \
    --predictions ./out/gates/event_labels --truth ./data/eval/labels.csv
  ```
  程序将按 `event_index` 对齐预测与真值，统计各细胞类型的 TP/FP/FN 并输出 `evaluation.csv`（列含 `sample_id,population,precision,recall,f1,tp` 等）。

若保持 `mode: auto` 且未在命令行指定，工具会根据真值是否包含 `points` 列自动判别。无论哪种模式，当某个门/细胞类型的事件数不足 10 个时，日志会提示 `LOW_COUNTS`，请结合上下文谨慎解读。

### 7. 排查与调优

- **如何挑选训练文件？**
  - `data/train/` 中可以放多份 FCS；程序会对每个 plot 自动计算与目标图的 L2 距离并记录在日志中，距离最小的训练图像将参与该 plot 的配准。
  - 如果想固定只使用某一训练文件，可暂时移除其他 FCS，或在训练门 CSV 中只为该文件提供门并删除其它文件。
- **如何生成训练门 CSV？**
  - 若手头只有事件级 CSV（类似截图），可以：
    1. 用 Python 读取该 CSV，筛选出某个 `cell_type` 的点；
    2. 在可视化工具中绘制多边形并导出顶点；
    3. 将顶点列表填入 `points` 列，保存为 UTF-8 编码 CSV。
  - 也可以使用 FlowJo 等软件在 FCS 上直接圈门后导出门定义，再整理列名为项目要求格式。
- **如何设定量程？**
  - 计算事件级 CSV 的分位数：
    ```python
    import pandas as pd
    df = pd.read_csv('your_events.csv')
    print(df['FSC-A'].quantile([0.001, 0.999]))
    ```
  - 将得到的上下界填入配置。确保所有训练与目标文件共享同一量程，以便配准在相同坐标系工作。

## 测试与样例数据

项目提供基于高斯分布的合成数据单元测试，确保：

- 密度图生成、训练样本选择、B-样条配准、门形变均可运行。
- 端到端冒烟测试验证从训练到导出 CSV 的完整流程。

执行测试：

```bash
pytest
```

## 常见问题排查

- **缺少通道或量程**：配置中的 `panel.channels` 必须在所有 FCS/CSV 中存在；`panel.ranges` 需覆盖全部通道。若读取 CSV/截图发现有空值，请在预处理时补齐。
- **补偿矩阵缺失**：若 FCS 元数据无 `$SPILLOVER` 且配置为 `auto`，会提示未补偿，可在 `panel.compensation` 指定矩阵 CSV（格式为方阵，首行首列为通道名）。
- **配准失败回退**：当 SimpleITK 优化异常时会记录 `WARNING` 并自动使用恒等变换继续流程，可通过降低 `registration.iterations` 或增大 `imaging.smooth_sigma` 缓解。
- **CSV points 无效**：解析失败的门会被跳过并记录日志，请确保 `points` 字符串是合法的 JSON 风格数组，如 `[(0.1,0.2),(0.3,0.4)]`。可在保存前使用 `json.loads(points.replace('(', '[').replace(')', ']'))` 验证。
- **事件 CSV 如何参与流程？**：事件 CSV 主要用于估计量程或生成真值门。若想直接跑流程，建议先转为 FCS（例如借助 FlowKit/FlowCore）后再走统一的补偿与配准。

## 后续扩展建议

- **并行化**：可在 `pipeline.run()` 中按目标文件或 plot 级别引入并发（多进程/线程），注意 SimpleITK 线程安全性与日志同步。
- **结果缓存**：针对重复的目标文件或密度图，可缓存成像与配准结果以降低重复计算开销。

## 许可证

本项目以 MIT 许可证发布，可自由用于科研与工程场景。详见 `LICENSE`（如需）。
