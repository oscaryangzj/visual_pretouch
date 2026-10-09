# visual_pretouch

**在拇指越过手机屏幕中线时，预测它最终会落在哪里。**

这是一个用于验证视觉触摸前预测假设的离线研究原型。当前任务限定为：固定摄像头视角和手机姿态，单用户左手拇指从左向右越过中线，预测右半屏的最终触点。目标样机为 **Huawei Mate 80 Pro**，采集区域为 **1280×2832 px、70.6×156.2 mm**。

## Demo

下方是 M2 Model B（Gradient Boosting）的演示片段：视频展示拇指轨迹、中线和跨线时生成的预测落点。

![M2 Model B demo：拇指轨迹与跨线预测落点](docs/assets/m2_model_b_preview.gif)

[查看完整 M2 demo 视频](demo/demo_m2_model_b_gradient_boosting.mp4)

该片段用于展示可视化效果，**不对应下方 10-session LOSO 汇总指标**。结果协议和逐项实验记录见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 当前结果

M2 与 M3 在相同的 `single_only` 协议下进行 10-session leave-one-session-out（LOSO）评估，共 218 个有效触摸样本。表中数值是 10 个 session 指标的宏平均；误差单位为 mm，命中率以成功预测为分母。

| 方法 | 平均误差 | 中位误差 | Hit@10 mm | Hit@15 mm | Hit@20 mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| M2 训练集增量中位数 | 14.90 | 13.60 | 33.1% | 54.8% | 73.2% |
| M2 Model A · Ridge | 14.59 | 13.30 | 47.5% | 72.2% | 85.8% |
| **M2 Model B · Gradient Boosting** | **10.87** | **9.28** | **54.0%** | **80.8%** | **91.2%** |
| M3 · tiny LSTM | 11.85 | 11.01 | 45.4% | 67.1% | 87.9% |
| M3 · tiny GRU | 11.84 | 10.23 | 50.8% | 70.9% | 86.6% |

当前数据中，M2 Model B 的汇总误差最低；小型 LSTM/GRU 尚未稳定超过它。这是同一用户的小规模开发结果，不代表跨用户或独立设备泛化。历史数据包含 8 个 Mate 80 Pro 和 2 个 Pura X session，因此也不是纯 Mate 80 Pro 评估。

此表采用 `single_only` 筛选协议；旧 `m2-baseline` tag 使用较早规则，结果不能直接与此表混比。

## 处理流程

1. 用 [`collect.html`](collect.html) 采集触点记录和设备参数，同时录制操作视频。
2. 用 MediaPipe Hands 提取手部关键点，并用 LK 光流维持短时拇指轨迹连续性。
3. 通过本地 review 页面逐次检查轨迹，决定触摸是否保留，并标注手机屏幕四角。
4. 用四角单应变换把拇指位置映射到归一化屏幕坐标；用白闪辅助视频与触点记录对齐。
5. 只评估触摸前恰好一次左→右跨线的保留样本。M2 使用跨线时及之前的因果特征；M3 使用跨线前 300 ms 的轨迹序列。
6. 按 session 留一评估，报告毫米误差和 Hit@r；预测点限制在屏幕范围内。

## 快速开始

需要 Python 3.11、Conda 和可读取采集视频的 FFmpeg。先安装依赖：

```bash
conda create -n visual_pretouch python=3.11 -y
conda activate visual_pretouch
python -m pip install -r requirements.txt
```

### 准备并审核 session

把原始视频和 `collect.html` 导出的采集文件放入 `dataset/session_example/`，再对每个 session 运行审核工具。将命令中的目录名替换为实际 session 目录：

```bash
python scripts/review_session.py --session-dir dataset/session_example
```

在页面中审核每次拇指追踪并标注屏幕四角。审核结果保存在 session 内的 `review.csv` 和 `calibration.json`；追踪准备缓存也保存在该目录。原始数据和派生结果不提交到 Git。

### 运行 M2 / M3 量化

确认 `config.yaml` 的 `data_split` 中列出的 session 都已准备好，然后运行 M2 的无视频 LOSO 评估：

```bash
python scripts/m2_all_sessions.py \
  --output-root outputs/m2_single_cross_loso_run01
```

用同一批 M2 fold 结果运行配对的 M3 LSTM/GRU 评估：

```bash
python scripts/m3_sequence_models.py \
  --m2-root outputs/m2_single_cross_loso_run01 \
  --output outputs/m3_sequence_models_run01
```

输出目录必须尚不存在。再次运行时使用新的目录名，例如把 `run01` 改为 `run02`；脚本不会覆盖已有实验结果。

### 生成单个 M2 视频 demo

当前 `config.yaml` 将 `session_20260917075744362_8` 设为测试集。下面的命令渲染该 session 后 10 次触摸：

```bash
python scripts/m2_endpoint_models.py \
  --test-session session_20260917075744362_8 \
  --last-n 10 \
  --output outputs/m2_demo_run01
```

查看预测误差和命中率：

```bash
python scripts/baseline_metrics.py outputs/m2_demo_run01
```

数据格式、坐标含义、筛选规则和指标定义以项目文档为准，详见下方链接。

## 仓库结构

```text
collect.html       触点采集页面
config.yaml        设备、数据划分和算法参数
scripts/           采集后处理、review、M1/M2/M3 评估工具
tests/             自动化测试
docs/              研究背景、架构、数据契约、决策和实验记录
demo/              可直接查看的演示视频
```

常用文档：

- [研究问题、当前范围与非目标](docs/PROJECT_CONTEXT.md)
- [系统架构和数据流](docs/ARCHITECTURE.md)
- [数据格式、坐标与时间定义](docs/DATA.md)
- [已接受决策及其依据](docs/DECISIONS.md)
- [实验协议、结果和限制](docs/EXPERIMENTS.md)

## 测试

在已安装项目依赖的 `visual_pretouch` 环境中运行：

```bash
python -m unittest discover -s tests -v
```

模型超参数和固定数据划分维护在 [`config.yaml`](config.yaml)；新增实验请使用新的 `outputs/` 目录，避免覆盖已有结果。

## 当前限制

- 相机视角和手机姿态固定；屏幕四角仍需人工标注。
- 手机遮挡下的手部追踪可能漏检或漂移，需人工 review。
- 当前数据来自单一用户，且历史评估混有两种设备。
- 白闪同步存在时间误差；当前结果是离线可行性评估，不是实时或智能眼镜部署结果。
