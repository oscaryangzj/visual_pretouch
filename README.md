# visual_pretouch

视觉触摸前预测的研究原型。研究问题和 P0 范围见 [项目背景](docs/PROJECT_CONTEXT.md)。

## 当前状态

P0 代码已实现，范围限定为左手拇指从左向右跨越中线并预测右半屏触点。首组 40 次真实数据已跑通旋转、白闪对齐、关键点、B1/B2、评估和渲染闭环；视频中的相机相对手机发生移动，不满足单次静态标定假设，因此尚无可报告的研究结论。

2026-09-16 的 30 次 Mate 80 Pro 会话的数据质量检查已完成：触点与白闪完整、屏幕持续可见；现有全画面识别选错手，裁剪后操作手仍频繁漏检，且静态标定存在漂移。核查图与方法见 [会话检查报告](outputs/review_session_20260916090348042_3_v1/REPORT.md)，正式预测评估尚未开展。

已输出沿用裁剪诊断坐标的 [逐帧拇指标记视频](outputs/thumb_marked_session_20260916090348042_3_v1/thumb_marked.mp4)，用于观察检出点与漏检；复现脚本和配置保存在同目录。

随后进行了保持输入与阈值相同的 [轻量/完整模型对照](outputs/thumb_marked_model1_session_20260916090348042_3_v1/COMPARISON.md)：完整模型也未解决连续漏检，可在报告中查看新标记视频与诊断统计。

2026-09-17 新采集的 [30 次会话检查](outputs/review_session_20260917013532709_2_v1/REPORT.md) 已完成：触点与白闪完整，画面较稳定，全画面轻量 MediaPipe 检出覆盖率 87.5%，优于本次固定裁剪。视频为 4K、25 FPS；仍需核查真实跨线帧、区域标定和毫米尺度，尚未开展落点预测评估。

该会话的 [拇指追踪核查视频](outputs/thumb_marked_session_20260917013532709_2_v1/thumb_marked.mp4) 展示全画面检测点和手机区域放大视图：1080p、25 FPS、1500 帧，保留声音，漏检不补点。配置、坐标及复现脚本保存在同目录。

当前提取器固定使用 MediaPipe 0.10.14 的旧 `mp.solutions.hands` 接口，以上结果均基于该版本；新版 HandLandmarker 尚未对照验证。Git 保存代码、配置、文档及轻量诊断结果，原始数据、派生数据、生成媒体和缓存保留本地，不随提交保存。

## 文档导航

每类信息只在对应文档中维护，其余文档通过链接引用。

| 文档 | 职责 |
| --- | --- |
| [AGENTS.md](AGENTS.md) | 编码代理的工作规则与阅读顺序 |
| [PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) | 为什么做、研究问题、P0 范围和非目标 |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | 模块职责、接口边界和信息流 |
| [DATA.md](docs/DATA.md) | 数据格式、坐标、时间和有效性约定 |
| [DECISIONS.md](docs/DECISIONS.md) | 决策理由、状态与待定问题 |
| [EXPERIMENTS.md](docs/EXPERIMENTS.md) | 实验协议、指标、运行记录与结论 |

当前仓库业务内容包括采集页面、离线 CLI 和上述文档。

## 当前 milestone：M1 最小闭环（流程已跑通，待有效数据验证）

用一段短会话完成采集、视频对齐、拇指跨线预测和可视化评估。统一入口是 `scripts/visual_pretouch.py`。

验收要求：

1. 采集网页导出真实触点，视频中能识别对应白闪事件。
2. 完成人工四角标定、拇指轨迹提取和指定方向的跨线检测。
3. 跑通平均触点与当前拇指尖投影两个 baseline，遵守实验文档的划分和信息边界。
4. 输出逐次预测记录、误差与提前时间统计，以及可以逐帧核查的叠加视频。
5. 保存输入标识、配置快照和代码版本，结果不覆盖。

M1 验证流程是否可信；P0 还需要补齐线性外推、多提前时间评估和正式分析。M1 完成不代表研究假设成立。

## 运行与验证

## 运行

```bash
conda activate visual_pretouch
python -m pip install -r requirements.txt
python scripts/visual_pretouch.py --help
```

采集时在手机浏览器打开 `collect.html`，选择样机（Huawei Pura X、Huawei Mate 80 Pro 或 iPhone 16），点击“开始”，每轮采集 30 次，完成后导出 touch CSV 和 session JSON。确认两个文件均已保存并启动新录像后，点击“重新开始采集”即可开始下一轮，也可先更换样机。页面显示并导出所选设备的配置分辨率、毫米尺寸，同时记录实际浏览器 CSS 尺寸；具体字段和单位见 [DATA.md](docs/DATA.md#屏幕尺寸与设备配置)。独立 HTML 不读取本地 YAML，修改 `config.yaml` 后需同步页面中的设备参数。用视频第一帧人工标定四角；若已有四角像素坐标，也可用 `--points-file` 跳过窗口操作。

典型离线流程如下，输出路径必须是尚不存在的新路径：

```bash
python scripts/visual_pretouch.py calibrate \
  --video data/raw/<session_id>/video.mp4 \
  --output data/derived/<session_id>/calibration.json

python scripts/visual_pretouch.py extract \
  --video data/raw/<session_id>/video.mp4 \
  --calibration data/derived/<session_id>/calibration.json \
  --output data/derived/<session_id>/trajectory.csv

python scripts/visual_pretouch.py align \
  --video data/raw/<session_id>/video.mp4 \
  --calibration data/derived/<session_id>/calibration.json \
  --touches data/raw/<session_id>/touches.csv \
  --output data/derived/<session_id>/alignment.csv

python scripts/visual_pretouch.py predict \
  --trajectory data/derived/<session_id>/trajectory.csv \
  --alignment data/derived/<session_id>/alignment.csv \
  --baseline-touches data/raw/<training_session>/touches.csv \
  --output data/derived/<session_id>/predictions.csv

python scripts/visual_pretouch.py evaluate \
  --predictions data/derived/<session_id>/predictions.csv \
  --touches data/raw/<session_id>/touches.csv \
  --alignment data/derived/<session_id>/alignment.csv \
  --output outputs/<experiment_id>

python scripts/visual_pretouch.py render \
  --video data/raw/<session_id>/video.mp4 \
  --calibration data/derived/<session_id>/calibration.json \
  --trajectory data/derived/<session_id>/trajectory.csv \
  --predictions data/derived/<session_id>/predictions.csv \
  --touches data/raw/<session_id>/touches.csv \
  --alignment data/derived/<session_id>/alignment.csv \
  --method b2_linear --output outputs/<experiment_id>/demo.mp4
```

`evaluate` 会生成 `prediction_details.csv`、`quality.csv`、`metrics.json`、`manifest.json`、配置快照和图表。在 `config.yaml` 的 `devices.<device_id>.region_width_mm`、`region_height_mm` 中填写实测操作区域宽高，单位为毫米；两项齐全后才生成毫米指标。

新采集记录的 `device_id` 自动选择对应配置。旧记录没有该字段时，使用 `config.yaml` 的 `device` 默认值，或在 `evaluate` 命令后加 `--device huawei_mate_80_pro` / `--device iphone_16`。同一次评估只处理一台设备；实际使用的设备和毫米尺寸写入指标与 manifest。

实现前先处理 [待定问题](docs/DECISIONS.md#待定问题)，只解决当前步骤依赖的事项。
