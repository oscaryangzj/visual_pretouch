# visual_pretouch

视觉触摸前预测的研究原型。研究问题和 P0 范围见 [项目背景](docs/PROJECT_CONTEXT.md)。

## 当前状态

P0 代码已实现，范围限定为左手拇指从左向右跨越中线并预测右半屏触点。首组 40 次真实数据已跑通旋转、白闪对齐、关键点、B1/B2、评估和渲染闭环；视频中的相机相对手机发生移动，不满足单次静态标定假设，因此尚无可报告的研究结论。

2026-09-16 的 30 次 Mate 80 Pro 会话的数据质量检查已完成：触点与白闪完整、屏幕持续可见；现有全画面识别选错手，裁剪后操作手仍频繁漏检，且静态标定存在漂移。核查图与方法见 [会话检查报告](outputs/review_session_20260916090348042_3_v1/REPORT.md)，正式预测评估尚未开展。

已输出沿用裁剪诊断坐标的 [逐帧拇指标记视频](outputs/thumb_marked_session_20260916090348042_3_v1/thumb_marked.mp4)，用于观察检出点与漏检；复现脚本和配置保存在同目录。

随后进行了保持输入与阈值相同的 [轻量/完整模型对照](outputs/thumb_marked_model1_session_20260916090348042_3_v1/COMPARISON.md)：完整模型也未解决连续漏检，可在报告中查看新标记视频与诊断统计。

2026-09-17 新采集的 [30 次会话检查](outputs/review_session_20260917013532709_2_v1/REPORT.md) 已完成：触点与白闪完整，画面较稳定，全画面轻量 MediaPipe 检出覆盖率 87.5%，优于本次固定裁剪。视频为 4K、25 FPS；仍需核查真实跨线帧、区域标定和毫米尺度，尚未开展落点预测评估。

该会话的 [拇指追踪核查视频](outputs/thumb_marked_session_20260917013532709_2_v1/thumb_marked.mp4) 展示全画面检测点和手机区域放大视图：1080p、25 FPS、1500 帧，保留声音，漏检不补点。配置、坐标及复现脚本保存在同目录。

当前提取器固定使用 MediaPipe 0.10.14 的 `mp.solutions.hands` 接口，默认使用完整版（`tracking.model_complexity: 1`）。以上历史结果保留各自配置；第 9 次点击的同帧骨架核查发现轻量模型混淆手指，完整版改善了抽查帧定位，详见 [实验记录](docs/EXPERIMENTS.md)。Git 保存代码、配置、文档及轻量诊断结果，原始数据、派生数据、生成媒体和缓存保留本地，不随提交保存。

完整版仍会在拇指和下方手指之间跳变。当前提取与审核共用 `scripts/thumb_tracking.py`：通过 OpenCV 光流连续追踪同一指尖，拒绝与图像运动明显不符的模型点；跟踪失效时留空。原始模型点保留，审核页面可切换“连续追踪／MediaPipe 原始点”，并用蓝色区分光流结果。处理约定与限制见 DATA，验证见 EXPERIMENTS。

2026-09-17 用户确认：当前“完整版 MediaPipe + 光流连续追踪”的效果暂时够用，采用本版继续逐次 touch 人工审核，保存保留／舍弃选择，再开展保留试次的跨线与落点分析。该确认是当前工具可用性的判断，不代表人工审核已完成、整段指尖准确率已验证或 M1 已验收；反馈记录见 [EXPERIMENTS.md](docs/EXPERIMENTS.md#实验登记)。

后续 `_8` 会话暴露了长段无法重新定位的问题。当前手掌检测与关键点接受门槛同时设为 0.2，保留完整版和光流检查；该组完全无追踪的点击窗口由 12/30 减至 1/30，但仍有长段缺失，不能视为已全面解决。对照和限制见 [漏检诊断](outputs/mediapipe_gap_audit_session_20260917065914185_8_v1/REPORT.md)。

当前白盒 baseline 的数据划分写在 `config.yaml:data_split`：`session_20260917075240834_5` 为测试集，其余 9 个已完成标注的 session 为训练集，验证集为空。此前 `session_20260917075744362_8` 的 28 次保留、2 次舍弃结果仍保存在原 session。

当前分支为 `m2-direct-endpoint`。M2 初版已实现并跑通：训练集增量中位数、Model A（Ridge）和 Model B（Gradient Boosting）都直接预测“跨线点到最终触点”的二维增量。依赖、自动测试、真实训练、量化和视频抽帧核查均已完成；当前结果是 `_5` 后 10 次的小规模测试 demo，不代表完整测试集的正式泛化结论。

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

## 当前 milestone：M2 直接落点（初版已跑通）

M2 保持 P0 的首次左→右跨线预测时刻和固定数据划分，比较三种方法：训练集二维增量中位数、Ridge、Gradient Boosting。输入仅限跨线及之前最多 300 ms 的归一化拇指轨迹特征；测试 session `_5` 不参与填充、标准化、选参或训练。完整方法、候选参数、指标和验收顺序见 [M2 实验协议](docs/EXPERIMENTS.md#m2-直接落点实验)，模块边界和数据字段分别见 [ARCHITECTURE.md](docs/ARCHITECTURE.md#m2-直接落点入口) 与 [DATA.md](docs/DATA.md#m2-样本与预测数据)。

当前已实现并验证：`scripts/m2_endpoint_models.py`、`config.yaml:m2`、scikit-learn 依赖声明，以及 baseline 评估／视频渲染的复用改动。

接下来按以下顺序推进：

1. 在完整 `_5` 测试 session 上重复量化，避免只看后 10 次。
2. 用更多独立测试 session 重复验证训练集增量中位数、Ridge 和 Gradient Boosting。
3. 根据重复结果判断是否进入 M3 的时序模型探索。
4. 继续保持每次运行的输入、配置、git commit 和输出目录可追溯。

M2 入口会自动读取 `config.yaml:data_split.test_sessions` 中唯一的测试 session；不传 `--last-n` 时评估该 session 的全部 touch。输出目录默认自动生成，不会覆盖已有结果，名称格式为 `outputs/m2_direct_endpoint_v1_<YYYYMMDDHHMMSS>/`。复现后 10 次 demo 可运行：

```bash
conda activate visual_pretouch
python scripts/m2_endpoint_models.py \
  --last-n 10
```

运行完整测试 session：

```bash
python scripts/m2_endpoint_models.py
```

如需固定输出目录，可继续传入 `--output outputs/<experiment_name>`；该目录必须不存在。

## 上一 milestone：M1 最小闭环（流程已跑通，待有效数据验证）

用一段短会话完成采集、视频对齐、拇指跨线预测和可视化评估。统一入口是 `scripts/visual_pretouch.py`。

验收要求：

1. 采集网页导出真实触点，视频中能识别对应白闪事件。
2. 完成人工四角标定、拇指轨迹提取和指定方向的跨线检测。
3. 按当前 demo 协议跑通 B1/B2，并单列 B3 oracle 对照；方法与信息边界见实验文档。
4. 输出逐次预测记录、误差与提前时间统计，以及可以逐帧核查的叠加视频。
5. 保存输入标识、配置快照和代码版本，结果不覆盖。

M1 验证流程是否可信；P0 仍需完成有效数据上的多提前时间评估和正式分析。M1 完成不代表研究假设成立。

### 三 baseline demo（代码与真实会话视频已生成）

本轮方法为 B1 跨线投影、B2 统计时间线性外推、B3 真实时间线性外推。当前按 `config.yaml:data_split` 固定训练和测试 session，B2 中位数只从训练 session 汇总；本次视频展示测试 session 的后 10 次触摸。公式、训练数据使用规则、屏幕约束及 oracle 限制统一见 [EXPERIMENTS.md](docs/EXPERIMENTS.md#baseline)，不沿用旧 B3 Ridge 编号。

`scripts/baseline_demo.py` 复用审核缓存及四角文件，生成训练时间先验、三方法预测、误差明细和同一组触摸的三个视频。当前由 `data_split.training_sessions` 固定训练清单，测试目标必须属于 `data_split.test_sessions`，不会把测试 session 混入训练统计。模块边界见 [ARCHITECTURE.md](docs/ARCHITECTURE.md#当前-demo-入口)。

验收要求：三方法同一跨线帧触发并冻结；B2 共用可追溯的训练时间先验；demo 明示训练集结果，B3 明示 oracle；最终预测点在屏幕内；实际点击后显示真实触点与误差；保留试次中的预测失败如实展示。数学边界、自动测试与真实会话抽帧核查已通过；当前仍是训练集开发 demo。

复现当前测试集 demo（输出目录必须不存在）：

```bash
conda activate visual_pretouch
python scripts/baseline_demo.py \
  --session-dir dataset/session_20260917075240834_5 \
  --last-n 10 \
  --output outputs/baseline_demo_session_20260917075240834_5_last10_v7
```

每个视频拼接后 10 次原始触摸的片段，保留失败状态；源时间窗保存在 manifest。输出包含三个 `demo_*.mp4`、`predictions.csv`、`trajectory.csv`、`metrics.json`、`time_prior.json`、标定和配置快照、代码补丁与入口快照。视频不含音轨，默认 0.5 倍速。缺少目标 session 的标定时直接提示补标，不新建结果目录。以下旧 CLI 命令仍用于历史 baseline。

量化已有 baseline 输出时，传入输出目录即可。默认报告毫米平均／中位误差及 hit@10/15/20；`--format json` 或 `--format csv` 可输出机器可读结果：

```bash
python scripts/baseline_metrics.py \
  outputs/baseline_demo_session_20260917075240834_5_last10_v4
```

### 逐次 touch 可视化审核（已实现）

`scripts/review_session.py` 在本地浏览器页面中逐次查看拇指追踪，点击“保留”或“舍弃”。支持任意 session：按当前完整版 MediaPipe 配置准备，复用参数一致的机器缓存及已有白闪诊断。页面显示实际使用的模型。人工选择保存 session 内的 `review.csv`，四角标注单独保存到 `calibration.json`；首次自动处理另外保留一个隐藏的机器缓存，避免重开时重复推理。

第一阶段验收要求：

1. 全部 touch 都有审核入口，显示视频、拇指尖、漏检和当前审核状态。
2. 支持播放、暂停、慢放、逐帧、重播及前后试次切换。
3. 点击保留或舍弃后立即保存，能够续审和修改决定，无需填写原因。
4. 一个文件包含全部 touch 的选择，原始数据保持不变。

该工具服务于 M1 的有效数据检查，不改变研究任务或表示 M1 已验收完成。模块与交互见 [ARCHITECTURE.md](docs/ARCHITECTURE.md#逐次-touch-审核工具)，字段、预览时间和存储约定见 [DATA.md](docs/DATA.md#人工审核数据)，子集使用规则见 [EXPERIMENTS.md](docs/EXPERIMENTS.md#人工审核与子集分析)。

当前会话的审核命令：

```bash
conda activate visual_pretouch
python scripts/review_session.py \
  --session-dir dataset/session_20260917075240834_5
```

把 `--session-dir` 换成任意采集目录（一个 MP4 和一个 `*_touches.csv`；也接受 `touches.csv`）。首次处理会显示进度，完成后自动打开本地浏览器页面，按点击编号跳转；保留／舍弃后自动进入下一次。提供播放、慢放、逐帧、重播和扩大时间窗口。再次运行相同命令会读取 `<session>/review.csv`，定位第一个未审核项。关闭页面后可在终端按 Ctrl+C 停止服务。

没有 `calibration.json` 时先进入第一次触摸。选择四角清楚的暂停帧，依次点击网页操作区域的左上、右上、右下、左下，再点“保存四角”；可以放大查看、撤销角点或重标。已有 `review.csv` 的决定不会清空，保存首次四角后回到第一个未审核项。

之后每个保留的触摸都可点“标注本次四角”，保存后留在本次；未单独标注的触摸自动沿用上一次的四角。页面显示实际沿用哪一次，“下一个保留项”可跳过舍弃项。已有单独标注可重新标注，或点“沿用上一次四角”撤销本次四角；只影响后续未单独标注项。全部四角仍只存 session 内同一个 JSON，旧版单次标定也能直接读取。三 baseline 按触摸使用对应四角；同一次触摸内仍使用固定映射，不能补偿动作中的屏幕移动。

已有白闪诊断只有一份时自动发现；存在多份时，用 `--events PATH` 明确选择。旧追踪 CSV 只通过 `--tracking PATH` 显式使用，避免自动混入轻量模型结果。默认自动顺序提取拇指和亮度，白闪数量及相对点击时间均符合配置容差才进入审核；失败时保留已提取坐标，提示配置 `review.preparation.flash_roi_uv` 或传入核查过的 `--events`。不会截断多余事件或强配漏闪。`--prepare-only` 只完成准备和检查，不启动页面。已有旧版审核清单且仅有一份时，首次打开会迁移其选择。`--no-open` 只打印地址，不自动打开浏览器。原始 HEVC 视频需使用支持该编码的浏览器，例如 macOS Safari。

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
