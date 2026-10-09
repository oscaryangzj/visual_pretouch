# 架构

## 职责与状态

本文维护模块边界和信息流，不定义字段、指标或研究范围。P0 使用一个可独立运行的 Python CLI 承载离线阶段，不建立通用框架。

## 模块划分

| 入口 | 职责 | 输入 → 输出 |
| --- | --- | --- |
| `collect.html` | 选择样机、读取浏览器尺寸、显示目标、接收触摸、触摸后闪白、导出记录 | 采集配置 → touch CSV、会话元数据 |
| `scripts/visual_pretouch.py calibrate` | 人工依次点击操作区域四角，保存单应变换 | 视频参考帧 → 标定文件 |
| `scripts/visual_pretouch.py extract` | 顺序提取手部关键点，将拇指尖映射至屏幕坐标 | 视频、标定 → 逐帧轨迹 |
| `scripts/visual_pretouch.py align` | 检测白闪，与触摸记录匹配，输出时间质量信息 | 视频、touch CSV → 对齐表 |
| `scripts/visual_pretouch.py predict` | 按配置识别跨线事件或固定提前时间，用可用历史轨迹生成预测 | 轨迹、配置、可选训练触点 → 预测表 |
| `scripts/visual_pretouch.py evaluate` | 关联真实触点，执行预先约定的评估 | 预测、对齐、touch CSV → 明细、指标和图表、manifest |
| `scripts/visual_pretouch.py render` | 生成可核查的叠加视频 | 视频、标定、轨迹、预测、触点、对齐 → demo video |
| `scripts/baseline_demo.py` | 接入已审核缓存、拟合时间先验，运行本轮三 baseline 并拼接点击片段 | 已审核 session、四角、配置 → 时间先验、预测、指标、三个视频 |
| `scripts/m2_endpoint_models.py` | 从跨线前因果轨迹构造特征，拟合增量中位数、Ridge 和 Gradient Boosting，生成测试预测与视频 | 固定训练／测试 session、审核与四角 → 特征表、模型、预测、指标、视频 |
| `scripts/m3_sequence_models.py` | 构造跨线前因果序列，训练 tiny 单向 LSTM／GRU，并与同折 M2 结果配对 | 已审核 session、M2 LOSO 结果 → 序列、模型、预测、指标 |
| `scripts/baseline_metrics.py` | 从任意 baseline 预测表汇总毫米误差和命中率 | 输出目录或 `predictions.csv` → 终端表格／JSON／CSV |
| `scripts/review_session.py`、`scripts/review_session.html` | 本地浏览器显示逐次追踪，保存保留／舍弃选择与四角标注 | session、可选已有追踪与事件 → 机器缓存、审核文件、标定文件 |

历史阶段按子命令运行；本轮 demo 用独立小入口连接已审核数据，复用 `scripts/visual_pretouch.py` 的读取、投影与跨线函数。文件字段以 [DATA.md](DATA.md) 为准。

`calibrate`、`extract`、`align` 和 `render` 必须通过同一个视频读取入口打开视频。该入口应用视频流的显示旋转元数据，保证四个阶段看到相同方向、尺寸和像素坐标；不生成或覆盖一份旋转后的原始视频。

## 信息流与预测边界

```text
camera video ─→ calibration ─→ thumb trajectory ─→ crossing ─→ prediction
      │                                                          │
      └─→ flash detection ─┐                                     │
touch CSV ────────────────→ alignment ─────────────────────────→ evaluation
                                                                 │
                                          video + intermediates ─→ demo
```

跨线预测接口只接收截至当前帧的轨迹、预先固定的配置，以及训练数据产生的模型参数。它不得接收当前试次的真实触点、目标坐标、未来轨迹或距真实触摸的剩余时间。

当前 demo 的 B1/B2 遵守以上接口。用户明确要求的 B3 通过独立、显式命名的 oracle 入口接收当前试次对齐触摸时间，仅作为事后诊断；与 B2 共用历史速度估计和最终坐标约束，不接收真实触点位置或未来轨迹。该例外不进入 B1/B2 的输入或参数拟合。

对齐表用于离线关联和评估；固定提前时间的诊断由评估侧选择历史截止帧，再调用相同预测接口，不能把真实触摸时间传入预测器作为特征。

轨迹平滑、缺失值处理和速度估计也必须遵守截止帧边界。提取器按时间正序运行，避免双向平滑或未来帧插值。渲染器可以显示事后标签，但这些标签不能回流到预测输入。

## 配置与产物

### 当前 demo 入口

1. 校验 session 的原始文件、已审核缓存和 `review.csv` 身份；将缓存中的连续追踪坐标投影成轨迹，不重新运行 MediaPipe。完整点击序列负责动作边界，人工选择负责本次参与的试次。`config.yaml:data_split` 固定训练、验证和测试 session，测试目标必须属于 `test_sessions`。
2. 读取 review 中保存的实际网页操作区域标定，按原始触摸顺序解析单独标注和继承关系。当前动作全部图像坐标使用该动作同一单应变换，再检测跨线和估计速度；不将不同变换下的坐标连起来检测跨线。渲染也使用本次变换和边界。
3. 顺序提取跨线事件及截至跨线的历史速度，再应用配置中的统一速度衰减系数。时间先验拟合模块汇总指定训练 session 中符合条件的已审核试次，输出冻结的时间先验文件；B2 预测只读取此先验。追加审核结果时通过新处理版本重拟合，不自动修改既有结果。
4. B1/B2 和独立 B3 oracle 分支生成预测，通过同一屏幕约束函数，保存约束前后结果。数据字段由 DATA 维护，算法和分块协议由 EXPERIMENTS 维护。
5. 评估分开列出可用信息方法与 oracle 对照，渲染同试次三个方法。跨线时冻结预测，实际点击发生后才显示 GT、误差和提前时间；试次切换清除上次状态，缺失处断开轨迹。边界上的预测标记绘制也限制在操作区域内。

上述链路由 `scripts/baseline_demo.py` 承载。训练统计、因果位置预测与显式 oracle 入口分开；渲染器逐试次清除预测，按同一时间窗输出三份视频。标定缺失时停止；其他训练 session 的不可用输入记录原因，不强行合并。审核文件保持原样，派生文件写入新的实验目录；无需重新标记保留／舍弃。

### M2 直接落点入口

M2 复用 `baseline_demo.prepare_session` 得到逐 touch 的固定四角映射、全部左→右跨线、审核状态和投影轨迹，不重新提取 MediaPipe。按 `prediction.crossing_policy: single_only` 只纳入触摸前恰好一次左→右跨线的样本；因果特征只读取当前动作开始至跨线帧的轨迹，标签是 `touch_uv - crossing_uv`。

```text
reviewed train sessions ─→ causal feature rows ─→ grouped CV ─→ frozen models
reviewed test session  ─→ causal feature rows ───────────────→ prediction
crossing position + predicted delta ─→ raw endpoint ─→ clip [0,1] ─→ metrics/video
```

训练集增量中位数不做模型拟合。Model A 使用训练集缺失值中位数填充、缺失指示、特征标准化和 Ridge；Model B 使用同样的训练集填充与缺失指示，再对 u/v 增量分别拟合 Gradient Boosting。模型选择按 `session_id` 分组交叉验证，任何 fold 都不能把同一 session 同时放入训练和验证。测试 session 不参与填充、标准化、参数选择或最终拟合。

M2 的 `single_only` 10-session LOSO 和 M3 的配对 10-fold LOSO 均已完成。M2 Model B 在当前协议下宏平均 mean／median error 为 10.87／9.28 mm；M3 LSTM／GRU 均未超过 Model B。完整表格与限制见 [EXPERIMENTS.md](EXPERIMENTS.md)。

### M3 小型时序模型入口

`m3_sequence_models.py` 复用 M2 的审核数据、四角映射、`single_only` 样本 ID、标签、LOSO 划分和裁剪／指标函数。只取跨线前 300 ms，构造 10 个时间点的相对坐标与有效掩码，分别训练一层、hidden size 16 的单向 LSTM 和 GRU。每个 LOSO fold 的预处理只在训练 session 拟合；测试 session 不进入标准化、early stopping 或模型训练。该入口只生成量化结果和模型产物，不渲染视频。

`config.yaml` 是超参数入口：轨迹窗口、检测阈值、外推时间、目标采样设置和随机种子等随实现加入，不在脚本中散落硬编码默认值。已确定的坐标语义不能作为随意调参项。

设备配置集中在 `devices`。`evaluate` 根据 touch CSV 的设备标识选择物理尺寸；旧记录可由默认设备或 `--device` 指定。评估产物保留实际选择的设备配置。独立 HTML 内嵌设备参数和采集配置的快照，可直接打开，不读取本地 YAML；配置修订后须同步页面。页面显示并导出配置中的分辨率和毫米尺寸，另行读取浏览器 CSS 尺寸，不推算原生分辨率。

会话事实、标定结果和实验配置分开保存，格式见 [DATA.md](DATA.md)。每次实验保存配置快照并使用新输出目录；已有原始数据与实验产物不覆盖。

## 检查边界

每一步保留可检查的中间文件。标定失败、关键点缺失或白闪匹配不确定时，输出明确状态，不生成看似有效的默认坐标。实验侧统一报告失败与排除原因。

## 逐次 touch 审核工具

`scripts/thumb_tracking.py` 是提取器与审核准备共用的连续追踪模块。MediaPipe 提供候选拇指点；模块用前一帧的已跟踪位置与当前图像计算 LK 光流，并作反向验证。候选与实际图像运动接近时接受模型定位，候选跳到其他手指或模型漏检时仅在有效期限内接受图像追踪。丢失后需连续稳定的候选才能重新初始化。模块只使用当前及过去帧，不使用点击目标、白闪事件或未来帧。

Python 标准库提供仅监听本机的 HTTP 服务，普通 HTML／JavaScript 提供审核页面，不引入 Web 框架。浏览器直接播放原始视频，服务支持 Range 请求以便跳转；Canvas 根据已保存的逐帧时间戳叠加拇指点和短历史轨迹。视频方向和追踪尺寸须一致，漏检不补点，轨迹在缺失处断开。

读取 session 的 touch CSV，以及白闪诊断／对齐表。白闪来源可显式指定；只有一份诊断时自动发现，有歧义则要求指定。追踪默认由 `scripts/review_prepare.py` 使用当前配置按时间顺序提取整段视频的拇指图像坐标和亮度；旧追踪 CSV 仅在显式指定时使用，避免未记录模型的历史坐标绕过配置校验。亮度上升候选需同时满足事件数及与原始点击相对时间的一致性；失败则提示指定屏幕亮度采样 ROI 或已有对齐表，不强行顺序配对。ROI 只用于采样亮度，不改变跟踪输入，也不执行自动屏幕检测。准备结果合并为一个 session 内的隐藏缓存，复用前校验源文件哈希、模型版本和跟踪参数；无需每个 session 编写脚本。

页面显示全部点击的编号和选择、当前片段、追踪来源及白闪相对时间。绿色显示模型定位，蓝色显示光流追踪；可切换原始模型坐标，检查实际改变。叠加优先使用浏览器已呈现帧的时间戳，跳转期间清除旧叠加。支持点击选择试次、播放／暂停、逐帧、慢放、重播、扩大预览和放大查看。保留／舍弃各为一个按钮，决定立即保存并进入下一次；可回看修改或撤销，无需舍弃原因和备注。

缺少标定时，页面先固定到第一次触摸，暂停视频并要求依次点击操作区域四角；第一个角点落下后锁定参考帧。后端校验点序、边界和变换，原子保存文件。后续保留项可在当前片段单独标注、重标或恢复沿用上一触摸；页面叠加实际生效的边界并显示来源，可跳转下一个保留项。后端统一解析继承关系，供页面与 demo 共用。该流程复用已有追踪，不修改审核选择；文件格式和旧版兼容由 DATA 定义。

保留／舍弃决定只写 session 目录的审核 CSV；四角写独立标定 JSON；自动准备使用一个机器缓存。格式与续审语义由 [DATA.md](DATA.md#人工审核数据) 维护。旧版审核清单仅有一份且新文件不存在时，迁移其已有选择；旧产物保留。后续分析按 touch 标识读取选择，不能依靠行号关联。
