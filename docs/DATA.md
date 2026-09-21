# 数据约定

## 职责与状态

本文是数据语义、坐标和时间的唯一维护位置。当前实现遵守以下数据契约；已有真实采集数据，质量检查与运行记录见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 坐标与 ground truth

* 标准操作区域左上角为 `(u,v)=(0,0)`，右下角为 `(1,1)`；u 向右、v 向下。
* 中线是 `u=0.5`。P0 会话固定记录 `hand=left`、`crossing_direction=left_to_right`，左右以标准操作区域为准，不以相机画面的左右为准。
* 拇指从中线左侧向右跨越；采集目标限于右半屏 `0.5<u<=1`、`0<=v<=1`。中线上的触点不属于右半屏。具体起始姿态和复位流程仍待确定。
* GT 是浏览器记录的实际触摸开始位置，不能用显示目标位置替代。目标坐标仅用于采集检查，不作为预测特征。
* 自 2026-09-17 起，新采集目标整体限制在操作区域下方 2/3（避开顶部 1/3）；中心位置额外预留目标半径，纵向上限为 `v<0.92`。参数维护于 `config.yaml` 的 `collection`，独立采集页面内嵌相同配置并随 session JSON 导出。此前会话的采样范围保持原样。
* 浏览器坐标使用操作区域内的 CSS 像素：`u=(clientX-left)/width`，`v=(clientY-top)/height`；同时保存当时的区域尺寸和位置。
* 设备像素分辨率、浏览器 CSS 尺寸和毫米物理尺寸分别记录，定义见下节；触点归一化始终使用当次采集区域的 CSS 尺寸。
* 人工标定区域必须与网页归一化区域一致。四角顺序：左上、右上、右下、左下；相机图像坐标原点在左上，单位为像素。
* 单应变换得到的是拇指尖的图像投影位置，不是指尖在屏幕平面上的真实三维位置。区域外投影保留原值，不静默裁剪到 `[0,1]`。
* 毫米评估需要实测该操作区域宽高 `region_width_mm`、`region_height_mm`；缺失时毫米指标标记不可用，不用分辨率猜测。

## 屏幕尺寸与设备配置

当前采集格式为 v3：分辨率与毫米尺寸使用用户填写的设备配置，移除 v2 的 DPR 估算字段。原有 v1/v2 文件保持不变，离线流程继续支持。

| 字段 | 定义与来源 |
| --- | --- |
| `device_id` | 开始采集前手动选择的设备，单会话固定 |
| `device`（session JSON） | 对应设备显示名称 |
| `screen_width_css_px`、`screen_height_css_px` | 浏览器 `screen.width/height` 报告的屏幕 CSS 像素宽高 |
| `device_pixel_ratio` | 浏览器 `devicePixelRatio` |
| `screen_width_px`、`screen_height_px` | 所选设备配置中的原生像素分辨率，与浏览器 CSS 尺寸分别保存 |
| `region_width_mm`、`region_height_mm` | 所选设备配置中的操作区域物理宽高，不由浏览器推算 |
| `region_left/top/width_css_px/height_css_px` | `phone-area.getBoundingClientRect()` 得到的实际采集区域，用于触点归一化 |
| `fullscreen` | 当时浏览器是否进入全屏；不支持全屏时保留实际页面区域 |

touch CSV 在每次触摸时读取浏览器尺寸，并保存该会话所选设备的配置尺寸；session JSON 保存开始时进入全屏尝试之后的尺寸快照。v2 的 `screen_width_estimated_px/height_estimated_px` 仅是历史估算字段，不改写为原生分辨率。浏览器 CSS 尺寸与 DPR 仍记录，但不再用于推算设备分辨率。

每轮采集是独立 session。发起 touch CSV 和 session JSON 下载后可重新开始：保留样机选择，新一轮清空内存记录、重置 trial_id 并生成新 session_id，重新读取尺寸快照。网页无法获知文件是否保存成功，需在重开前确认两个文件均已保存。旧会话文件和采集数量不变。

`config.yaml` 的 `devices` 按 `huawei_pura_x`、`huawei_mate_80_pro`、`iphone_16` 分组。三台设备的原生分辨率和毫米尺寸均由用户填写；具体数值以配置为准。各设备 `region_width_mm`、`region_height_mm` 对应实测采集区域物理宽高；全屏时对应屏幕显示区，不能填机身尺寸，非全屏时须对应实际网页区域。独立 HTML 内嵌同一份设备参数快照，配置变化后需同步页面。

评估优先使用 touch CSV 的 `device_id`，没有时使用配置中的 `device` 默认值，也可通过 `--device` 明确指定。混合设备或显式参数与记录冲突时报错。毫米评估仍读取选中设备的配置并写入指标和 manifest；v3 原始导出另外保留采集时的尺寸快照，后续配置修订不修改原始记录。旧版配置快照仍可使用原 `evaluation.region_width_mm/height_mm`。

## 时间与事件

* 视频帧进入标定、关键点提取、白闪检测和渲染前，统一应用视频流的显示旋转元数据；后续 `image_size_px`、四角点和关键点像素坐标都以旋转后的显示方向为准。原始视频保持不变，不通过重编码覆盖方向信息。
* 视频帧编号从 0 开始，时间使用相对首帧的毫秒时间戳；优先使用视频实际时间戳，不默认按 60 FPS 推算。
* 浏览器触摸时间使用单调时钟并保存单位；与视频时钟分属不同时间轴，不能直接相减。
* `flash_frame` 表示检测到白闪的帧；`touch_time_video_ms` 表示对齐后估计的触摸时刻，两者不能无条件视为相同。
* 保存对齐方法、估计延迟和时间不确定性；未校准白闪延迟时必须标为近似对齐。白闪用于关联标签，不作为跨线触发信息。
* `prediction_time_ms` 是跨线检测在当时已有帧上能够触发的时刻。跨线比较、去抖和多次跨线选择规则见待定决策，实施前需固定。
* 没有跨线、跨线不早于触摸、关键点缺失和对齐不确定都要显式记录，不能伪造事件。

## 存储布局

```text
data/raw/<session_id>/
  video.mp4                 原始视频（保留原始编码；实际扩展名可不同）
  touches.csv               原始网页导出
  session.json              设备、参与者匿名编号、手别、区域、采集协议
data/derived/<session_id>/<processing_id>/
  calibration.json
  trajectory.csv
  alignment.csv
  quality.csv
outputs/<experiment_id>/
  manifest.json
  config.yaml               本次运行的配置快照
  predictions.csv
  metrics.json
  figures/
  demo.mp4
```

实现中 `quality.csv` 由评估阶段生成在实验输出目录，记录对齐、ground truth 和每个预测组合的去向；如需把它放入派生目录，可直接复制而不修改内容。

原始文件不可原地编辑；校正通过新派生版本表达。处理和实验 ID 必须唯一，目录已存在时不覆盖。暂不创建空目录占位。

## 最小字段契约

CSV 使用 UTF-8、表头和明确单位；缺失数值留空，另列原因，不用 0 表示缺失。各文件保存 `schema_version`（CSV 可由关联元数据保存）。

| 文件 | 必需内容 |
| --- | --- |
| `session.json` | session_id、匿名 participant_id、hand、crossing_direction、device_id、device、屏幕与采集区域尺寸快照、区域物理尺寸（未知为 null）、采集配置和试次协议 |
| `touches.csv` | session_id、device_id、trial_id、touch_index、touch_time_browser_ms、client_x/y、上述屏幕尺寸字段与 fullscreen、region_left/top/width_css_px/height_css_px、touch_u/v、target_u/v |
| `calibration.json` | session_id、参考帧号、图像尺寸、四角像素坐标、3×3 homography、操作区域定义 |
| `trajectory.csv` | frame_index、video_time_ms、thumb_tip_x/y_px、thumb_u/v、valid、invalid_reason；关键点方法及版本保存在处理元数据中 |
| `alignment.csv` | trial_id、touch_index、flash_frame、flash_time_video_ms、touch_time_video_ms、alignment_method、uncertainty_ms、status |
| `quality.csv` | trial_id（帧级问题可另加 frame_index）、阶段、状态、原因；保留所有试次的去向 |
| `predictions.csv` | trial_id、method、prediction_mode、prediction_frame、prediction_time_ms、pred_u/v、status、failure_reason |
| `manifest.json` | experiment_id、session_ids、输入文件及哈希、processing_id、配置快照路径、git commit、dirty 状态、运行命令、运行环境/依赖版本、输出路径 |

正式实验优先使用已提交的代码；未提交改动参与运行时需另存代码补丁（含参与运行的未跟踪代码），仅记录 dirty=true 不足以复现。

### 连续追踪坐标（trajectory v2）

当前 `thumb_tip_x/y_px` 是连续追踪的图像坐标，投影至 `thumb_u/v` 的坐标定义不变。新增 `raw_thumb_tip_x/y_px` 保存未处理的 MediaPipe 第 4 点，`tracking_source` 取 `mediapipe`、`optical_flow` 或 `missing`。光流点由相邻真实图像计算，不是沿用旧位置或未来插值；光流失效或超过期限时 `valid=0`、坐标留空，原因 `thumb_tracking_lost`。旧 v1 文件仍保留原语义。

连续追踪参数集中在 `tracking.continuity`，像素阈值定义在该模块缩小后的输入图像上。需通过前后向误差和光度误差检查；模型候选偏离图像跟踪点超过匹配半径则不接受。纯光流最多持续配置的 500 ms，重新定位要求连续 3 帧候选稳定且靠近最后跟踪位置。参数是当前开发会话上选取的值，尚未跨会话验证。初始定位仍依赖模型，光流也可能漂移；不能把有输出等同于准确。

MediaPipe 的 `min_detection_confidence` 控制手掌检测接受门槛，`min_tracking_confidence` 控制关键点模型的手部存在接受门槛；后者不是每个指尖的定位准确度。当前两者均为 0.2，以减少握持场景中的重新定位失败，未改变连续追踪判定。旧 0.5 轨迹保留其处理身份，不混用人工选择；阈值对照见 EXPERIMENTS。

此变更明确改变关键点处理和缺失判定，不改变 GT、中线、预测时间或自动保留规则。正式实验需固定配置及处理版本，不能混用旧原始轨迹与新连续轨迹。

## Demo 预测数据

本节定义新版派生数据，不改写已生成的旧预测文件。算法、统计量和训练数据使用规则的唯一维护位置为 [EXPERIMENTS.md](EXPERIMENTS.md#baseline)。

### 跨线与时间语义

`crossing_time_ms` 是首次有效左→右跨线采样帧的原始视频时间戳：相邻有效点满足前点 `u<=0.5`、后点 `u>0.5`，间隔不超过 `tracking.max_gap_ms`。使用后点所在帧，不插值到中线，不事后选择最后一次跨线。动作边界沿用完整点击序列和配置的复位间隔，审核舍弃不会合并相邻动作。

三种方法的 `prediction_frame/time_ms` 均等于该跨线帧及其时间，B3 的时间例外不改变触发时刻。`horizon_ms` 表示本次外推时长；B1 为 0，B2 来自冻结时间先验，B3 来自当前对齐触摸时间减跨线时间。B3 的值虽以真实剩余时间命名，仍具有对齐时间的不确定性。

### 时间先验文件

新处理目录内的 `time_prior.json` 保存 `schema_version`、统计量名称、`prior_horizon_ms`、实际贡献的 `(session_id,trial_id,touch_index)` 清单和样本数、各样本剩余时间及其分布摘要。记录训练 session 清单、各 session 的审核清单哈希、轨迹／对齐标识及配置快照引用；跨 session 不能只靠 trial_id 关联。没有有效样本时保存失败状态，时长留空；不写入原始 session 文件。

### 预测表 v4

保留旧预测表的试次、方法、预测模式、帧、时间、最终 `pred_u/v`、状态和失败原因。新增字段：

| 字段 | 语义 |
| --- | --- |
| `raw_pred_u/v` | 边界约束前的预测，允许有限越界值 |
| `clipped` | 有效预测是否被边界裁剪：1／0；失败时留空 |
| `horizon_ms`、`horizon_source` | 外推时长与来源：`none`／`training_median`／`oracle_touch_time` |
| `velocity_u_per_ms`、`velocity_v_per_ms` | B2/B3 共用的原始历史速度，单位为归一化坐标／ms；B1 留空 |
| `velocity_scale` | B2/B3 共用的速度衰减系数；B1 留空 |
| `applied_velocity_u_per_ms`、`applied_velocity_v_per_ms` | `velocity_scale × velocity`，即实际参与外推的速度；B1 留空 |
| `uses_oracle_time` | 方法是否使用当前试次事后触摸时间；B3 为 1，B1/B2 为 0 |
| `split` | 数据集归属；由 `config.yaml:data_split` 固定。训练先验记录为 `train`，测试 session 的预测与指标记录为 `test` |
| `calibration_source_trial_id`、`calibration_source_touch_index` | 本次实际使用的四角来自哪个原始 touch |
| `calibration_inherited`、`calibration_reference_frame` | 是否沿用其他 touch 的四角、该标注的参考帧 |

最终有限 `pred_u/v` 均在 `[0,1]` 内，主评估和渲染使用这些坐标。原始轨迹 `thumb_u/v` 保持原语义，不作预测边界裁剪。失败坐标留空，不能产生默认边界点。稳定方法标识和公式见 EXPERIMENTS；schema v4 增加速度衰减字段，防止与未衰减的既有输出混用。

当前 demo 的 `trajectory.csv` 按动作保存投影结果，增加 `trial_id`、`touch_index` 和 `calibration_source_trial_id`，以 `(trial_id,touch_index,frame_index)` 标识一行。每个动作内使用同一映射；覆盖动作开始至点击后预览结束的源帧，缺失帧保留并标记无效。原生像素轨迹和审核缓存不改变。实验目录的 `calibration.json` 是完整四角文件快照，可核对每次使用的变换；旧 CLI 的单次 `extract` 仍只使用顶层静态映射。

### M2 样本与预测数据

M2 的样本单位是一次原始 touch，以 `(session_id, trial_id, touch_index)` 唯一标识。`split` 来自固定 `data_split`；训练样本仅纳入 `keep=1`、GT 有效且触摸前存在首次左→右跨线的试次。舍弃、无效 GT 和无跨线试次仍保留排除原因，测试评估不能静默删除模型失败。

标签定义为 `delta_u=touch_u-crossing_u`、`delta_v=touch_v-crossing_v`。标签只用于训练或事后评估，不进入测试特征。特征的最大时间戳必须满足 `max_feature_time_ms <= prediction_time_ms`；禁止读取真实剩余触摸时间、跨线后的轨迹、当前试次目标位置或 GT。

首版特征以归一化屏幕坐标计算，包括跨线位置、动作开始至跨线的时长，以及跨线前 50/100/200/300 ms 内的 u/v 最小二乘速度、u/v 位移和有效点数；另记录 300 ms 内的有效比例、最大有效点间隔、轨迹长度、直线度，以及 50 ms 与 300 ms 速度模长之差。轨迹窗口只使用不超过 `tracking.max_gap_ms` 的最近连续有效段，缺失特征保留为空值。填充值和缺失指示只能由训练 fold 或最终训练集拟合。

M2 输出约定：

| 文件／字段 | 语义 |
| --- | --- |
| `samples.csv` | 实际训练和测试样本、因果特征、标签、纳入状态、设备物理尺寸 |
| `feature_schema.json` | 特征顺序、标签定义、因果截止点和缺失值规则 |
| `model_selection.json` | 按 session 分组的 fold、候选参数、分数和最终参数 |
| `models/median_delta.json` | 训练集二维增量中位数及贡献样本数 |
| `models/*.joblib` | Model A/B 最终训练管线；只能与同一次特征 schema 和配置配套使用 |
| `predicted_delta_u/v` | 模型输出的跨线点到落点增量 |
| `raw_pred_u/v` | `crossing_uv + predicted_delta_uv`，允许有限越界 |
| `pred_u/v`、`clipped` | 裁剪到 `[0,1]` 后用于主指标和视频的坐标，以及是否发生裁剪 |

稳定方法标识为 `m2_median_delta`、`m2_model_a_ridge`、`m2_model_b_gradient_boosting`。M2 预测不使用 oracle 时间，`prediction_frame/time_ms` 与三种方法共用的首次跨线帧一致。

## 人工审核数据

审核单位为原始 touch。保留／舍弃的唯一结果文件是 `<session>/review.csv`，与原始视频和采集文件同目录；不更改原始文件，也不创建审核输出目录。CSV 使用 UTF-8，包含全部原始点击：

| 字段 | 定义 |
| --- | --- |
| `trial_id` | 原始试次编号，不重新编号 |
| `touch_index` | 原始试次内触摸编号 |
| `keep` | `1` 保留，`0` 舍弃，空白为未审核 |

session 由所在目录确定，按 `(trial_id, touch_index)` 匹配原始 touch，不按行号匹配。舍弃无需原因或备注。每次选择原子更新这一个文件；跳转或播放不隐式保留。支持修改和撤销决定，重启核对全部 touch 标识后定位第一个未审核项。所有未审核项处理完成后才是完整的人工选择结果。旧版清单迁移只转换已有决定，不改变 touch 标识。

默认查看白闪前 1000 ms 至白闪后 200 ms，按视频边界截断。可扩大到上一白闪之后，首条扩大到视频开始。白闪只是查看锚点，不表示精确触摸时间或动作边界，也不改变预测窗口和 crossing 规则。逐帧查看按已有追踪的时间戳定位。

`config.yaml` 的 `review.before_flash_ms`、`review.after_flash_ms`、`review.display_max_width_px`、`review.trace_tail` 和 `review.slow_playback_multiplier` 分别控制预览区间、页面最大宽度、轨迹长度和慢放倍率。

后续分析读取此文件中的 `keep=1`；未审核项不默认为保留。实验需要固定筛选结果时，由实验记录保存实际使用的文件哈希和试次标识，不要求审核工具生成实验目录或版本导出。

### 屏幕四角标注

review 将全部四角保存到 `<session>/calibration.json`。schema v2 的顶层保留首次标定的字段，并增加 `touch_calibrations` 数组；数组只保存后续单独标注的触摸，每条记录以原始 `(trial_id,touch_index)` 关联。各条标注包含 `image_points_px`、`homography_image_to_screen`、`image_size_px`、`reference_frame_index`、`reference_time_ms`、`session_id`、视频身份／方向和 `annotation_source=review`。像素坐标与旋转后的原视频尺寸一致，不使用浏览器缩放后的显示坐标；四角映射到 `(0,0)`、`(1,0)`、`(1,1)`、`(0,1)`。

按完整原始触摸顺序解析：首次使用顶层标定，后续有单独标注则覆盖当前映射，没有则继续沿用上一触摸的有效映射。继承可连续传递，不能使用后续标注回填前面的触摸，也不插值变换。保存后从当前触摸向后更新，遇到下一个单独标注即停止；“沿用上一次四角”移除本次单独标注。新单独标注仅允许 `keep=1`，但之后修改保留／舍弃不会删除已有四角；完整序列包含舍弃项，已存几何仍可向后沿用。

缺少标定时必须先在第一次触摸片段的一帧上补标。已有 schema v1 自动作为首次标定读取并向后沿用，不因打开页面重写文件；各标注校验 session、图像尺寸和视频身份。补标与重标不修改 `review.csv`、原视频、touch 文件或追踪缓存，也不要求重新审核。保存时原子更新这一个 JSON；实验保存实际使用的完整标定快照和哈希，既有结果不随重标改变。逐触摸补标只处理不同动作之间的位置变化，同一动作内仍是静态映射。

### 自动准备缓存

`<session>/.review_cache.json` 是可再生成的机器缓存，与人工结果 `review.csv` 分开。包含旋转后视频尺寸／方向、FPS、全部帧的实际时间戳和拇指图像坐标（漏检为 null）、亮度序列、白闪候选与配对说明、源视频和 touch 文件 SHA256、MediaPipe 版本、跟踪及准备参数。只生成这一个缓存，不复制视频，不另建审核 outputs 目录。

缓存 schema v2 的 `frames` 保存连续追踪坐标，`raw_frames` 保存原始模型坐标，`tracking_status` 逐帧保存来源。身份另含连续追踪实现的 SHA256，避免代码变化后复用旧处理结果。仅连续追踪参数／实现变化时，核对模型输入身份后复用原始坐标并重新扫描图像，无需重复模型推理；旧 v1 缓存的 `frames` 可作为原始模型来源。历史缓存归档不作为默认输入。

机器缓存按输入和参数校验。尚无人审结果时，模型配置变化会自动重新提取；成功替换前将旧缓存保留为 `.review_cache.<旧内容SHA256>.json`，归档不作为当前追踪源。已有人工结果后跟踪输入或模型配置变化仍会报错，避免旧选择默默对应新轨迹；有意重做时先归档旧的 `review.csv`。仅修改白闪 ROI 会重新扫描亮度并复用已提取坐标。异常中断不会写入不完整的新缓存。页面模型标识来自实际缓存身份，显式传入的 CSV 标为外部来源，不推测其模型。

准备超参数位于 `review.preparation`：`input_height_px` 是模型输入高度；`flash_roi_uv` 为旋转后图像的归一化矩形，null 表示全画面；`flash_rise_thresholds_gray` 是亮度上升阈值候选；`flash_min_interval_ms` 合并相邻闪光上升；`flash_max_residual_ms` 限制视频闪光时间减浏览器 touch 时间、再去除中位偏移后的最大残差。阈值选择属于事后审核定位，不用于预测特征。白闪仍为近似同步，残差不是绝对触摸时间误差。

## 数据有效性

原始记录始终保留。错误方向、最终同侧触点、重复触摸、未匹配白闪、轨迹缺失等先标记；具体排除规则在实验开始前固定并在实验记录中报告数量。模型失败不能静默从分母删除。

采集中的开始/复位动作与目标点击必须能区分；试次边界方案确定后补入 session 协议。不能根据最终预测误差反过来决定某个试次是否有效。
