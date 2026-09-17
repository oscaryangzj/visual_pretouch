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

## 数据有效性

原始记录始终保留。错误方向、最终同侧触点、重复触摸、未匹配白闪、轨迹缺失等先标记；具体排除规则在实验开始前固定并在实验记录中报告数量。模型失败不能静默从分母删除。

采集中的开始/复位动作与目标点击必须能区分；试次边界方案确定后补入 session 协议。不能根据最终预测误差反过来决定某个试次是否有效。
