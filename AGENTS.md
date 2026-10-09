# AGENTS.md

规划就只要在文档中写好要怎么实现，不要随便动项目代码！！！

本仓库是一个视觉触摸前预测的研究原型。

当前优先目标是 **快速、可信地验证研究假设**，而不是构建生产级系统。

固定样机采用 Huawei Mate 80 Pro

"screenWidth":1280,"screenHeight":2832

采集条件固定摄像头视角与手机相对摄像头的摆放姿态。

仓库中用到的超参数放到 config.yaml 中

## 开始工作前

在修改代码之前，必须先阅读：

1. `docs/PROJECT_CONTEXT.md`
2. `README.md`
3. `docs/ARCHITECTURE.md`
4. `docs/DATA.md`
5. `docs/DECISIONS.md`
6. `docs/EXPERIMENTS.md`，开展或分析实验时

其中：

* `PROJECT_CONTEXT.md` 负责说明研究背景、当前研究问题、P0 范围和明确非目标；
* `README.md` 负责说明当前项目状态、目录结构、运行方式和当前 milestone。

如果用户最新指令和文档冲突，以用户最新指令为准，并同步更新相关文档。

## 核心原则

> 最快得到可信实验结论，比构建最通用的系统更重要。

优先选择简单、可检查、易调试、可复现、依赖少的实现。

不要为了工程上的完整性主动增加复杂度。

## 不要擅自扩大范围

除非用户明确要求，否则不要主动加入：

* GRU / LSTM / TCN / Transformer；
* SAM；
* ArUco / AprilTag；
* 自动手机检测；
* 3D 手部重建；
* 深度估计；
* 实时推理；
* 智能眼镜部署；
* 复杂 Web 框架；
* 数据库；
* Docker；
* 大型实验管理框架。

## 实验规则

不要静默修改：

* 坐标系定义；
* centerline 定义；
* crossing 方向；
* ground truth；
* prediction time；
* 指标；
* 数据过滤方式。

预测只能使用 prediction time 及之前的信息，避免 future information 和 label leakage。

## 数据与实验

原始数据不可原地修改。

实验应尽量能够追溯到：

```text
experiment
→ data session
→ config
→ git commit
→ outputs
```

不要覆盖已有实验结果。

## 修改代码时

务必使用名为 visual_pretouch 的 conda 虚拟环境，如果没有就先创建。

开始前先检查当前仓库和已有实现。

只做当前任务需要的最小修改。

不要顺手重构无关代码，也不要因为“可以更优雅”就重写已经工作的模块。

## 依赖

`requirements.txt` 只保留当前代码真正需要的依赖。

不要使用 `pip freeze` 直接生成整个环境依赖。

没有必要时不要新增大型框架。

## 完成任务后

说明：

* 修改了什么；
* 修改了哪些文件；
* 怎么运行；
* 怎么验证；
* 当前还有什么限制。

如果当前 milestone 或项目状态发生变化，更新 `README.md`。
