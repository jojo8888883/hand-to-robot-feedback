# Hand-to-Robot Feedback

在 SPIDER 上实现的一次物体状态反馈重规划：机械手执行途中，根据物体实际状态调整剩余动作，以改善初始位置偏移后的抓取保持和轨迹跟踪。

本仓库保存 2026-10-04 完成的阶段结果，来源为研究工作区提交 `a3bd8dc`。六组实验已完成；这是仿真中途纠偏的实现与对照，尚未接入视觉状态估计。

## 看效果

[播放 wipe_board 四格对照](results/native_v2_01/feedback_coverage01/wipe_board/s0/feedback_coverage_4grid.mp4) · [播放 pour_tube 四格对照](results/native_v2_01/feedback_coverage01/pour_tube/s0/feedback_coverage_4grid.mp4)

每段约 4.5 秒。左上继续原控制，右上重复原控制，左下依据预计物体状态重新规划，右下依据实际物体状态重新规划。视频从真实仿真执行状态渲染。

| 固定开发片段 | 原控制 | 原控制副本 | 预计物体状态重规划 | 实际物体状态重规划 |
|---|---:|---:|---:|---:|
| pour_tube | 0/3 | 0/3 | 2/3 | 3/3 |
| wipe_board | 0/3 | 0/3 | 0/3 | 3/3 |

通过指标为末 50 个真实参考步的位置和朝向跟踪。每个片段复用一条原控制，三次是恢复搜索种子。完整定义与原始数值见[冻结协议](NATIVE_V2_FEEDBACK_COVERAGE_PROTOCOL.md)、[结果报告](V2_FEEDBACK_COVERAGE_RESULT_REPORT.md)和[JSON 汇总](results/native_v2_01/feedback_coverage01/summary.json)。

## 实现

运行从官方预处理三维示范开始，已经贯通控制优化、连续物理执行、固定 1.5 秒处的一次反馈重规划和结果回放。物体姿态目前取自仿真器，速度由过去两帧姿态估计。重规划期间仿真暂停；尚未接入人手视频解析前端、图像姿态跟踪、机械臂或真机。

主要入口是 `native_spider/probe_v2_feedback_coverage.py`，分析入口是 `native_spider/analyze_v2_feedback_coverage.py`。SPIDER 提供官方优化算法与配置，本项目增加执行适配、状态反馈和匹配对照；不将观察后重规划本身宣称为新算法。

## 文件与运行条件

`native_spider/` 保存入口、全部本地导入模块、六份实验配置及依赖版本。`jobs/` 保存六个已完成 Autopilot 作业的命令和资源配置，不应重复提交原 job ID。`results/` 保存小型结果记录及两段代表性视频。

代码沿用原运行环境路径。运行需要官方 SPIDER 源码、所列 Python/CUDA 环境、场景/网格和保存的控制、参考、执行数组。这些大体积运行资产保留在原工作目录及 A800，未包含在本源码备份中；单独克隆仓库不能立即重跑或重算所有指标。具体位置、固定版本与恢复说明见[阶段小结](FEEDBACK_CHECKPOINT.md)。

两条固定开发案例中的改善已观测到；更多场景的稳定效果及带视觉误差的恢复属于后续阶段。
