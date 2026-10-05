# 灵巧手中途纠偏：阶段保存点

2026-10-05。状态：本轮实现、六组对照、结果分析和视频交付已完成；自动研究暂停交流。

## 已经得到的成果

在 SPIDER 官方优化器基础上，加入一次物体状态反馈：原控制执行至 1.5 秒，读取物体当前姿态，用过去两帧姿态估计速度，重新规划剩余 3 秒，再从相同的实际状态执行。物体初始位置偏移 5 mm 时，这个改动在两个固定开发案例上改善了后续跟踪。

| 固定片段 | 继续原控制 | 原控制副本 | 按预计物体状态重规划 | 按实际物体状态重规划 |
|---|---:|---:|---:|---:|
| pour_tube | 0/3 | 0/3 | 2/3 | 3/3 |
| wipe_board | 0/3 | 0/3 | 0/3 | 3/3 |

这里每个片段复用一条原控制，三次只改变恢复搜索种子；通过指最后 50 个真实参考步均满足物体位置和朝向误差界限。六组作业均已完成，12 次重规划均在首个四迭代搜索中接受，原始执行数组、误差和视频都已保存。实现与检查已结束，不需要为了“做完这一轮”再训练或补跑。

用户查看 wipe_board seed0 四格视频后观察到：右下保持抓住物体，另三格后来掉落。这记录为对该段画面的直接反馈，不替代其他试验的指标。

## 流程搭到哪里

目标流程：人手视频 → 手和物体三维运动 → 机械手动作参考 → 仿真控制优化 → 执行 → 视觉估计当前物体状态 → 调整后续动作。

当前运行从官方预处理三维示范开始，已经贯通控制优化、连续物理执行、一次状态反馈重规划和实际执行结果回放。自己的视频解析前端尚未接入；反馈姿态来自仿真器，尚未接 FoundationPose 或其他图像估计器；当前是悬浮 XHand 仿真，没有机械臂或真机。

SPIDER 提供官方优化算法和配置。本轮代码加入执行适配、固定时刻的物体状态反馈、同状态对照及结果分析。观察后重规划属于已有控制思路；本保存点是一个有效的实现与实验成果，未主张新的算法贡献。

## 代码、结果与观看入口

主运行入口：[probe_v2_feedback_coverage.py](native_spider/probe_v2_feedback_coverage.py)。分析入口：[analyze_v2_feedback_coverage.py](native_spider/analyze_v2_feedback_coverage.py)。它们调用的本地 Python 模块随同保存，未为整理而改变已执行算法。

固定条件在 `native_spider/V2_FEEDBACK_{POUR_TUBE,WIPE_BOARD}_S{0,1,2}_SPEC.json`。六份 Autopilot job 配置在 `jobs/h2r-20261004-spider-v2-feedback-*.json`；均为已完成作业记录，不能再次提交。冻结规则见 [NATIVE_V2_FEEDBACK_COVERAGE_PROTOCOL.md](NATIVE_V2_FEEDBACK_COVERAGE_PROTOCOL.md)，完整说明见 [V2_FEEDBACK_COVERAGE_RESULT_REPORT.md](V2_FEEDBACK_COVERAGE_RESULT_REPORT.md)，数值见 [summary.json](results/native_v2_01/feedback_coverage01/summary.json)。

[wipe_board seed0 四格视频](results/native_v2_01/feedback_coverage01/wipe_board/s0/feedback_coverage_4grid.mp4) · [pour_tube seed0 四格视频](results/native_v2_01/feedback_coverage01/pour_tube/s0/feedback_coverage_4grid.mp4)

四格顺序：左上原控制，右上原控制副本，左下预计物体状态重规划，右下实际物体状态重规划。视频由真实执行状态渲染，约 4.5 秒；3.91 秒之后为参考补齐段，不计入主要指标。

## 保存范围与恢复条件

本地 Git 提交保存源代码、配置、协议、报告和小型 JSON 结果。按照项目现有 `.gitignore`，模型、网格、NPZ 数组和视频保留在现有数据目录，不强制加入 Git；因此代码提交不是包含全部资产的独立发行包。

本地工作根为 `/Users/jojo8888883/Desktop/26summer/autoresearch/research-runs/hand-to-robot-20261003-continuation`。A800 工作根为 `/mnt/sdc/dty_user/hand_to_robot_20261002`。反馈输出位于两端各自的 `results/native_v2_01/feedback_coverage01/{task}/s{seed}`，含 `trajectories.npz`、`prefix.npz`、`boundary_state.npz`、两臂优化/重建记录和视频。既有控制与参考位于 `results/native_v2_01/coverage01/{task}/nominal_s0/{trajectory_mjwp_fast.npz,official_loaded_reference.npz}`。

运行依赖现有 A800 环境 `env/spider-native-py312` 与源码 `repos/spider-native`。官方源码固定 `44717007de41cbef7565dff7ff9f4453557a2d3d`，数据固定 `8b0cdf2b24cc86d1c859050c5b0317ddbec59a63`。依赖版本记录在 `native_spider/installed_packages.txt`、`locked-requirements.txt`；资产入口和配置语义见 `native_spider/v2_readiness/ADDITIONAL_INPUTS.md` 与 `diagnostics/V2_FEEDBACK_COVERAGE_INTERFACE.md`。保留已有 MuJoCo 兼容设置和静态检查绕过记录，不将本适配运行称为原封不动的官方默认入口。

只复查已有结果时，可在有 NumPy、Matplotlib、Pillow、imageio 的本地 Python 环境运行：

```bash
python research-runs/hand-to-robot-20261003-continuation/native_spider/analyze_v2_feedback_coverage.py
```

这读取已有数组并生成汇总/图，不启动新优化。原始控制生成入口 `run_v2_coverage_condition.py`、`run_native_v2_robust.py` 一并保存，供理解来源；当前不重生成已完成控制。

## 当前边界与下一步

这轮小实验已经完成，可以作为阶段成果保留。两条固定开发示范及三次恢复搜索尚不能代表广泛成功率；仿真在重规划时暂停，准确姿态来自仿真器，完整倒液体/擦拭效果未建模。相同原控制副本存在数值轨迹差异，已在原报告记录。

下一步可在此保存点上接视觉估计，检验图像误差下恢复是否仍成立。该阶段尚未执行，先完成官方视觉跟踪输入准备并冻结新对照。当前先交流，heartbeat24 保持暂停；既有八卡资源授权不变。
