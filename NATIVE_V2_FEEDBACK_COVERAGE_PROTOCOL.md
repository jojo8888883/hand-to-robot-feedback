# 两个已有官方片段的观测状态恢复诊断

状态：FROZEN_BEFORE_EXECUTION，2026-10-04。沿用 hand_to_robot_continuous_20261003；最多8张现有A800，既有Autopilot，无总GPU时限。不是新论文主张或历史KILL复活。

## 问题和固定输入

勺子三种速度来源均2/3抬升、1/3跟踪，速度来源无稳定增益；实际状态重新规划比原动作更有效。下一步检验这种恢复是否也出现在已有pour_tube、wipe_board片段。两者已有官方控制和毫米偏移下实际分离；接口记录 `diagnostics/V2_FEEDBACK_COVERAGE_INTERFACE.md`。有限差分和观测反馈是已有解法，不称为方法创新。

官方源码44717007de41cbef7565dff7ff9f4453557a2d3d、数据8b0cdf2b24cc86d1c859050c5b0317ddbec59a63。各任务仅复用 `results/native_v2_01/coverage01/{task}/nominal_s0/trajectory_mjwp_fast.npz` 和 `official_loaded_reference.npz`；不重新生成原动作。它们是各自固定seed0控制。新seed0/1/2仅改变本次恢复搜索随机数，并不代表三个示范或三个原策略。

任务、偏移预先固定：pour_tube的(0,+5)mm，即现有九点grid bank5；wipe_board的(-5,0)mm，即bank1。两个位置选自已观察分离的单轴偏移，属于开发案例，不作held-out泛化。九点grid、名义bank4、dt=.01、1024world、margin=-.005不变。每job按旧控制真实执行150步，保存九点live bank。反馈时点统一step150，不随新结果变动；总控制450步，参考真实poststeps1..391、末59步补齐。

## 两个规划臂和实际执行

两臂都使用actual bank的机器人qpos/qvel0:18、相同time/ctrl/mocap和零外力；solver qacc_warmstart共同置零。均用已修复public GPU reset_data→写basic inputs→forward，保留静态world/floor默认变换。不得手工清空所有派生缓存。

`nominal_object`：物体qpos18:25和qvel18:24来自bank4，即无偏移旧动作运行至同一时刻的预测。机器人本体状态仍是真实actual bank。因此这是有机器人本体反馈、无物体观测的重规划比较臂，不是未修改的官方完整pipeline。

`observed_object`：物体qpos来自actual bank；物体速度仅从该bank的step148/150姿态及时间戳估计，沿用固定20ms差分和world SO3 log→当前body旋转。估计器不读未来帧、reference或真实物体qvel。准确模拟器姿态作为观测，尚无图像估计。

因此测试的是物体姿态与因果速度这一整组观测输入的价值，不拆出单独pose/velocity因果结论。两臂其余输入相同；nominal角速度本来就在nominal body坐标，不转到actual body，因为其规划物体姿态也为nominal。不得将任何规划假设写入真实执行状态。

两臂固定顺序nominal_object→observed_object；各自重置相同seed，warmstart=旧controls[150:450]，reference[151:451]，官方optimizer不改奖励、采样、阈值或退火。每次4iterations、最多4attempts；未接受则真实执行旧控制回退，不能执行未接受候选。记录全部候选计算量和接受情况。

实际执行四组 `base/base_duplicate/nominal_object/observed_object`，全部恢复同一原actual bank31字段，再连续执行300步，没有中途reset/revert或再次反馈。四组150步共享前缀，状态保存450×4×1×25/qvel450×4×1×24、controls4×450×18。视频2×2，从已存实际状态渲染。

## 判定与边界

主要结果逐任务报告末50个真实参考步342..391的严格跟踪代理：每步物体根位置误差<0.1m且SO3误差<0.5rad。lift10cm与两示范动作不符，不作为恢复判据。另报poststeps151..391平均位置cm和SO3角误差、分离视频、搜索未接受的次数/计算量。真实倒液体/擦拭效果未建模，不声称完整任务成功。

每任务只有当observed_object至少2/3次搜索接受且末窗跟踪通过，并且其通过数同时超过nominal_object和base各自的通过数、两个差值都大于base/duplicate通过数的绝对差，才标记 `OBSERVED_OBJECT_RECOVERY_SIGNAL`；否则 `NO_FROZEN_TASK_RECOVERY_SIGNAL`。只有两个任务都满足才可说本次两开发片段都观察到恢复信号。连续误差可能揭示阈值掩盖的差别，但不能替代冻结通过条件。无按结果丢种子、换偏移或换时点。

先同时提交两任务seed0，核对新任务的规划状态来源、因果帧、公共机器人状态/重建、实际bank恢复、控制回退和任务指标。各自接口通过后固定提交seed1/2，不能按科学结果选择。普通工程错误最小修复；真实轨迹已保存时只修后处理。主agent检查实际视频/误差后直接做诊断结论，不新增审阅循环。

## 作业与停止点

新runner `native_spider/probe_v2_feedback_coverage.py`，分析 `native_spider/analyze_v2_feedback_coverage.py`。SPEC `native_spider/V2_FEEDBACK_{POUR_TUBE,WIPE_BOARD}_S{0,1,2}_SPEC.json`。六job `h2r-20261004-spider-v2-feedback-{pour-tube,wipe-board}-s{0,1,2}`；输出 `results/native_v2_01/feedback_coverage01/{task}/s{seed}`。保存冻结协议、config来源、物体规划输入、估计器输入、随机数、两臂优化记录、实际轨迹/视频和Autopilot exit。

六组结束后收集、计费、主agent查看、按上述条件结论；本协议不含增多反馈次数、搜窗口/权重/偏移或加入感知噪声。结果决定下一具体问题，再另行冻结。整个连续研究不因一次诊断终止；无费用、真机或外部发布。
