# 两个额外V2片段的同状态反馈接口

2026-10-04；gpt-6.1-sol / high。只读元数据、数组和CPU模型/loader；未仿真、未使用GPU、未改runner。

**两任务在数据和物理接口上均可复用现有同状态 replan 核心，无模型维度或输入闭包阻塞。** 当前 runner 通过传入coverage SPEC即可解析对应task/scene/raw数据，已有完整450控制与842行参考；但不能把勺子的持续抬升判据、固定偏移/反馈时间和三速度比较标签自动转成跨任务实验。

## 确切输入路径

远端以下六个文件已确认存在。控制是已完成官方优化的保存输出，作为回放/replan warmstart；重新加载reference仍来自各任务原始 `trajectory_kinematic.npz`，并非拿优化输出冒充示范输入。

| 任务 | 用途 | 本地完整路径 | 本地状态 | 远端完整路径 | 远端状态 |
|---|---|---|---|---|---|
| pour_tube | controls | `/Users/jojo8888883/Desktop/26summer/autoresearch/research-runs/hand-to-robot-20261003-continuation/results/native_v2_01/coverage01/pour_tube/nominal_s0/trajectory_mjwp_fast.npz` | 存在 | `/mnt/sdc/dty_user/hand_to_robot_20261002/results/native_v2_01/coverage01/pour_tube/nominal_s0/trajectory_mjwp_fast.npz` | 存在 |
| pour_tube | reference | `/Users/jojo8888883/Desktop/26summer/autoresearch/research-runs/hand-to-robot-20261003-continuation/results/native_v2_01/coverage01/pour_tube/nominal_s0/official_loaded_reference.npz` | 存在 | `/mnt/sdc/dty_user/hand_to_robot_20261002/results/native_v2_01/coverage01/pour_tube/nominal_s0/official_loaded_reference.npz` | 存在 |
| pour_tube | spec | `/Users/jojo8888883/Desktop/26summer/autoresearch/research-runs/hand-to-robot-20261003-continuation/native_spider/V2_COVERAGE_POUR_TUBE_SPEC.json` | 存在 | `/mnt/sdc/dty_user/hand_to_robot_20261002/native_spider/V2_COVERAGE_POUR_TUBE_SPEC.json` | 存在 |
| wipe_board | controls | `/Users/jojo8888883/Desktop/26summer/autoresearch/research-runs/hand-to-robot-20261003-continuation/results/native_v2_01/coverage01/wipe_board/nominal_s0/trajectory_mjwp_fast.npz` | 存在 | `/mnt/sdc/dty_user/hand_to_robot_20261002/results/native_v2_01/coverage01/wipe_board/nominal_s0/trajectory_mjwp_fast.npz` | 存在 |
| wipe_board | reference | `/Users/jojo8888883/Desktop/26summer/autoresearch/research-runs/hand-to-robot-20261003-continuation/results/native_v2_01/coverage01/wipe_board/nominal_s0/official_loaded_reference.npz` | 存在 | `/mnt/sdc/dty_user/hand_to_robot_20261002/results/native_v2_01/coverage01/wipe_board/nominal_s0/official_loaded_reference.npz` | 存在 |
| wipe_board | spec | `/Users/jojo8888883/Desktop/26summer/autoresearch/research-runs/hand-to-robot-20261003-continuation/native_spider/V2_COVERAGE_WIPE_BOARD_SPEC.json` | 存在 | `/mnt/sdc/dty_user/hand_to_robot_20261002/native_spider/V2_COVERAGE_WIPE_BOARD_SPEC.json` | 存在 |

作者原控制副本分别仍在：

* `/mnt/sdc/dty_user/hand_to_robot_20261002/native_spider/v2_coverage01/pour_tube/processed/oakinkv2/xhand/right/pour_tube/0/trajectory_mjwp_fast.npz`
* `/mnt/sdc/dty_user/hand_to_robot_20261002/native_spider/v2_coverage01/wipe_board/processed/oakinkv2/xhand/right/wipe_board/0/trajectory_mjwp_fast.npz`

实际CPU loader输入相应同一目录的 `trajectory_kinematic.npz`；scene位于其父task目录 `scene.xml`，所有引用资源均能由 `MjModel.from_xml_path` 读取。当前本地scene/task_info/config元数据在 `native_spider/v2_dataset/processed/oakinkv2/xhand/right/{pour_tube,wipe_board}/`，实际执行使用SPEC指定的远端隔离副本 `native_spider/v2_coverage01/{task}/`。来源固定官方SPIDER `44717007de41cbef7565dff7ff9f4453557a2d3d`、retarget input `8b0cdf2b24cc86d1c859050c5b0317ddbec59a63`。

## CPU加载和地址

CUDA隐藏后，只调用MuJoCo模型加载及官方 `spider.io.load_data`（device=cpu），没有 `setup_env`、forward或step：

| 项目 | pour_tube | wipe_board |
|---|---|---|
| nq / nv / nu | 25 / 24 / 18 | 25 / 24 / 18 |
| neq / na / nsensor / nmocap | 0 / 0 / 0 / 10 | 0 / 0 / 0 / 10 |
| 显式碰撞pair数 | 132 | 132 |
| freejoint | right_object_joint，body=right_object | 相同 |
| 物体 qpos地址 | 18:21世界xyz，21:25 wxyz | 相同 |
| 物体 qvel地址 | 18:21世界线速度，21:24 body-local角速度 | 相同 |
| 机器人 qpos/qvel | 0:18；前三slide、后三串联hinge、12手指关节 | 相同XHand布局 |
| 保存控制 | 3×150×18，展平450×18，有限 | 相同 |
| 原始示范 | 196×25 qpos、196×24 qvel，50Hz | 相同 |
| CPU loader | qpos842×25、qvel842×24、ctrl842×18、contact842×10、contact_pos842×10×3，全部有限 | 相同 |
| 有效参考 | 392行含row0；实际poststep1..391；尾部填充450行 | 相同 |
| floor | geom0，world body0，plane | 相同 |

两CPU加载的qpos均与各自保存 `official_loaded_reference.npz` 在float32精度上逐元素相等。原raw只有qpos/qvel/frequency，官方loader按 `source/spider/io.py:53–65` 从机器人qpos生成reference control，并按 `77–112` 插值和补尾；缺contact数据时填零，与contact reward=0一致。这是已用官方路径，不是当前接口缺文件。

模型地址依据各 `scene.xml:165–166` 的freejoint及实际CPU `jnt_qposadr=18/jnt_dofadr=18`；参考行数由各 `reference_info.json` 与本次数组加载共同确认。metadata写200帧/pre_grasp1.5秒不替代实际196行输入的事实。

## 配置差异与复用边界

逐项比较两个 `results/native_v2_01/coverage01/{task}/nominal_s0/config.yaml` 和勺子 `robust01/nominal_s0/config.yaml`：差异只有task、dataset_dir、model_path、data_path、output_dir以及npair（158→132）。物体/对象网格不同，pour_tube为O02@0011@00003/hold，wipe_board为O02@0055@00002/brush_whiteboard。两者的 `task_info.json:15–16` 有此记录。

时序和优化参数仍相同：dt=.01、ctrl_steps150、horizon_steps300、knot_steps100、1024候选、4iterations、DR1/xy0/margin-.005、contact_guidance=false、terminate_resample=false、零外力、object action dims0。搜索接受阈值position .1m / rotation1.0rad；奖励与噪声配置相同。现有CPU INVDISCRETE兼容与静态检查bypass也相同。碰撞pair数由config动态读取，不需要写死158。

可以直接复用的部分：`probe_v2_causal_velocity.py:239–255` 根据SPEC加载任务、模型和参考，31字段live bank/共同重建/静态world保留/官方optimize/失败回退，以及`496–524`从同一actual bank实际执行。维度断言25/24/18、地址18:25、floor geom0、zero equality和150/300时序，在这两个实际模型上都成立。

**不能自动沿用的硬编码/科学语义：**

* `probe_v2_causal_velocity.py:18–21,315–331` 固定三个速度臂、step150反馈、147/149估计窗口、bank4名义/bank8(+5,+5mm)实际。覆盖SPEC不决定这些是下一阶段选定的条件；本说明不选择它们。
* `123–134` 及 `probe_native_v2_position.py:61–108` 固定391真实步和末50步，并报告持续抬升10cm。本次两数据的时间长度恰好匹配，但 **lift判据不匹配任务参考动作**：已有报告 `V2_COVERAGE_RESULT_REPORT.md:14` 记录末窗参考最低抬升分别仅3.48cm/7.51cm。应由主agent冻结合适的恢复判定；不能将默认lift false作为pour/wipe恢复失败。
* 现有coverage只有seed0控制/SPEC；`probe_v2_pool_selection.py:31–34` 会核对SPEC seed。直接拿seed0 SPEC运行seed1/2会被拒绝。是否复用同一保存控制做不同搜索seed，或另生成控制，是主agent的下一协议选择，本轮不增加输入或运行。
* 新结果仍需独立protocol标签和output/job路径；当前condition硬编码 `NATIVE_V2_CAUSAL_VELOCITY_PROTOCOL.md`（`134`），它不能把未授权跨对象实验标为旧协议结果。

当前真实阻塞是尚未冻结跨对象恢复的条件与评价语义，不是loader、模型或控制缺失。CPU可用性不证明GPU新条件已执行或已能恢复；现有coverage baseline本身完整，后续接口不必重写优化器或建立通用adapter。
