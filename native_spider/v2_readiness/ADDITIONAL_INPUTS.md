# 官方 OakInk-v2 额外输入候选

固定官方 `retarget/retarget_example` revision `8b0cdf2b24cc86d1c859050c5b0317ddbec59a63` 中，找到两个额外 right/XHand trial0 候选：**pour_tube** 与 **wipe_board**。二者的scene、kinematic输入及所有scene引用的网格都在该revision中，物体ID均不同于现有pick_spoon_bowl。它们满足元数据层面的额外输入准备条件；作者task_info均标注1.5s预抓取，尚未确认发布NPZ内实际保留的预抓取过程。

检查日期2026-10-04；执行者gpt-6.1-sol/high。只查询官方目录并读取两份task_info、scene XML、saved config文本和本地既有asset manifest；没有下载轨迹/网格/视频、仿真、GPU、子agent，也未修改protocol/state。没有把任何优化输出当输入。这里只准备独立样例，不代表方法选择或实验阶段启动。

## 两个候选

| 项目 | pour_tube/0 | wipe_board/0 |
|---|---|---|
| dataset / robot / embodiment | oakinkv2 / xhand / right | oakinkv2 / xhand / right |
| right对象ID | O02_0011_00003 | O02_0055_00002 |
| 作者primitive | hold | brush_whiteboard |
| 原始sequence | scene_03__A004/seq__f2d09be48f383f8caf97__2023-04-22-18-52-52 | scene_02__A007/seq__eb5f919d73a9ac759146__2023-04-20-14-20-06 |
| 作者预抓取/抓取后窗口 | 1.5s / 2.5s | 1.5s / 2.5s |
| 作者n_frames / ref_dt | 200 / .02s | 200 / .02s |
| 作者grasp_range | [1191,1931] | [867,2593] |
| 作者oakink_window | [1011,1491] | [687,1167] |
| kinematic NPZ目录大小 | 77,592 bytes | 77,592 bytes |
| scene XML大小 | 41,042 bytes | 41,064 bytes |
| 对象资产闭包 | visual.obj + convex/0.obj至8.obj | visual.obj + convex/0.obj至8.obj |
| 新对象资产总大小 | 489,197 bytes | 359,442 bytes |
| saved config维度 | nq25 / nv24 / nu18 | nq25 / nv24 / nu18 |
| 输入闭包判定 | 发布文件与全部scene网格引用可用 | 发布文件与全部scene网格引用可用 |

现有对象为O02_0030_00002、primitive=scoop、来自scene_01__A001，碰撞网格0.obj至10.obj。两个候选来自不同sequence、对象与操作角色，能避免只在同一勺子片段上解释结果。pour_tube的任务名与hold primitive、wipe_board的brush_whiteboard primitive提供用途差异，但仅这些元数据不能证明具体抓取形态；本次未读取OBJ顶点或看输入视频，不把“试管圆柱抓取/板擦宽面抓取”写成已核实事实。两个不同对象资产可作为后续比较抓取几何的候选。

官方元数据来源：[pour_tube task_info](https://huggingface.co/datasets/retarget/retarget_example/blob/8b0cdf2b24cc86d1c859050c5b0317ddbec59a63/processed/oakinkv2/xhand/right/pour_tube/task_info.json)、[wipe_board task_info](https://huggingface.co/datasets/retarget/retarget_example/blob/8b0cdf2b24cc86d1c859050c5b0317ddbec59a63/processed/oakinkv2/xhand/right/wipe_board/task_info.json)。同一revision的right目录只有pick_spoon_bowl、pour_tube、stir_beaker、wipe_board；找到这两个后未扩展源库搜索，也未读取第三个新候选的内容。

## 精确输入路径与闭包

以下均为该固定revision内的仓库相对路径，保存目录结构即可；author config中的/home/ubuntu/spider-test路径是作者机器绝对路径，不是可直接用于本机的输入地址。

**pour_tube/0**

- scene：`processed/oakinkv2/xhand/right/pour_tube/scene.xml`
- 原始kinematic输入：`processed/oakinkv2/xhand/right/pour_tube/0/trajectory_kinematic.npz`
- 元数据：`processed/oakinkv2/xhand/right/pour_tube/task_info.json`
- saved config参考：`processed/oakinkv2/xhand/right/pour_tube/0/config.yaml`
- 对象视觉网格：`processed/oakinkv2/assets/objects/O02_0011_00003/visual.obj`
- 对象碰撞网格：`processed/oakinkv2/assets/objects/O02_0011_00003/convex/{0,1,2,3,4,5,6,7,8}.obj`，花括号逐项展开为九个确实存在的文件。

官方输入链接：[scene](https://huggingface.co/datasets/retarget/retarget_example/blob/8b0cdf2b24cc86d1c859050c5b0317ddbec59a63/processed/oakinkv2/xhand/right/pour_tube/scene.xml)、[trial0输入目录](https://huggingface.co/datasets/retarget/retarget_example/tree/8b0cdf2b24cc86d1c859050c5b0317ddbec59a63/processed/oakinkv2/xhand/right/pour_tube/0)、[对象资产](https://huggingface.co/datasets/retarget/retarget_example/tree/8b0cdf2b24cc86d1c859050c5b0317ddbec59a63/processed/oakinkv2/assets/objects/O02_0011_00003)。

**wipe_board/0**

- scene：`processed/oakinkv2/xhand/right/wipe_board/scene.xml`
- 原始kinematic输入：`processed/oakinkv2/xhand/right/wipe_board/0/trajectory_kinematic.npz`
- 元数据：`processed/oakinkv2/xhand/right/wipe_board/task_info.json`
- saved config参考：`processed/oakinkv2/xhand/right/wipe_board/0/config.yaml`
- 对象视觉网格：`processed/oakinkv2/assets/objects/O02_0055_00002/visual.obj`
- 对象碰撞网格：`processed/oakinkv2/assets/objects/O02_0055_00002/convex/{0,1,2,3,4,5,6,7,8}.obj`，花括号逐项展开为九个确实存在的文件。

官方输入链接：[scene](https://huggingface.co/datasets/retarget/retarget_example/blob/8b0cdf2b24cc86d1c859050c5b0317ddbec59a63/processed/oakinkv2/xhand/right/wipe_board/scene.xml)、[trial0输入目录](https://huggingface.co/datasets/retarget/retarget_example/tree/8b0cdf2b24cc86d1c859050c5b0317ddbec59a63/processed/oakinkv2/xhand/right/wipe_board/0)、[对象资产](https://huggingface.co/datasets/retarget/retarget_example/tree/8b0cdf2b24cc86d1c859050c5b0317ddbec59a63/processed/oakinkv2/assets/objects/O02_0055_00002)。

**共同机器人依赖**

两个scene的compiler meshdir均为`../../../assets/`，mesh列表引用13个right XHand STL，与已有勺子输入闭包一致。`processed/oakinkv2/assets/robots`是26-byte发布链接文本，目标`../../oakink/assets/robots`；本地复用已建立的相同目录映射，不能将链接文本当STL或XML。

实际共同文件根目录为`processed/oakink/assets/robots/xhand/assets/`，精确文件名：

- `right_hand_link.STL`
- `right_hand_thumb_bend_link.STL`
- `right_hand_thumb_rota_link1.STL`
- `right_hand_thumb_rota_link2.STL`
- `right_hand_index_bend_link.STL`
- `right_hand_index_rota_link1.STL`
- `right_hand_index_rota_link2.STL`
- `right_hand_mid_link1.STL`
- `right_hand_mid_link2.STL`
- `right_hand_ring_link1.STL`
- `right_hand_ring_link2.STL`
- `right_hand_pinky_link1.STL`
- `right_hand_pinky_link2.STL`

13个STL合计6,939,992 bytes；都已在既有输入资产中。两个scene均无外部include，纹理均为builtin，不需要额外纹理图片。scene明确引用的对象visual及九个convex文件均在固定revision目录中，没有发现发布路径缺件。这里的“闭包可用”是文件存在与引用闭合，尚未做MuJoCo实际加载、NPZ解析或动态检查。

## 下载体积估计与剩余输入语义

| 范围 | pour_tube | wipe_board |
|---|---:|---:|
| 新增最低输入：scene + kinematic NPZ +对象网格 | 607,831 bytes，约0.580 MiB | 478,098 bytes，约0.456 MiB |
| 新增输入加task_info与saved config | 611,755 bytes | 482,038 bytes |
| 完全冷启动最低闭包，含13个共同STL与链接文本 | 7,547,849 bytes，约7.198 MiB | 7,418,116 bytes，约7.074 MiB |
| 冷启动闭包加task_info与saved config | 7,551,773 bytes | 7,422,056 bytes |

估计使用官方tree列出的文件大小求和，不含HTTP协议开销。若准备两个输入并复用当前机器人资产，新增最低下载合计1,085,929 bytes，约1.04 MiB；加两组元数据合计1,093,793 bytes。无需下载raw大注释、scene_eq、trajectory_ikrollout、trajectory_mjwp_fast或视频来建立这两个发布kinematic输入闭包。

作者task_info的1.5s预抓取只是元数据证据。既有勺子输入曾出现task_info与发布kinematic首帧/帧数差异，不能把其他任务同样的200帧声明直接升级为实际NPZ已保留完整1.5s接近过程。未来若主agent选择其一，解析真实kinematic帧数和首段手-物体相对运动即可决定是否满足该阶段输入要求；本任务不下载或解析这些数组，也不以未知之处扩大搜索。saved config都指向task目录scene.xml与trial0的trajectory_kinematic.npz，显示相同nq/nv/nu维度，但实际加载与启动可运行性未由本次元数据检查证明。

结论：有两个官方同revision、不同对象与sequence、预抓取元数据匹配且发布文件闭包完整的候选。若要求“实际kinematic中的预抓取过程已经确认”，本次证据仍不足；无需因此继续扩大源库搜索。这里没有选择或启动任何新增方法/实验阶段。
