# 运行条件

仓库里有重规划代码、实验配置、[结果记录](results/native_v2_01/feedback_coverage01/summary.json)和两段仿真回放。重跑还需要原实验的场景、控制和参考轨迹。

## 准备环境和输入

- 用 Linux、NVIDIA GPU 和 EGL。原实验环境使用 Python 3.12、PyTorch 2.11.0+cu126、MuJoCo 3.7.0、MuJoCo Warp 3.7.0.1 和 Warp 1.12.1。[完整环境记录](native_spider/installed_packages.txt)
- 在 `$SPIDER_ROOT/repos/spider-native` 放置[原实验使用的 SPIDER 提交](https://github.com/facebookresearch/spider/tree/44717007de41cbef7565dff7ff9f4453557a2d3d)。`SPIDER_ROOT` 指包含 `repos/` 的工作目录。
- 准备数据版本 `8b0cdf2b24cc86d1c859050c5b0317ddbec59a63` 对应的 OakInk-v2 场景、运动参考和网格。[输入文件清单](native_spider/v2_readiness/ADDITIONAL_INPUTS.md)
- 为每个案例准备原实验的 `trajectory_mjwp_fast.npz` 和 `official_loaded_reference.npz`。前者包含该案例的 seed 0 控制，`ctrl` 整理后为 450×18。后者包含加载后的参考，`qpos` 至少 451 行、每行 25 维，仿真步长为 0.01 秒。这两份数组需要从原实验归档取得。

## 用准备好的输入跑一组

以 `pour_tube` 的 seed 0 为例。复制 [原实验配置](native_spider/V2_FEEDBACK_POUR_TUBE_S0_SPEC.json)，把 `author_cli_arguments` 中的 `dataset_dir` 换成该案例在本机的数据根目录，目录下应有 `processed/oakinkv2/`，场景中的网格引用应能找到文件。其余实验参数沿用原配置。从仓库根目录执行，先替换以下路径。

```bash
export SPIDER_ROOT=/absolute/path/to/work-root
spec=/absolute/path/to/copied-pour-tube-s0.json
controls=/absolute/path/to/trajectory_mjwp_fast.npz
reference=/absolute/path/to/official_loaded_reference.npz
output=/absolute/path/to/new-output-directory

python native_spider/probe_v2_feedback_coverage.py \
  --seed 0 --spec "$spec" --controls "$controls" --reference "$reference" \
  --output-dir "$output" --job-id local-pour-tube-s0 --preflight
```

预检在 CPU 上检查配置和数组，并写出 `preflight.json`。输入就绪后，去掉 `--preflight` 即可启动仿真与重规划，结果写到独立的输出目录。换案例或种子时选用对应的 `V2_FEEDBACK_*_SPEC.json`，并让 `--seed` 与配置一致。每个案例的三个恢复搜索种子都复用该案例的一份原控制。

## 原实验记录

[实验协议](NATIVE_V2_FEEDBACK_COVERAGE_PROTOCOL.md)记录了输入、对照和通过阈值。[`jobs/`](jobs/) 保存已完成作业的配置，其中的机器路径和作业 ID 用于追溯。原控制生成脚本沿用当时的服务端目录和辅助文件。[分析脚本](native_spider/analyze_v2_feedback_coverage.py)需要六组完整的执行和优化数组，现有小型结果 JSON 可直接查阅。
