本目录实现 UR5 + RG2 + 三路 RGB + SpaceMouse 的人工接管式 DAgger。新增程序复用当前 OpenPI 仓库的推理客户端、模型、训练脚本和 LeRobot 数据接口。硬件适配模块的本地副本放在 `hardware/`，来源见该目录的 `PROVENANCE.md`。

一个 episode 内可以多次从模型切到人工，再交回模型。每一帧都保存观测、实际下发动作和控制来源。结束回合时由操作者标注成功、失败或中止。随后将原始专家示教与本轮有效人工纠正段合并，微调模型，再用新模型采集下一轮。

1. 找到真正的仓库根目录

   Windows 上的根目录为 `D:\desp\研究生工作\ur5\openpi_lab-main\openpi_lab-main`，其中包含 `pyproject.toml`、`src` 和 `scripts`。

   后续 Bash 命令在 Ubuntu 22.04 的仓库根目录执行。若使用 WSL，进入目录的命令为

   ```bash
   cd '/mnt/d/desp/研究生工作/ur5/openpi_lab-main/openpi_lab-main'
   ```

   Ubuntu 实机上请将路径换成该电脑上的仓库路径。原 OpenPI 项目的 GPU 环境要求见根目录 README。模型服务器所在电脑需要 NVIDIA GPU。

2. 安装环境

   使用 Python 3.11，将环境放在新增目录内。需要已经安装 `uv`，并具备下载 Python 依赖和模型权重的网络条件。

   ```bash
   export UV_PROJECT_ENVIRONMENT="$PWD/ur5_dagger/.venv"
   GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --python 3.11
   uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" -r ur5_dagger/requirements.txt
   source ur5_dagger/.venv/bin/activate
   ```

   `uv sync --frozen` 使用根目录锁定的 LeRobot 版本。数据格式为 v2.1。实机还需要 RealSense SDK/USB 权限、SpaceMouse HID 权限，以及与当前 UR5 兼容的 External Control URCap。LeRobot 视频编码需要 ffmpeg。

   新开终端后也要进入仓库根目录并执行上述 `source` 命令。

3. 先运行模拟检查

   ```bash
   python ur5_dagger/selftest.py
   python ur5_dagger/collect.py --validate
   python ur5_dagger/collect.py --config ur5_dagger/mock.yaml --auto-demo --no-preview
   ```

   自动模拟运行两条六秒轨迹，每条包含两次人工纠正。保存位置为 `ur5_dagger/output/mock_round1`。再次运行时使用新的目录，例如加上 `--output ur5_dagger/output/mock_round2`。

   `--auto-demo` 会自动生成模型动作、SpaceMouse 输入和成功结果，用来检查程序流程。实机模式由操作者标注结果。去掉 `--no-preview` 可以查看三路画面。

   Windows PowerShell 可单独运行 CPU 回归检查，环境也放在新增目录内。

   ```powershell
   Set-Location 'D:\desp\研究生工作\ur5\openpi_lab-main\openpi_lab-main'
   py -3.11 -m venv ur5_dagger/.venv_mock
   ur5_dagger/.venv_mock/Scripts/python.exe -m pip install 'numpy<2' scipy 'pydantic>=2,<3' PyYAML
   ur5_dagger/.venv_mock/Scripts/python.exe ur5_dagger/selftest.py
   ```

   这条 Windows 路径需要先安装 Python 3.11。模型与实机采集按前面的 Ubuntu 环境运行。

4. 准备实机配置

   ```bash
   cp ur5_dagger/real.example.yaml ur5_dagger/real.yaml
   python ur5_dagger/collect.py --config ur5_dagger/real.yaml --validate
   ```

   `real.example.yaml` 带有采集项目中记录的设备标识，包括机器人 `192.168.100.100`、CB3 125 Hz、External Control 端口 50002，以及三台相机序列号。需要在实际电脑上核对这些标识、TCP、工作空间和 SpaceMouse 方向。核对后在 `real.yaml` 中设置 `robot.tcp_configured`、`robot.workspace_configured` 和 `robot.motion_enabled` 为 `true`。

   新配置的最大平移速度为 0.05 m/s，最大旋转速度为 0.15 rad/s。人工和模型动作都经过相同的速度限制与工作空间裁剪，模型的夹爪预测以 0.5 为阈值转为 0/1，记录的是裁剪和量化后实际提交的命令。

   UR 示教器需要运行已有的 External Control + RG2 worker 程序。RG2 使用 input 24–26、output 24–28。这段示教器程序应沿用之前运行成功的实机程序，它的源文件未包含在提供的采集项目内。不要同时运行其他占用这些输入寄存器或机械臂控制端口的程序。

   RG2 的状态两维由 worker 报告的目标总开度换算为 `[width_mm/2000, -width_mm/2000]`。数据会保存这一来源。连接时不会主动开合夹爪，开始执行后采用 `initial_grasp` 设定的初始命令，0 张开、1 闭合。已有夹持物时应核对这个值。

5. 启动模型服务器

   使用与本目录动作定义和图像映射一致的 UR5 权重。默认模型结构为 pi0.5、32 维内部填充、10 步 action horizon，8 维状态通过 pi0.5 的离散状态 token 输入。训练与推理的 `--horizon`、`--lora`、`--fps` 和 `--repo-id` 应保持一致。其他机器人平台的权重需要经过 UR5 微调。

   ```bash
   python ur5_dagger/run_model.py serve \
     --checkpoint /absolute/path/to/your/ur5_checkpoint \
     --repo-id local/ur5_dagger --horizon 10 --fps 30 --lora
   ```

   `/absolute/path/to/your/ur5_checkpoint` 应指向包含 `params/` 或 `model.safetensors`、以及 `assets/` 的实际 checkpoint 目录。全量训练得到的权重去掉 `--lora`。本目录没有附带训练好的 UR5 权重。

   如果模型在另一台 GPU 电脑上运行，把 `real.yaml` 的 `dagger.policy.host` 改成那台电脑的 IP。端口默认 8000。新服务器会发布动作定义与 FPS，采集端会检查这两项。

   已有其他电脑上的 UR5 权重时，还应核对其相机映射、状态编码、增量坐标系、夹爪编码和网络结构。新的服务器声明只适用于本目录定义的模型配置。

6. 启动连续接管采集

   在第二个终端激活环境，然后执行

   ```bash
   python ur5_dagger/collect.py --config ur5_dagger/real.yaml
   ```

   每条回合开始前按 Enter。按键可以在采集终端或相机窗口使用。默认从人工模式开始。

   | 操作 | 行为 |
   |---|---|
   | H | 人工接管，SpaceMouse 控制机械臂 |
   | P | 交给模型，使用当前观测重新推理 |
   | 推动 SpaceMouse 超过阈值 | 模型模式自动转为人工接管 |
   | SpaceMouse 按键 0 | 切换夹爪，模型模式下同时触发人工接管 |
   | SpaceMouse 按键 1 | 人工速度档位切换，仍受全局最大速度限制 |
   | G | 人工模式切换夹爪 |
   | 空格 | 暂停运动并继续记录，控制来源为 hold |
   | S | 停止本条轨迹，标注成功并保存 |
   | F | 停止本条轨迹，标注失败并保存 |
   | A | 标注中止并保存，进入下一回合 |
   | Q | 标注中止并保存，退出采集 |

   切回模型前将 SpaceMouse 回中，否则自动接管条件会立即再次触发。纠正后仍可交回模型，同一个 episode 会持续记录。人工模式下摇杆回中时，零动作帧仍标为人工。

   推理位于独立进程，等待模型时机械臂保持。控制权切换会清空缓存，并丢弃之前未完成的推理结果。推理连接错误时进入 hold，可以按 H 继续人工操作。服务器恢复后重新启动采集会建立新连接。

   S 代表任务最终成功。只有没有人工接管且最终成功时，额外的 `autonomous_success` 才为 true。达到回合时限或发生异常时标为 aborted。异常停止后的数据尽量保留，并记录原因。

7. 查看数据字段

   | 字段 | 含义 |
   |---|---|
   | `observation.state` | float32[8]，绝对 TCP 的 xyz 与旋转向量，再加两维夹爪状态 |
   | `observation.images.agentview_1` | 第一全局 RGB |
   | `observation.images.agentview_2` | 第二全局 RGB |
   | `observation.images.eye_in_hand` | 腕部 RGB |
   | `action` | float32[7]，基座系 TCP 单周期增量 + 0/1 夹爪指令 |
   | `complementary_info.intervention` | bool[1]，本帧是否由人工控制 |
   | `complementary_info.action_source` | int64[1]，0 保持、1 模型、2 人工 |
   | `complementary_info.target_pose` | 实际提交的绝对 TCP 目标 |
   | `complementary_info.host_time` | 本帧实际主机时间，秒，monotonic |
   | `complementary_info.robot_host_time` | 机械臂状态主机时间 |
   | `complementary_info.gripper_host_time` | 夹爪状态主机时间 |
   | `complementary_info.camera_host_times` | 三路图像到达主机时间 |
   | `complementary_info.camera_frame_numbers` | 三路相机帧号 |

   LeRobot 会生成 `timestamp`、`frame_index`、`episode_index`、`task_index`。其 timestamp 是基于固定 FPS 的时间轴，实际到达时间另存于上述字段。观测在下发当前动作前读取。action 是命令增量，与下一帧实测位姿差可能不同。

   每条轨迹的结果在 `meta/dagger/episode_000000.json` 等文件内，包含 outcome、success、intervention_count、autonomous_success、reason 和 committed。committed 为 false 表示回合保存中断，需检查后再纳入训练。任务文字保持为任务指令，结果单独保存。

8. 合并原始示教与人工纠正段

   ```bash
   python ur5_dagger/prepare_training.py \
     --config ur5_dagger/real.yaml \
     --demo /absolute/path/to/original_expert_dataset \
     --hil ur5_dagger/output/real_round1 \
     --output ur5_dagger/output/training/local/ur5_dagger \
     --repo-id local/ur5_dagger --min-frames 10
   ```

   `--demo` 和 `--hil` 都可重复。它们均应为 LeRobot v2.1，且具有本目录定义的三路图像、8D state、7D 增量 action 和相同 FPS。每轮指定新的输出目录，原始数据始终保留。

   原始专家示教整条纳入。HIL 轨迹只提取连续人工控制段，每段单独成为训练 episode，避免动作序列跨越模型控制区间。成功和失败回合中的有效人工段都会保留。中止回合默认跳过，核对数据后可加 `--include-aborted`。`--min-frames` 应不小于训练的 action horizon，过短的纠正段会计入跳过数量。

   相机映射为 agentview_1 → base_0_rgb、agentview_2 → right_wrist_0_rgb、eye_in_hand → left_wrist_0_rgb。三个槽位均启用。TCP 增量保持原值。训练数据来源与原始帧区间保存在 `meta/dagger_training_sources.json`。

9. 计算统计量并微调

   ```bash
   python ur5_dagger/run_model.py stats \
     --data-home ur5_dagger/output/training \
     --repo-id local/ur5_dagger --horizon 10 --fps 30 --lora

   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python ur5_dagger/run_model.py train \
     --data-home ur5_dagger/output/training \
     --repo-id local/ur5_dagger --horizon 10 --fps 30 --lora \
     --exp-name round1 --batch-size 8 --steps 10000
   ```

   统计量和 checkpoint 都写入 `ur5_dagger/output/`。`--batch-size` 和步数可按 GPU 显存和数据量调整。全量微调去掉 `--lora`。默认从 pi05_base 初始化。下一轮使用 `--base-params /absolute/path/to/previous_checkpoint/params` 从上一轮 JAX 权重继续微调，原始示教和前几轮纠正段继续保留在合并输入中。

   后续使用步骤 5 的 serve 命令加载新 checkpoint，再收集下一轮。checkpoint 的实际步数目录以训练输出为准。`--resume` 用于从同一个实验目录恢复被打断的训练。

10. 文件与验证范围

   新增的 Python 入口为 `collect.py`、`prepare_training.py` 和 `run_model.py`。其余新文件分别处理控制、推理进程、记录、配置和回归检查。原有目录中的 Python 源码和配置文件无需更改。

   本次 13 项检查通过，覆盖配置、动作限幅、控制权切换、迟到推理结果、模拟回合、结果提交、真实 LeRobot v2.1 写入与读回、人工段导出、三路 MP4 编码与 PyAV 解码，以及原有 OpenPI 客户端和服务器的本地通信。检查中的模型输出与硬件均为模拟。LeRobot 检查使用仓库锁定的源代码，Windows 临时测试环境使用 Python 3.13、NumPy 2.2.6、PyAV 17 和 datasets 3.6。正式运行按前面的 Python 3.11 和根目录锁定环境安装，保持 NumPy 1.26.4。

   实际权重推理、GPU 微调和 UR5/RG2/相机实机运行需要在对应设备上验证。运动进程有心跳超时、状态新鲜度和控制器 watchdog 检查，实机运动仍依赖现场急停、正确 TCP 与工作空间配置。
