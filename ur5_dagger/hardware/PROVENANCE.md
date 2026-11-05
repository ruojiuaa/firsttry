# Hardware snapshot

These files were copied into this new directory on 2026-11-05 from the user's local UR5 capture project, with permission to read that project.

The source was `ur5_lerobot_7dim-main/ur5_lerobot_7dim-main/src/ur5_capture` under the parent workspace.
The copied modules are `config.py`, `geometry.py`, `robot.py`, `teleop.py`, `cameras.py`, `gripper.py`, `rg2.py`, and `_rtde/*.py`.
The source project's Apache 2.0 license is included as `LICENSE`.
The Universal Robots RTDE client retains its original BSD license headers.

The collector imports these local copies. The capture project's files are not imported at runtime.
Any corrections needed by this example are applied only to these new copies.

The setup uses UR5 CB3, External Control URCap, and a controller-side RG2 worker with input registers 24–26 and output registers 24–28.
The RG2 worker must already be part of the pendant program. Its implementation was not present in the supplied capture project.
The reported gripper width is the worker's accepted target width. The two state coordinates are `[width_mm / 2000, -width_mm / 2000]`.
