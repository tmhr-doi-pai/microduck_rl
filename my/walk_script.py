"""台本（速度指令のリスト）どおりにMicroduckを歩かせ、動画に保存する"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")   # 画面なしで描画
sys.path.insert(0, "scripts")               # 公式の infer_policy.py を借りる

import mujoco
import imageio.v2 as imageio
import infer_policy as ip

ONNX = "duck_walk.onnx"
OUT = "walk_script.mp4"

# 台本: (前後速度 m/s, 左右速度 m/s, 旋回速度 rad/s, 秒)
# 範囲の目安: 前後 ±0.3 / 左右 ±0.2 / 旋回 ±1.5
SCRIPT = [
    (0.0, 0.0, 0.0, 2.0),   # その場で立つ
    (0.2, 0.0, 0.0, 4.0),   # 前に歩く
    (0.0, 0.0, 1.0, 3.0),   # 左に回る
    (0.2, 0.0, 0.0, 3.0),   # また前に歩く
    (0.0, 0.0, 0.0, 2.0),   # 止まる
]

# --- シミュレーションと歩行AIの準備（infer_policy.py と同じ設定） ---
bam = ip.load_bam_model(ip.BAM_KP_FW, 7.4, None)
model, data, bam_ctrl, _ = ip.load_mujoco_with_bam(
    ip.MICRODUCK_XML, bam, 0.005, 0.1, ip.BAM_VIN_MIN)
policy = ip.PolicyInference(
    model, data, walking_onnx_path=ONNX, bam_ctrl=bam_ctrl,
    use_projected_gravity=True, new_cmd_obs=True)

# --- 初期姿勢 ---
fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
adr = model.jnt_qposadr[fj]
data.qpos[adr:adr + 3] = [0.0, 0.0, 0.125]
data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
for i, q in enumerate(policy.joint_qpos_indices):
    data.qpos[q] = policy.default_pose[i]
bam_ctrl.reset(data.qpos)
policy.set_position_targets(policy.default_pose)
mujoco.mj_forward(model, data)
print("観測の次元:", policy.get_observations().size, "(61なら正常)")

# --- カメラ（胴体を追いかける） ---
renderer = mujoco.Renderer(model, height=480, width=640)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = policy.trunk_base_id
cam.distance, cam.elevation, cam.azimuth = 0.8, -20, 135

# --- 台本どおりに実行（50Hz で歩行AIを呼ぶ） ---
frames = []
for vx, vy, wz, sec in SCRIPT:
    policy.set_vel_cmd(vx, vy, wz)
    for _ in range(int(sec * 50)):
        action = policy.infer()
        policy.apply_action(action)
        for _ in range(4):          # 物理は 200Hz
            bam_ctrl.update()
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    x, y = data.qpos[adr], data.qpos[adr + 1]
    print(f"  位置: x={x:+.2f} m, y={y:+.2f} m / 胴体の高さ: {data.qpos[adr + 2] * 1000:.0f} mm")
imageio.mimsave(OUT, frames, fps=50)
print("保存しました:", OUT)
