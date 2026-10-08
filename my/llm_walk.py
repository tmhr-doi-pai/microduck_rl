"""日本語の指示 → LLM → 台本 → Microduckが歩く → 動画"""
import os, sys, json, re, math
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "scripts")

import requests
import mujoco
import imageio.v2 as imageio
import infer_policy as ip

ONNX = "duck_walk.onnx"
OUT = "llm_walk.mp4"
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# --- 実測にもとづく換算（walk_script.py の結果より）---
WALK_CMD = 0.2      # 歩くときに出す指令 [m/s]
WALK_REAL = 0.08    # そのときの実際の速さ [m/s]
TURN_CMD = 1.0      # 回るときに出す指令 [rad/s]
TURN_REAL = 1.0     # そのときの実際の回転の速さ [rad/s]

SYSTEM_PROMPT = """あなたは小型二足歩行ロボットMicroduckの動作プランナーです。
ユーザーの日本語の指示を、次の動作の列に変換してください。
使える動作:
- {"action": "forward", "distance_m": 数値}   前に進む（0.05〜1.0）
- {"action": "backward", "distance_m": 数値}  後ろに下がる（0.05〜0.5）
- {"action": "turn_left", "angle_deg": 数値}  左（反時計回り）に回る（10〜360）
- {"action": "turn_right", "angle_deg": 数値} 右（時計回り）に回る（10〜360）
- {"action": "wait", "seconds": 数値}         その場で待つ（0.5〜5）
距離や角度の指定がない場合は、前進は0.3m、回転は90度にしてください。
出力は次の形式のJSONだけにしてください。説明文は不要です。
{"steps": [ ... ], "comment": "ロボットとしての一言（20文字以内）"}"""


def ask_llm(instruction):
    """LLMに指示を渡し、動作の計画(JSON)を受け取る"""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY が設定されていません")
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": MODEL,
              "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "content": instruction}],
              "temperature": 0},
        timeout=60)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", text, re.S)     # JSON部分だけ取り出す
    if not m:
        sys.exit("LLMの返答からJSONを読み取れませんでした:\n" + text)
    return json.loads(m.group(0))


def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


def to_script(plan):
    """LLMの計画を、(前後, 左右, 旋回, 秒) の台本に換算する（範囲チェック付き）"""
    script = [(0.0, 0.0, 0.0, 1.0)]              # 最初に1秒立つ
    for s in plan.get("steps", [])[:10]:         # 最大10動作まで
        a = s.get("action")
        if a == "forward":
            d = clamp(s.get("distance_m", 0.3), 0.05, 1.0)
            script.append((WALK_CMD, 0.0, 0.0, d / WALK_REAL))
        elif a == "backward":
            d = clamp(s.get("distance_m", 0.3), 0.05, 0.5)
            script.append((-WALK_CMD, 0.0, 0.0, d / WALK_REAL))
        elif a in ("turn_left", "turn_right"):
            ang = clamp(s.get("angle_deg", 90), 10, 360)
            sign = 1.0 if a == "turn_left" else -1.0
            script.append((0.0, 0.0, sign * TURN_CMD, math.radians(ang) / TURN_REAL))
        elif a == "wait":
            script.append((0.0, 0.0, 0.0, clamp(s.get("seconds", 1.0), 0.5, 5.0)))
        else:
            print("  知らない動作なのでスキップ:", s)
    script.append((0.0, 0.0, 0.0, 1.0))          # 最後に1秒立つ
    return script


def run(script):
    """台本どおりにシミュレーションし、動画に保存する（walk_script.py と同じ）"""
    bam = ip.load_bam_model(ip.BAM_KP_FW, 7.4, None)
    model, data, bam_ctrl, _ = ip.load_mujoco_with_bam(
        ip.MICRODUCK_XML, bam, 0.005, 0.1, ip.BAM_VIN_MIN)
    policy = ip.PolicyInference(
        model, data, walking_onnx_path=ONNX, bam_ctrl=bam_ctrl,
        use_projected_gravity=True, new_cmd_obs=True)

    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    adr = model.jnt_qposadr[fj]
    data.qpos[adr:adr + 3] = [0.0, 0.0, 0.125]
    data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    for i, q in enumerate(policy.joint_qpos_indices):
        data.qpos[q] = policy.default_pose[i]
    bam_ctrl.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = policy.trunk_base_id
    cam.distance, cam.elevation, cam.azimuth = 0.8, -20, 135

    frames = []
    for vx, vy, wz, sec in script:
        policy.set_vel_cmd(vx, vy, wz)
        for _ in range(int(sec * 50)):
            policy.apply_action(policy.infer())
            for _ in range(4):
                bam_ctrl.update()
                mujoco.mj_step(model, data)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
        x, y, z = data.qpos[adr:adr + 3]
        qw, qx, qy, qz = data.qpos[adr + 3:adr + 7]
        yaw = math.degrees(math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        print(f"  位置: x={x:+.2f} m, y={y:+.2f} m, 向き={yaw:+.0f}度 / 高さ {z * 1000:.0f} mm")

    imageio.mimsave(OUT, frames, fps=50)
    print("保存しました:", OUT)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('使い方: uv run python my/llm_walk.py "前に30cm進んで左を向いて"')
    instruction = sys.argv[1]
    print("指示:", instruction)
    plan = ask_llm(instruction)
    print("LLMの計画:", json.dumps(plan, ensure_ascii=False, indent=2))
    script = to_script(plan)
    for vx, vy, wz, sec in script:
        print(f"  台本: 前後{vx:+.2f} 左右{vy:+.2f} 旋回{wz:+.2f} を {sec:.1f}秒")
    run(script)
