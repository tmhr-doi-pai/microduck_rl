"""向きのフィードバック付き：日本語の指示 → LLM → 動作 → 歩行 → 動画

使い方（リポジトリのトップで実行）:
  uv run python my/llm_walk_fb.py "前に30cm進んで、左を向いて、また20cm進んで"
  uv run python my/llm_walk_fb.py --plan my/last_plan.json          # LLMを呼ばず、前回の計画で実行
  uv run python my/llm_walk_fb.py --plan my/last_plan.json --no-fb  # 比較用：補正なし（時間で動かす）
"""
import argparse
import json
import math
import sys

import mujoco
import imageio.v2 as imageio

import llm_walk as lw   # ask_llm, clamp, 換算の定数, ip(infer_policy) を再利用
ip = lw.ip

# --- フィードバック制御のパラメータ ---
K_YAW = 1.5                    # 向きのずれ[rad] → 旋回指令[rad/s] の比例ゲイン
MAX_WZ_WALK = 0.8              # 歩行中に加える補正旋回の上限 [rad/s]
TURN_TOL = math.radians(3)     # 回転を終える許容誤差
TURN_MIN = 0.4                 # 回転指令の最小値（小さすぎると回らないため）
PLAN_FILE = "my/last_plan.json"


def wrap(a):
    """角度を -π〜π に収める"""
    return (a + math.pi) % (2 * math.pi) - math.pi


def clip(v, lim):
    return max(-lim, min(lim, v))


def plan_to_actions(plan):
    """LLMの計画を ("walk", 距離m) / ("turn", 角度rad) / ("wait", 秒) の列にする（範囲チェック付き）"""
    acts = []
    for s in plan.get("steps", [])[:10]:
        a = s.get("action")
        if a == "forward":
            acts.append(("walk", lw.clamp(s.get("distance_m", 0.3), 0.05, 1.0)))
        elif a == "backward":
            acts.append(("walk", -lw.clamp(s.get("distance_m", 0.3), 0.05, 0.5)))
        elif a in ("turn_left", "turn_right"):
            ang = math.radians(lw.clamp(s.get("angle_deg", 90), 10, 360))
            acts.append(("turn", ang if a == "turn_left" else -ang))
        elif a == "wait":
            acts.append(("wait", lw.clamp(s.get("seconds", 1.0), 0.5, 5.0)))
        else:
            print("  知らない動作なのでスキップ:", s)
    return acts


def ideal_goal(acts):
    """計画どおりに完璧に動いた場合のゴール (x, y, 向き)"""
    x = y = yaw = 0.0
    for kind, v in acts:
        if kind == "walk":
            x += v * math.cos(yaw)
            y += v * math.sin(yaw)
        elif kind == "turn":
            yaw += v
    return x, y, yaw


class Sim:
    """MuJoCo上のMicroduck＋歩行ポリシー＋動画記録"""

    def __init__(self):
        bam = ip.load_bam_model(ip.BAM_KP_FW, 7.4, None)
        self.model, self.data, self.bam_ctrl, _ = ip.load_mujoco_with_bam(
            ip.MICRODUCK_XML, bam, 0.005, 0.1, ip.BAM_VIN_MIN)
        self.policy = ip.PolicyInference(
            self.model, self.data, walking_onnx_path=lw.ONNX, bam_ctrl=self.bam_ctrl,
            use_projected_gravity=True, new_cmd_obs=True)

        m, d, p = self.model, self.data, self.policy
        fj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.adr = m.jnt_qposadr[fj]
        d.qpos[self.adr:self.adr + 3] = [0.0, 0.0, 0.125]
        d.qpos[self.adr + 3:self.adr + 7] = [1, 0, 0, 0]
        for i, q in enumerate(p.joint_qpos_indices):
            d.qpos[q] = p.default_pose[i]
        self.bam_ctrl.reset(d.qpos)
        p.set_position_targets(p.default_pose)
        mujoco.mj_forward(m, d)

        self.renderer = mujoco.Renderer(m, height=480, width=640)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = p.trunk_base_id
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 0.8, -20, 135
        self.frames = []

    def pose(self):
        """胴体の (x, y, 向きrad, 高さ)"""
        d, a = self.data, self.adr
        x, y, z = d.qpos[a:a + 3]
        qw, qx, qy, qz = d.qpos[a + 3:a + 7]
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return float(x), float(y), yaw, float(z)

    def step(self, vx, wz):
        """速度指令を出して 1制御周期（1/50秒）進め、1コマ記録する"""
        p = self.policy
        p.vel_cmd[:] = (vx, 0.0, wz)     # set_vel_cmd() は毎回表示が出るので直接設定
        p._update_command()
        p.apply_action(p.infer())
        for _ in range(4):               # 物理は 200Hz
            self.bam_ctrl.update()
            mujoco.mj_step(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.cam)
        self.frames.append(self.renderer.render())

    def hold(self, sec):
        for _ in range(int(sec * 50)):
            self.step(0.0, 0.0)


def execute(sim, acts, feedback):
    target_yaw = 0.0                     # 本来向いているべき向き
    sim.hold(1.0)
    for kind, v in acts:
        x0, y0, _, _ = sim.pose()

        if kind == "walk":
            sign = 1.0 if v > 0 else -1.0
            dist = abs(v)
            t_nom = dist / lw.WALK_REAL                     # 実測の速さから見積もった時間
            max_steps = int((t_nom * 2 if feedback else t_nom) * 50)
            for _ in range(max_steps):
                x, y, yaw, _ = sim.pose()
                if feedback:
                    # 目標の向きに沿って進んだ距離
                    prog = sign * ((x - x0) * math.cos(target_yaw) + (y - y0) * math.sin(target_yaw))
                    if prog >= dist:
                        break
                    wz = clip(K_YAW * wrap(target_yaw - yaw), MAX_WZ_WALK)   # 向きのずれを戻す
                else:
                    wz = 0.0
                sim.step(sign * lw.WALK_CMD, wz)

        elif kind == "turn":
            target_yaw = wrap(target_yaw + v)
            if feedback:
                # 実際に回った角度を積算し、残りが無くなるまで回る（360度回転にも対応）
                remaining = v
                prev = sim.pose()[2]
                max_steps = int(abs(v) / lw.TURN_REAL * 2 * 50) + 50
                for _ in range(max_steps):
                    if abs(remaining) < TURN_TOL:
                        break
                    wz = clip(K_YAW * remaining, lw.TURN_CMD)
                    if abs(wz) < TURN_MIN:
                        wz = math.copysign(TURN_MIN, remaining)
                    sim.step(0.0, wz)
                    now = sim.pose()[2]
                    remaining -= wrap(now - prev)
                    prev = now
            else:
                for _ in range(int(abs(v) / lw.TURN_REAL * 50)):
                    sim.step(0.0, math.copysign(lw.TURN_CMD, v))

        elif kind == "wait":
            sim.hold(v)

        x, y, yaw, z = sim.pose()
        shown = f"{v:+.2f} m" if kind == "walk" else f"{math.degrees(v):+.0f} 度" if kind == "turn" else f"{v:.1f} 秒"
        print(f"  {kind:4s} {shown:>9s} → 位置 x={x:+.2f} y={y:+.2f} m, 向き={math.degrees(yaw):+.0f}度, 高さ {z * 1000:.0f} mm")
    sim.hold(1.0)


def main():
    ap = argparse.ArgumentParser(description="向きのフィードバック付き LLM歩行")
    ap.add_argument("instruction", nargs="?", help="日本語の指示")
    ap.add_argument("--plan", help="LLMを呼ばずに、保存済みの計画(JSON)を使う")
    ap.add_argument("--no-fb", action="store_true", help="比較用：フィードバック補正なし")
    args = ap.parse_args()

    if args.plan:
        with open(args.plan, encoding="utf-8") as f:
            plan = json.load(f)
    elif args.instruction:
        print("指示:", args.instruction)
        plan = lw.ask_llm(args.instruction)
        plan["instruction"] = args.instruction
        with open(PLAN_FILE, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        print("計画を保存しました:", PLAN_FILE)
    else:
        sys.exit(__doc__)

    print("計画:", json.dumps(plan, ensure_ascii=False, indent=2))
    acts = plan_to_actions(plan)
    gx, gy, gyaw = ideal_goal(acts)

    feedback = not args.no_fb
    print("モード:", "フィードバック補正あり" if feedback else "補正なし（時間で動かす）")
    sim = Sim()
    execute(sim, acts, feedback)

    x, y, yaw, _ = sim.pose()
    err = math.hypot(x - gx, y - gy)
    yaw_err = math.degrees(wrap(yaw - gyaw))
    print(f"理想のゴール: x={gx:+.2f} y={gy:+.2f} m, 向き={math.degrees(wrap(gyaw)):+.0f}度")
    print(f"実際のゴール: x={x:+.2f} y={y:+.2f} m, 向き={math.degrees(yaw):+.0f}度")
    print(f"→ 位置の誤差 {err * 100:.1f} cm / 向きの誤差 {yaw_err:+.0f} 度")

    out = "llm_walk_fb.mp4" if feedback else "llm_walk_nofb.mp4"
    imageio.mimsave(out, sim.frames, fps=50)
    print("保存しました:", out, f"（{len(sim.frames) / 50:.1f} 秒）")


if __name__ == "__main__":
    main()
