"""提出用動画を一括で作る（タイトル → しくみ → 学習 → 指示3つ → 補正の比較 → まとめ）

使い方（リポジトリのトップで実行）:
  uv run python my/make_video.py            # LLMに3つの指示を出して撮影（計画は my/video_plans.json に保存）
  uv run python my/make_video.py --reuse    # 保存済みの計画を使って撮り直す（LLMを呼ばない）

出力: pai_final_video.mp4（1280x720, 25fps）と my/video_results.json（誤差の記録）
"""
import os
import sys
import json
import math
import argparse
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

import llm_walk as lw
import llm_walk_fb as fb

ip = lw.ip

W, H = 1280, 720
FPS = 25                 # 制御50Hzのうち2回に1回を描画
OUT = "pai_final_video.mp4"
PLANS_FILE = "my/video_plans.json"
RESULTS_FILE = "my/video_results.json"
ACCOUNT = "easyMicrobit"

INSTRUCTIONS = [
    "前に30cm進んで、左を向いて、また20cm進んで",
    "右に90度回ってから、前に40cm進んで、左に45度向いて",
    "前に20cm進んで、くるっと一回転して、10cm後ろに下がって",
]

BG = (18, 24, 38)
WHITE = (240, 240, 240)
GRAY = (170, 178, 190)
ACCENT = (255, 196, 64)
GREEN = (110, 220, 140)


# ---------------------------------------------------------------- 文字・画面
def find_font():
    cands = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
        "/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf",
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    try:
        out = subprocess.run(["fc-list", ":lang=ja", "-f", "%{file}\n"],
                             capture_output=True, text=True).stdout.split("\n")
        out = [o for o in out if o.strip()]
        if out:
            return out[0]
    except Exception:
        pass
    sys.exit("日本語フォントが見つかりません。tmdoi で `sudo apt install fonts-noto-cjk` を実行してください")


FONT_PATH = find_font()
_fonts = {}


def font(size):
    if size not in _fonts:
        _fonts[size] = ImageFont.truetype(FONT_PATH, size)
    return _fonts[size]


class Writer:
    def __init__(self, path):
        self.w = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=16)
        self.n = 0

    def add(self, frame, times=1):
        for _ in range(times):
            self.w.append_data(frame)
            self.n += 1

    def close(self):
        self.w.close()


def card(writer, lines, seconds, align="center"):
    """lines: [(文字, 大きさ, 色), ...]  （文字が "" なら空行）"""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    gap = 18
    heights = [(font(s).getbbox("あ")[3] if t else s // 2) for t, s, _ in lines]
    total = sum(heights) + gap * (len(lines) - 1)
    y = (H - total) // 2
    for (t, s, c), h in zip(lines, heights):
        if t:
            w = d.textlength(t, font=font(s))
            x = (W - w) // 2 if align == "center" else 120
            d.text((x, y), t, font=font(s), fill=c)
        y += h + gap
    writer.add(np.asarray(img), int(seconds * FPS))


_layers = {}


def caption(frame, top=None, bottom=None):
    """映像の上下に帯つきの字幕を重ねる"""
    h, w = frame.shape[:2]
    key = (h, w, top, bottom)
    if key not in _layers:
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        if top:
            d.rectangle([0, 0, w, 72], fill=(0, 0, 0, 160))
            tw = d.textlength(top, font=font(34))
            d.text(((w - tw) // 2, 14), top, font=font(34), fill=WHITE + (255,))
        if bottom:
            d.rectangle([0, h - 64, w, h], fill=(0, 0, 0, 160))
            tw = d.textlength(bottom, font=font(30))
            d.text(((w - tw) // 2, h - 54), bottom, font=font(30), fill=ACCENT + (255,))
        arr = np.asarray(layer).astype(np.float32) / 255.0
        _layers[key] = (arr[..., :3] * 255.0, arr[..., 3:4])
    rgb, a = _layers[key]
    return (frame.astype(np.float32) * (1 - a) + rgb * a).astype(np.uint8)


# ---------------------------------------------------------------- 計画の表示
def act_jp(kind, v):
    if kind == "walk":
        return f"{'前進' if v > 0 else '後退'} {abs(v) * 100:.0f}cm"
    if kind == "turn":
        return f"{'左' if v > 0 else '右'}回転 {abs(math.degrees(v)):.0f}°"
    return f"待機 {v:.1f}秒"


def waypoints(acts):
    x = y = yaw = 0.0
    pts = [(0.0, 0.0)]
    for kind, v in acts:
        if kind == "walk":
            x += v * math.cos(yaw)
            y += v * math.sin(yaw)
            pts.append((x, y))
        elif kind == "turn":
            yaw += v
    return pts


# ---------------------------------------------------------------- シミュレーション
class Sim:
    def __init__(self, w, h, acts, on_frame):
        bam = ip.load_bam_model(ip.BAM_KP_FW, 7.4, None)
        self.model, self.data, self.bam_ctrl, _ = ip.load_mujoco_with_bam(
            ip.MICRODUCK_XML, bam, 0.005, 0.1, ip.BAM_VIN_MIN)
        m, d = self.model, self.data
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, w)
        m.vis.global_.offheight = max(m.vis.global_.offheight, h)
        self.policy = ip.PolicyInference(
            m, d, walking_onnx_path=lw.ONNX, bam_ctrl=self.bam_ctrl,
            use_projected_gravity=True, new_cmd_obs=True)
        p = self.policy

        fj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.adr = m.jnt_qposadr[fj]
        d.qpos[self.adr:self.adr + 3] = [0.0, 0.0, 0.125]
        d.qpos[self.adr + 3:self.adr + 7] = [1, 0, 0, 0]
        for i, q in enumerate(p.joint_qpos_indices):
            d.qpos[q] = p.default_pose[i]
        self.bam_ctrl.reset(d.qpos)
        p.set_position_targets(p.default_pose)
        mujoco.mj_forward(m, d)

        # 固定カメラ：計画した経路全体が入るように置く
        self.ideal = waypoints(acts)
        xs = [q[0] for q in self.ideal]
        ys = [q[1] for q in self.ideal]
        extent = max(max(xs) - min(xs), max(ys) - min(ys), 0.3)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, 0.05]
        self.cam.distance = 0.7 + 1.3 * extent
        self.cam.elevation = -38
        self.cam.azimuth = 20

        self.renderer = mujoco.Renderer(m, height=h, width=w)
        self.on_frame = on_frame
        self.tick = 0
        self.trail = [(0.0, 0.0)]

    def pose(self):
        d, a = self.data, self.adr
        x, y, z = d.qpos[a:a + 3]
        qw, qx, qy, qz = d.qpos[a + 3:a + 7]
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return float(x), float(y), yaw, float(z)

    def _add(self, sc, gtype, size, pos, rgba, frm=None, to=None, width=0.0):
        if sc.ngeom >= sc.maxgeom:
            return
        g = sc.geoms[sc.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.array(size, dtype=np.float64),
                            np.array(pos, dtype=np.float64), np.eye(3).flatten(),
                            np.array(rgba, dtype=np.float32))
        if frm is not None:
            mujoco.mjv_connector(g, gtype, width, np.array(frm, dtype=np.float64),
                                 np.array(to, dtype=np.float64))
        sc.ngeom += 1

    def render(self, label):
        r = self.renderer
        r.update_scene(self.data, camera=self.cam)
        sc = r.scene
        Z = 0.003
        cap = mujoco.mjtGeom.mjGEOM_CAPSULE
        sph = mujoco.mjtGeom.mjGEOM_SPHERE
        # 計画した経路（緑）
        for (x0, y0), (x1, y1) in zip(self.ideal[:-1], self.ideal[1:]):
            self._add(sc, cap, [0, 0, 0], [0, 0, 0], [0.2, 0.85, 0.35, 0.7],
                      [x0, y0, Z], [x1, y1, Z], 0.004)
        gx, gy = self.ideal[-1]
        self._add(sc, sph, [0.018, 0, 0], [gx, gy, Z], [0.2, 0.85, 0.35, 0.8])
        self._add(sc, sph, [0.012, 0, 0], [0, 0, Z], [1, 1, 1, 0.9])
        # 実際の軌跡（青）
        for (x0, y0), (x1, y1) in zip(self.trail[:-1], self.trail[1:]):
            self._add(sc, cap, [0, 0, 0], [0, 0, 0], [0.25, 0.55, 1.0, 0.9],
                      [x0, y0, Z + 0.001], [x1, y1, Z + 0.001], 0.003)
        self.on_frame(r.render(), label)

    def step(self, vx, wz, label):
        p = self.policy
        p.vel_cmd[:] = (vx, 0.0, wz)
        p._update_command()
        p.apply_action(p.infer())
        for _ in range(4):
            self.bam_ctrl.update()
            mujoco.mj_step(self.model, self.data)
        self.tick += 1
        if self.tick % 5 == 0:
            x, y, _, _ = self.pose()
            self.trail.append((x, y))
        if self.tick % 2 == 0:
            self.render(label)

    def hold(self, sec, label):
        for _ in range(int(sec * 50)):
            self.step(0.0, 0.0, label)

    def close(self):
        try:
            self.renderer.close()
        except Exception:
            pass


def execute(sim, acts, feedback):
    """llm_walk_fb.execute と同じ制御（字幕のラベル付き）"""
    target_yaw = 0.0
    sim.hold(1.0, "スタート")
    for kind, v in acts:
        label = "▶ " + act_jp(kind, v)
        x0, y0, _, _ = sim.pose()
        if kind == "walk":
            sign = 1.0 if v > 0 else -1.0
            dist = abs(v)
            t_nom = dist / lw.WALK_REAL
            max_steps = int((t_nom * 2 if feedback else t_nom) * 50)
            for _ in range(max_steps):
                x, y, yaw, _ = sim.pose()
                if feedback:
                    prog = sign * ((x - x0) * math.cos(target_yaw) + (y - y0) * math.sin(target_yaw))
                    if prog >= dist:
                        break
                    wz = fb.clip(fb.K_YAW * fb.wrap(target_yaw - yaw), fb.MAX_WZ_WALK)
                else:
                    wz = 0.0
                sim.step(sign * lw.WALK_CMD, wz, label)
        elif kind == "turn":
            target_yaw = fb.wrap(target_yaw + v)
            if feedback:
                remaining = v
                prev = sim.pose()[2]
                max_steps = int(abs(v) / lw.TURN_REAL * 2 * 50) + 50
                for _ in range(max_steps):
                    if abs(remaining) < fb.TURN_TOL:
                        break
                    wz = fb.clip(fb.K_YAW * remaining, lw.TURN_CMD)
                    if abs(wz) < fb.TURN_MIN:
                        wz = math.copysign(fb.TURN_MIN, remaining)
                    sim.step(0.0, wz, label)
                    now = sim.pose()[2]
                    remaining -= fb.wrap(now - prev)
                    prev = now
            else:
                for _ in range(int(abs(v) / lw.TURN_REAL * 50)):
                    sim.step(0.0, math.copysign(lw.TURN_CMD, v), label)
        elif kind == "wait":
            sim.hold(v, label)
    sim.hold(1.0, "完了")


def errors(sim, acts):
    gx, gy, gyaw = fb.ideal_goal(acts)
    x, y, yaw, z = sim.pose()
    return {
        "ideal": [round(gx, 3), round(gy, 3), round(math.degrees(fb.wrap(gyaw)), 1)],
        "actual": [round(x, 3), round(y, 3), round(math.degrees(yaw), 1)],
        "pos_err_cm": round(math.hypot(x - gx, y - gy) * 100, 1),
        "yaw_err_deg": round(math.degrees(fb.wrap(yaw - gyaw)), 1),
        "height_mm": round(z * 1000, 0),
    }


# ---------------------------------------------------------------- 構成
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="保存済みの計画を使う（LLMを呼ばない）")
    args = ap.parse_args()

    # 1) 計画
    if args.reuse:
        with open(PLANS_FILE, encoding="utf-8") as f:
            plans = json.load(f)
    else:
        plans = []
        for ins in INSTRUCTIONS:
            print("LLMに問い合わせ:", ins)
            plan = lw.ask_llm(ins)
            plan["instruction"] = ins
            plans.append(plan)
            print(json.dumps(plan, ensure_ascii=False))
        with open(PLANS_FILE, "w", encoding="utf-8") as f:
            json.dump(plans, f, ensure_ascii=False, indent=2)
        print("計画を保存しました:", PLANS_FILE)

    wr = Writer(OUT)
    results = {"demos": [], "comparison": {}}

    # 2) 導入
    card(wr, [("言葉で歩くMicroduck", 72, WHITE),
              ("LLM × 強化学習の歩行AI（MuJoCoシミュレーション）", 36, GRAY),
              ("", 20, WHITE),
              (f"PAI最終課題_{ACCOUNT}", 30, ACCENT)], 4)
    card(wr, [("しくみ", 52, ACCENT),
              ("", 16, WHITE),
              ("① 日本語の指示", 38, WHITE),
              ("↓  LLM（OpenRouter）が「動作の列」をJSONで出力", 32, GRAY),
              ("② 距離・角度 → 速度と時間に換算（実測値を使用）＋安全チェック", 32, GRAY),
              ("↓  向きのフィードバック補正", 32, GRAY),
              ("③ 強化学習で学習した歩行AIが14関節を動かして歩く", 38, WHITE)], 8, align="left")
    card(wr, [("歩行AIの学習", 52, ACCENT),
              ("", 16, WHITE),
              ("PPO（rsl_rl）＋ mjlab（MuJoCo Warp）", 36, WHITE),
              ("4096体を並列に練習・14,750回・約8時間（GPU 1枚）", 36, WHITE),
              ("転倒：1回あたり 約25体 → 約0.4体", 36, GREEN),
              ("", 16, WHITE),
              ("緑の線＝計画した経路 ／ 青の線＝実際に歩いた軌跡", 30, GRAY)], 6)

    # 3) 指示ごとのデモ（補正あり）
    for i, plan in enumerate(plans, 1):
        acts = fb.plan_to_actions(plan)
        ins = plan.get("instruction", "")
        lines = [(f"指示 {i}", 40, ACCENT), (f"「{ins}」", 38, WHITE), ("", 16, WHITE),
                 ("LLMの計画", 32, GRAY)]
        lines += [("・" + act_jp(k, v), 34, GREEN) for k, v in acts]
        card(wr, lines, 4)

        top = f"指示{i}：「{ins}」"
        sim = Sim(W, H, acts, lambda fr, lab: wr.add(caption(fr, top, lab)))
        execute(sim, acts, feedback=True)
        e = errors(sim, acts)
        res = f"ゴールとのずれ {e['pos_err_cm']:.1f} cm ／ 向き {e['yaw_err_deg']:+.0f}°"
        sim.hold(2.0, res)
        sim.close()
        e["instruction"] = ins
        results["demos"].append(e)
        print(f"指示{i}:", res)

    # 4) 補正あり・なしの比較（指示1の計画で）
    plan = plans[0]
    acts = fb.plan_to_actions(plan)
    card(wr, [("向きのフィードバック補正の効果", 48, ACCENT),
              ("", 16, WHITE),
              ("同じ計画を「補正なし（時間で動かす）」と「補正あり」で実行", 34, WHITE),
              ("補正あり：歩きながら向きのずれを測って戻す／目標の角度まで回る", 30, GRAY)], 4)
    runs = {}
    for name, feedback in (("nofb", False), ("fb", True)):
        frames = []
        sim = Sim(W // 2, H // 2, acts, lambda fr, lab: frames.append(fr))
        execute(sim, acts, feedback=feedback)
        runs[name] = (frames, errors(sim, acts))
        sim.close()
    n = max(len(runs["nofb"][0]), len(runs["fb"][0]))
    en, ef = runs["nofb"][1], runs["fb"][1]
    result_text = f"ずれ：補正なし {en['pos_err_cm']:.1f} cm → 補正あり {ef['pos_err_cm']:.1f} cm"
    for k in range(n + 2 * FPS):
        canvas = np.full((H, W, 3), BG, dtype=np.uint8)
        for j, name in enumerate(("nofb", "fb")):
            fr_list = runs[name][0]
            fr = fr_list[min(k, len(fr_list) - 1)]
            canvas[180:180 + H // 2, j * (W // 2):(j + 1) * (W // 2)] = fr
        bottom = result_text if k >= n else None
        img = caption(canvas, "左：補正なし　　　　右：補正あり", bottom)
        wr.add(img)
    results["comparison"] = {"no_feedback": en, "feedback": ef, "instruction": plan.get("instruction", "")}
    print(result_text)

    # 5) まとめ
    card(wr, [("まとめ", 52, ACCENT),
              ("", 16, WHITE),
              ("・LLMが日本語の指示を動作の列に変換し、歩行AIがその通りに歩いた", 32, WHITE),
              ("・指令した速さと実際の速さのずれを実測し、換算に反映した", 32, WHITE),
              ("・向きのフィードバック補正で、ゴールのずれを比較した", 32, WHITE),
              ("", 16, WHITE),
              ("今後：技の使い分け（座る・キック）、カメラ画像の活用、実機での検証", 30, GRAY),
              ("", 16, WHITE),
              ("コード：github.com/tmhr-doi-pai/microduck_rl", 28, GREEN)], 8, align="left")

    wr.close()
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"保存しました: {OUT}（{wr.n / FPS:.1f} 秒）／ 結果: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
