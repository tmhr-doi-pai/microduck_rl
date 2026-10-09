# PAI最終課題：言葉で歩くMicroduck

日本語で「前に30cm進んで、左を向いて」と指示すると、二足歩行ロボット Microduck がシミュレーション内でその通りに歩くシステムです。

- 動画：（追加予定：YouTube URL）
- 作成者：tmhr_doi（omnicampusアカウント名）

## 概要

役割の異なる2つのAIを組み合わせています。

| 役割 | 担当 | 内容 |
|---|---|---|
| 頭（計画） | LLM（OpenRouter経由） | 日本語の指示を「動作の列（距離・角度）」に変換する |
| 足（運動） | 強化学習で学習した歩行ポリシー（ONNX） | 速度指令を受け取り、14個の関節を動かして転ばずに歩く |

LLMには「距離と角度」で考えさせ、速度と時間への換算はプログラム側で行います。換算には、実際にシミュレーションで測った速さを使っています。

## システム構成

```
日本語の指示
  ↓ LLM（JSONで動作の列を出力）
{"steps": [{"action": "forward", "distance_m": 0.3}, {"action": "turn_left", "angle_deg": 90}, ...]}
  ↓ 換算＋範囲チェック（my/llm_walk.py）
台本: (前後速度, 左右速度, 旋回速度, 秒) のリスト
  ↓ 歩行ポリシー duck_walk.onnx（50Hzで推論）
MuJoCo上のMicroduckが歩く → 動画に保存
```

## 追加したファイル

元のリポジトリ（pollen-robotics/microduck_rl）に対して、次のファイルを追加しました。それ以外はすべて元のリポジトリのコードです。

| ファイル | 内容 |
|---|---|
| `my/walk_script.py` | 手書きの台本どおりに歩かせ、動画を保存する（動作確認・速度の実測用） |
| `my/llm_walk.py` | 日本語の指示 → LLM → 台本 → 歩行 → 動画、を一括で行う本体 |
| `duck_walk.onnx` | 自分で学習した歩行ポリシー |

シミュレーションの部分は、元のリポジトリの `scripts/infer_policy.py` の部品（モーターモデルの読み込み、観測の作成、推論）を借りて使っています。

## 歩行ポリシーの学習

| 項目 | 内容 |
|---|---|
| タスク | `Mjlab-Velocity-Flat-MicroDuck`（平地で速度指令に従って歩く） |
| アルゴリズム | PPO（rsl_rl）、mjlab（MuJoCo Warp） |
| 並列環境数 | 4096 |
| 学習回数 | 14,750 iteration（約8時間、RTX 5060 Ti） |
| 入力 / 出力 | 観測61次元 / 関節目標角14次元、制御50Hz |

学習の初期は1回の評価あたり約25体が転倒していましたが、学習後は約0.4体まで減り、ほぼ転ばずに歩き続けられるようになりました。

## 実験結果

### 1. 台本どおりに歩かせた結果（my/walk_script.py）

| 区間 | 指令 | 実際の位置 | 解釈 |
|---|---|---|---|
| 立つ 2秒 | 0 | x=0.00 m | その場で足踏み |
| 前進 4秒 | 0.2 m/s | x=0.33 m | 約0.08 m/s で前進 |
| 左回転 3秒 | 1.0 rad/s | x=0.34 m | その場で約180度回転 |
| 前進 3秒 | 0.2 m/s | x=0.10 m | 逆向きになったため0.24 m戻った |
| 止まる 2秒 | 0 | x=0.10 m | その場で足踏み |

方向と回転は指令どおりでしたが、**前進の速さは指令の約4割（0.2 m/s → 実測 約0.08 m/s）**でした。そこで、LLMの計画を台本に換算するときは、この実測値を使っています。

### 2. 言葉で指示した結果（my/llm_walk.py）

指示：「前に30cm進んで、左を向いて、また20cm進んで」

| 区間 | 目標 | 実際 |
|---|---|---|
| 前進30cm | x=0.30 m | x=0.32 m |
| 左を向く | 向き+90度 | +100度（回転量+81度） |
| 前進20cm | 左へ0.20 m | 斜めに約0.2 m |
| ゴール | (0.30, 0.20) | (0.23, 0.24) |

LLMは指示を正しく動作の列に変換し、実測値による換算で距離もほぼ正確になりました。ゴールのずれ（約8cm）の主な原因は、前進中に向きが左へ約20度ずつずれていくことです。

## 工夫した点

- **LLMと歩行AIの役割分担**：LLMには距離と角度だけを考えさせ、物理的な速度・時間への換算はプログラムで行う構成にした。
- **実測にもとづく換算**：指令速度と実際の速さのずれを測定し、その値を換算に使うことで、移動距離の誤差を小さくした。
- **安全のための範囲チェック**：LLMの出力が想定外でも、距離・角度・時間を安全な範囲に収め、知らない動作は無視するようにした。

## 今後の課題

- 前進中の向きのずれを、姿勢の情報を使ったフィードバック制御で補正する。
- 実機のMicroduckで同じしくみを動かす（sim-to-real）。

## 実行方法

```bash
# 環境構築（CUDA対応GPUと uv が必要）
git clone https://github.com/tmhr-doi-pai/microduck_rl
cd microduck_rl
uv sync

# 学習済みの duck_walk.onnx を使って、台本どおりに歩かせる
uv run python my/walk_script.py

# 日本語の指示で歩かせる（OpenRouterのAPIキーが必要）
export OPENROUTER_API_KEY="（自分のキー）"
uv run python my/llm_walk.py "前に30cm進んで、左を向いて、また20cm進んで"
```

歩行ポリシーを自分で学習し直す場合：

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 --agent.logger tensorboard
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/velocity/（日時）_velocity/model_（番号）.pt \
  --onnx-file duck_walk.onnx
```

## 謝辞・ライセンス

- 元のリポジトリ：[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)（Apache License 2.0）
- 学習フレームワーク：[mjlab](https://github.com/mujocolab/mjlab)、モーターモデル：[BAM](https://github.com/Rhoban/bam)
- 開発の一部でAIアシスタント（Claude）を補助的に利用しました。
- このforkも Apache License 2.0 に従います。
