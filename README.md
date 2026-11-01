# PAI最終課題：押すAIと見るAIで、ずらされても立て直すPushT

Physical AI応用1講座で配布された教材 **roboops-workshop**（PushT・LeRobot ACT）をもとに作成しました。

ロボットアームの手先（青い丸）がT字のブロックを押し、緑のゴールに合わせる課題（PushT）を、2つのAIの組み合わせで解きます。

| 役割 | AI | 中身 |
|---|---|---|
| 押すAI | LeRobot ACT（模倣学習） | 人の操作データ206件から学習。ブロックとゴールの位置から、手先の次の位置を推論する |
| 見るAI | Gemini Robotics-ER（`gemini-robotics-er-2-preview`） | 外乱の直前・直後・その80ステップ後・各回の最後にシミュレーター画面を見て「ずれている／もう少し／完了」を判定し、完了ならその回を終える |

作業の途中でブロックを突き飛ばす**外乱**を入れ、押すAIが立て直せるか、見るAIがその状態を正しく見分けられるかを試します。

## 教材から変えたところ

| ファイル | 中身 |
|---|---|
| `src/demo.py` | 新規。ACTでの推論、外乱、Geminiによる判定、字幕付き動画と記録の書き出し |
| `src/add_title.py` | 新規。録画済みの動画の冒頭に説明画面を足す |
| `configs/train_final.yaml` | 新規。CI用の設定よりモデルと学習量を大きくした学習設定 |
| `configs/eval_final.yaml` | 新規。外乱なしの成功率を20回分測る評価設定 |

`src/train.py`・`src/download_dataset.py` などは教材のままです。

## 動かし方

Docker が必要です。

```bash
docker compose build
docker compose run --rm workshop python src/download_dataset.py --max-episodes 206
docker compose run --rm workshop python src/train.py --config configs/train_final.yaml
```

見るAIを使う場合は、Google AI Studio で発行した API キーを `.env` に書きます（`.env` はコミットしません）。

```text
GEMINI_API_KEY=xxxxxxxx
```

```bash
docker compose run --rm workshop python src/demo.py
# Geminiなしで試すとき
docker compose run --rm workshop python src/demo.py --no-gemini
```

動画は `artifacts/demo/demo.mp4`、判定の記録は `artifacts/demo/demo_log.json` に出ます。提出動画は、外乱から立て直した2回と失敗した1回を選んで撮り、冒頭に説明画面を足したものです。

```bash
docker compose run --rm workshop python src/demo.py --seeds 3006 3016 3002 --no-fallback
docker compose run --rm workshop python src/add_title.py
```

見るAIの役は、上限に当たったら `gemini-flash-lite-latest` に切り替えます。`--no-fallback` を付けると Robotics-ER だけで判定します（無料枠は1日20回）。

## 出典

- 教材：Physical AI応用1講座 roboops-workshop
- データ：[lerobot/pusht_keypoints](https://huggingface.co/datasets/lerobot/pusht_keypoints)
- シミュレーター：[gym-pusht](https://github.com/huggingface/gym-pusht)
- 方策：[LeRobot](https://github.com/huggingface/lerobot) ACT
