# PIR物理特性と骨格構造を考慮した次期アーキテクチャ案

## 目的

現在の最良モデルは、25Hz整列済みPIR特徴量をTCN/ConvTransformerへ入力し、Theia3Dのglobal joint座標57次元を直接回帰している。この構成はシンプルで強い一方、以下の構造を十分に使えていない。

- PIRは絶対温度画像ではなく、視野内の赤外線変化を持つスパースな時系列センサである。
- 天井配置では、各モジュール・各PIRチャンネルが異なる床面領域や人体部位への感度を持つ。
- Theia3D骨格は19関節の独立XYZではなく、pelvisをrootとする木構造と骨長制約を持つ。
- 歩行、しゃがみ、立ち上がりでは、global移動とpelvis相対姿勢の難しさが異なる。

本メモでは、PIRの物理的性質、天井配置、骨格制約を明示的に組み込む次期モデルを提案する。

## 関連研究からの示唆

mmWave radarやLiDARの骨格推定では、入力が点群やレンジ画像のようにスパースで情報量が少ないため、単一フレームではなくmulti-frame表現を使うことが多い。また、スパース点群をそのまま扱うだけでなく、Range-AzimuthやRange-Elevationの画像的表現へ投影してCNNで処理する方法、点群のトポロジーをgraphとして扱う方法、masked autoencoderで時空間表現を事前学習する方法が提案されている。

PIRはmmWaveやLiDARよりさらに情報が圧縮されているが、「少数センサの時系列から身体状態を復元する」という問題設定は近い。したがって、PIRでも以下が有効候補になる。

- multi-frame temporal encoder
- センサ配置・視野を使った幾何aware token化
- スパース観測を床面/身体空間のlatent heatmapへ戻す中間表現
- 骨格木・骨長・時間滑らかさを損失やdecoderに入れる
- 教師付き学習だけでなく、PIR時系列のmasked reconstructionや未来予測による事前学習を使う

## 提案1: Sensor-Structured TCN

### 狙い

現在の375次元入力をflattenしたまま扱うのではなく、`module x channel x feature_type` の構造へ戻して処理する。PIRの各モジュールと各センサチャンネルには意味があるため、初段で構造を潰さない。

### 入力表現

現在のfeature nameは以下の形式を持つ。

```text
192.168.11.2:PIR sensor 00 (center) [V]:lag_0.00s
192.168.11.2:PIR sensor 01 (tv) [V]:delta_0.16s
...
```

これを以下のtensorへ変換する。

```text
B x T x M x C x F
```

- `B`: batch
- `T`: 時系列窓
- `M`: PIRモジュール数
- `C`: 1モジュール内のPIRチャンネル数
- `F`: lag/delta/derivativeなどの特徴種別

### モデル構成

```text
PIR tensor
  -> channel encoder
  -> module encoder
  -> module graph attention
  -> temporal TCN/Transformer
  -> pelvis global head
  -> local pose head
```

### 期待する利点

- モジュール間の相対配置を学習しやすい。
- `center`, `tv`, `kitchen`, `closet`, `sofa` などチャンネル名の意味を埋め込みにできる。
- module dropoutやchannel dropoutを自然に入れられ、欠損モジュールに強くなる。

## 提案2: Ceiling Geometry Token

### 狙い

天井配置をモデルに渡す。現在はIPアドレスがfeature nameに入っているだけで、モジュールの物理位置や向きは使っていない。PIRは視野方向と床面位置に強く依存するため、配置情報を入れる価値が高い。

### 実装案

`pir_geometry.json` を追加する。

```json
{
  "192.168.11.2": {
    "name": "module_2",
    "position_mm": [0, 0, 2400],
    "yaw_deg": 0,
    "channels": {
      "PIR sensor 00 (center) [V]": {"direction": [0, 0, -1], "fov_deg": 35},
      "PIR sensor 01 (tv) [V]": {"direction": [1, 0, -1], "fov_deg": 35}
    }
  }
}
```

正確な配置がまだない場合は、まず仮の位置と学習可能embeddingを併用する。測定後に位置・向きを更新できるよう、モデル本体とは別ファイルにする。

### モデルへの入れ方

- module embedding: モジュールごとの天井位置、yaw、高さ
- channel embedding: 各PIRチャンネルの方向、FOV、部屋内ラベル
- attention bias: モジュール間距離に基づくbias
- soft beam basis: 各チャンネルが床面gridのどこを見ているかを表すsoft mask

## 提案3: PIR-to-Occupancy Latent

### 狙い

mmWave radarでスパース点群をRange-Azimuth画像へ投影する発想をPIRへ移植する。PIR信号から直接57次元を出すのではなく、一度「床面上の人物存在・移動・高さ変化」を表す低解像度latent mapへ変換する。

### 中間表現

```text
B x T x H x W x K
```

- `H x W`: 部屋床面grid
- `K`: occupancy, motion energy, vertical posture, confidenceなど

PIR各チャンネルのsoft beam maskを使い、信号を床面gridへback-projectする。完全な物理逆問題ではなく、学習可能なsoft projectionとして実装する。

### 利点

- global位置推定と姿勢推定を分けやすい。
- 歩行では床面位置・速度、しゃがみでは高さ変化latentが効く。
- viewerや論文で「PIRから推定された中間heatmap」として説明しやすい。

## 提案4: Bone-Aware Decoder

### 問題

現在は19関節XYZを独立に出しているため、骨長が揺れる、左右が不自然になる、しゃがみ時に足や膝の関係が破綻する可能性がある。

### 出力を分解する

```text
global pose = pelvis_global + local_skeleton
local_skeleton = forward_kinematics(bone_directions, bone_lengths)
```

予測対象を以下に分ける。

- pelvis global position: 3次元
- pelvis velocity: 補助タスク
- bone direction: 各骨の単位方向ベクトル
- bone length residual: 被験者固有の平均骨長からの小さい補正
- contact probability: 足裏接地の補助タスク

### 損失関数

```text
L = L_xyz
  + lambda_local * L_pelvis_relative
  + lambda_bone * L_bone_length
  + lambda_sym * L_left_right_symmetry
  + lambda_vel * L_velocity
  + lambda_acc * L_acceleration
  + lambda_floor * L_floor_contact
```

### 期待する効果

- 姿勢として不自然な予測を抑える。
- global移動誤差と身体形状誤差を分離して改善できる。
- しゃがみのような姿勢変化で、膝・足・pelvisの整合性を保ちやすい。

## 提案5: Multi-Task / Mixture-of-Experts

### 背景

混合学習では、ランダムウォーク専用モデルより歩行精度が下がり、しゃがみ入り単独モデルより全体精度は上がった。これは歩行としゃがみの信号分布が違うため、単一headが中間的な解を学んでいる可能性がある。

### 構成

```text
shared PIR encoder
  -> gait expert
  -> squat/transition expert
  -> router
  -> weighted pose prediction
```

routerはPIR時系列から条件をsoftに推定する。明示ラベルがない場合でも、trial名やTheiaのpelvis高さ変化から弱ラベルを作れる。

### 補助タスク

- 歩行/しゃがみ/遷移の状態分類
- pelvis高さ
- pelvis速度
- foot contact

これにより、モデルが「今は移動が大事」「今は高さ・膝曲げが大事」を切り替えやすくなる。

## 提案6: Self-Supervised PIR Pretraining

### 狙い

Theia付きデータは少ない。PIRだけなら今後も大量に取りやすいので、PIR時系列表現を事前学習する。

### タスク

- masked channel reconstruction: 一部モジュール/チャンネル/時間を隠して復元
- future prediction: 0.5秒後、1秒後のPIR特徴を予測
- temporal order prediction: 窓の順序が正しいか判定
- cross-module reconstruction: 一部モジュールを隠し、他モジュールから復元

PIRは動きに反応するため、masked reconstructionだけでなくfuture predictionが特に重要である。

## 優先実装順

### Stage 1: すぐ試す

1. `SensorStructuredTCN`
   - feature name parserで `M x C x F` に戻す。
   - module/channel embeddingを追加する。
   - TCN headは既存を流用する。
2. 骨格loss追加
   - bone length loss
   - pelvis-relative loss
   - velocity smoothness loss

この段階は既存スクリプトの拡張で実装できる。

### Stage 2: 精度改善の本命

1. `PIRGeometryTransformer`
   - `pir_geometry.json` を読み込む。
   - module graph attentionを追加する。
   - module/channel dropoutを学習時に入れる。
2. `BoneAwareDecoder`
   - pelvis global headとlocal skeleton headを分離する。
   - forward kinematicsでXYZを再構成する。

### Stage 3: 論文化しやすい発展

1. `PIR-to-Occupancy Latent`
   - soft beam basisで床面heatmapを生成。
   - global location headとpose headを分離。
2. `Mixture-of-Experts`
   - gait expertとsquat expertを導入。
   - routerを可視化して、動作ごとの専門化を示す。
3. `Self-Supervised PIR Pretraining`
   - TheiaなしPIRを活用。
   - masked/future reconstructionでencoderを事前学習。

## 次に実装する候補

最初の実装候補は `SensorStructuredTCN + bone/local losses` がよい。理由は以下である。

- 既存のTCNがすでに強く、encoderの置き換えだけで比較できる。
- 追加データなしで試せる。
- sensor/channel構造と骨格構造の両方を入れられる。
- 失敗してもbaselineとの比較が明確で、論文のablationにしやすい。

実験名は以下を推奨する。

```text
PIR-StructTCN
PIR-StructTCN+BoneLoss
PIR-GeoFormer
PIR-GeoFormer+BoneDecoder
PIR-MoE-GeoFormer
```

最初の評価では、既存の `PIR_gpu_mixed/model_output/baseline_ridge/aligned_dataset.npz` を使い、test trialは `rw_005` と `sit_008` を維持する。主要指標は全体RMSE、trial別RMSE、平均関節誤差、骨長変動、pelvis-relative MPJPEとする。

