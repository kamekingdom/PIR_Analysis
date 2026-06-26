# PIR物理・骨格構造awareアーキテクチャ探索レポート

## 目的

既存の最良構成は、TCN 2秒、TCN 4秒、ConvTransformer 2秒の予測平均アンサンブルであり、混合データに対してRMSE 301.4 mm、平均関節誤差389.7 mmであった。本探索では、PIRの物理的性質、天井配置、全身骨格の構造をモデルへ入れることで、この性能を改善できるか検証した。

## 考えたアーキテクチャ

### 1. Sensor-Structured TCN

PIR特徴量375次元をflattenのまま扱うのではなく、feature名をparseして以下の構造へ戻した。

```text
batch x time x module x channel x feature_type
```

各PIRモジュール、各PIRチャンネルにembeddingを付与し、センサtokenとしてframe encoderへ入力した。狙いは、IPアドレス、チャンネル名、lag/delta/d_dtの構造をモデルが使えるようにすることである。

### 2. Hybrid Structured TCN

Sensor-Structured TCNでは25個のセンサtokenを1つのframe表現へpoolingするため、空間差が潰れすぎる可能性があった。そこで、既存のflatten入力をraw branchとして残し、structured branchと融合するhybrid型を実装した。

```text
raw 375-dim branch
structured module/channel branch
  -> fusion
  -> temporal TCN
```

### 3. Raw Bone TCN

既存TCN本体は維持し、lossに骨格制約を追加した。追加した制約は以下である。

- pelvis-relative pose loss
- bone length consistency loss

これは「モデル表現は既存の強いTCNを使い、出力側だけ人体らしさを誘導する」構成である。

### 4. RootLocal TCN

57次元global XYZを独立に出力するのではなく、以下のように分解して出力した。

```text
global pose = pelvis global position + pelvis-relative local skeleton
```

しゃがみ動作ではpelvis高さと相対姿勢が重要であるため、global移動と身体形状を分離することを狙った。

### 5. Physics Weighted Ensemble

物理・骨格awareモデル単体は既存TCN単体を超えなかったが、誤差傾向が異なる可能性がある。そこで既存上位モデルにRaw Bone TCNを少量混ぜる重み付きアンサンブルを探索した。

最終重みは以下である。

| モデル | 重み |
|---|---:|
| TCN 2s | 0.30 |
| TCN 4s | 0.25 |
| ConvTransformer 2s | 0.30 |
| RawBoneTCN 4s | 0.15 |

## 結果

| 順位 | モデル | RMSE [mm] | MAE [mm] | 平均関節誤差 [mm] | P95平均関節誤差 [mm] | R2 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | PhysicsWeighted_Ensemble | 300.5 | 170.6 | 388.2 | 1112.1 | 0.930 |
| 2 | PreviousBest_Ensemble | 301.4 | 171.2 | 389.7 | 1110.4 | 0.930 |
| 3 | TCN_4s | 318.5 | 182.4 | 415.1 | 1178.2 | 0.921 |
| 4 | TCN_2s | 321.8 | 183.5 | 418.4 | 1164.2 | 0.920 |
| 5 | ConvTransformer_2s | 326.7 | 186.9 | 427.7 | 1184.8 | 0.917 |
| 6 | RawBoneTCN_4s | 331.7 | 190.9 | 434.3 | 1222.7 | 0.915 |
| 7 | RootLocalTCN_4s | 338.2 | 189.8 | 433.2 | 1249.9 | 0.911 |
| 8 | HybridStructTCN_4s | 351.2 | 206.8 | 476.5 | 1248.1 | 0.904 |
| 9 | StructTCN_4s | 352.3 | 198.4 | 456.6 | 1301.0 | 0.904 |
| 10 | Ridge | 1126.6 | 820.9 | 1812.7 | 2865.2 | 0.015 |

## trial別結果

最良モデル `PhysicsWeighted_Ensemble` のtrial別結果は以下である。

| trial | 条件 | rows | RMSE [mm] | MAE [mm] | 平均関節誤差 [mm] | P95平均関節誤差 [mm] |
|---|---|---:|---:|---:|---:|---:|
| `rw_005` | ランダムウォーク | 6663 | 279.7 | 147.6 | 337.5 | 1087.2 |
| `sit_008` | ランダムウォーク＋しゃがみ | 6975 | 319.1 | 192.5 | 436.6 | 1121.8 |

前回best ensembleでは、`rw_005` がRMSE 280.5 mm、`sit_008` がRMSE 320.0 mmであった。したがって、改善幅は小さいが、歩行・しゃがみの両方でわずかに改善した。

## 解釈

今回の探索で最も重要な知見は、PIR入力を単純にmodule/channel tokenへ構造化するだけでは未使用trialの精度が改善しなかったことである。validation lossは良く見える場合があったが、testでは悪化した。これは、現在のデータ数ではセンサtoken表現が学習trialの配置・動作に過適合しやすいことを示している。

一方で、既存TCNに軽い骨格lossを加えたRaw Bone TCNは単体では既存TCNに届かなかったものの、アンサンブルに入れると小さく改善した。これは、骨格制約付きモデルの誤差傾向が既存TCN/ConvTransformerと少し異なり、相補的に働いたためと考えられる。

RootLocal TCNは、pelvis globalとlocal skeletonを分けるという設計上は自然な構成だが、単体精度はRaw Bone TCNより低かった。原因として、global座標再構成をpelvisに強く依存させたため、pelvis誤差が全関節へ伝播した可能性がある。次に試すなら、pelvis依存を完全固定せず、global residual headを追加する方がよい。

## 現時点の最良アーキテクチャ

現時点では、単一の新規アーキテクチャではなく、以下の重み付きアンサンブルが最良である。

```text
PhysicsWeighted_Ensemble
  = 0.30 * TCN_2s
  + 0.25 * TCN_4s
  + 0.30 * ConvTransformer_2s
  + 0.15 * RawBoneTCN_4s
```

この構成は、既存の時系列特徴抽出能力を維持しながら、骨格制約付きモデルを少量混ぜて人体構造のバイアスを加える、という現実的な落としどころである。

## 次にやるべき改善

今回の結果から、次の本命は以下である。

### 1. Pelvis + Local + Residual Decoder

RootLocal TCNを改良し、完全なroot依存ではなくglobal residualを追加する。

```text
global_pose = pelvis_global + local_skeleton + small_global_residual
```

これにより、pelvis誤差の全身伝播を抑えつつ、local skeletonの構造を使える。

### 2. 正確な `pir_geometry.json` の導入

現在のstructured modelはmodule/channel embeddingだけであり、実際の天井位置・向き・FOVを使っていない。真に物理awareにするには、各PIRモジュールの天井位置、向き、各チャンネルの視野方向を入れる必要がある。

### 3. trial-holdout validation

今回、validation lossとtest精度が一致しないケースがあった。今後はvalidationもフレーム分割ではなく、丸ごとtrial holdoutにするべきである。特に論文用には、leave-one-trial-out cross validationが必要である。

### 4. sensor/channel dropout

PIR構造化モデルは過適合しやすいため、学習時にmodule dropout、channel dropout、time maskingを入れるべきである。これはmmWave/radar系のmasked reconstructionやmulti-frame robustnessの考え方に近い。

## 出力

- 実装スクリプト: `PIR_gpu/data/train_pir_physics_model.py`
- 比較レポート: `PIR_gpu_mixed/model_output/physics_model_comparison/experiment_report.md`
- trial別評価: `PIR_gpu_mixed/model_output/physics_model_comparison/trial_metrics.csv`
- 最良予測: `PIR_gpu_mixed/model_output/physics_ensemble_weighted_tcn2_tcn4_conv2_bone/test_predictions.csv`
- viewer: `PIR_gpu_mixed/model_output/comparison_viewer/index.html`

