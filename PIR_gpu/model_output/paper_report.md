# PIRセンサ時系列からTheia3D骨格座標を推定する時系列深層学習モデルの比較

## 要旨

本研究では、複数台のPIRセンサから得られる時系列電圧データを入力とし、Theia3Dで推定された3次元骨格座標を回帰するモデルを構築した。PIRデータはモジュールごとに5chを持ち、Theia3Dの教師データは25Hzのglobal joint座標として与えられる。PIRの実測サンプリング周期は必ずしも25Hzに一致しないため、Theia3Dの25Hzフレーム時刻を基準としてPIR特徴量を補間し、時刻整列済みデータセットを作成した。

評価では、Ridge回帰をベースラインとし、TCN、MLP、GRU、Transformer、ConvTransformer、および複数モデルの予測平均アンサンブルを比較した。trial `002`, `003`, `004` を学習に用い、trial `005` を未使用テストデータとして評価した。その結果、最良モデルはTCN 2秒、TCN 4秒、TCN 6秒、GRU 6秒の予測平均アンサンブルであり、全座標RMSEは240.1 mm、平均関節誤差は312.4 mm、R2は0.963であった。RidgeベースラインのRMSE 1184.1 mmと比較して、RMSEを79.7%削減した。

## 1. 背景と目的

PIRセンサは人物の動きに伴う赤外線変化を低コストかつ非接触に取得できる。一方で、PIR信号はカメラ画像や深度画像と異なり、空間的な情報が強く圧縮された時系列信号である。そのため、PIRデータから3次元骨格座標を推定するには、センサ配置、過去の信号変化、複数モジュール間の応答差を組み合わせて解釈する必要がある。

本研究の目的は、PIRセンサ時系列からTheia3Dの3D骨格座標を推定するための再現可能な学習・評価パイプラインを構築し、複数の時系列モデルを比較することである。特に、以下を重視した。

- Theia3Dの25Hzフレーム時刻に合わせたPIR特徴量の補間
- trial単位の学習・テスト分割による近接フレーム漏洩の抑制
- 全座標RMSE、関節別RMSE、フレーム平均関節誤差による評価
- ブラウザ上でのglobal座標と相対座標の3D可視化
- 複数モデルの比較とアンサンブルによる精度向上

## 2. データ

### 2.1 PIRデータ

PIRデータはtrialごとに `pir_data_192_168_11_x.csv` として保存されている。各CSVは1つのPIRモジュールに対応し、各モジュールは5chのPIR電圧値を持つ。本実験では5台のモジュールを使用したため、1時刻あたりの生PIR値は25次元である。

主な列は以下である。

- `timestamp_local`
- `timestamp_unix`
- `Time [sec]`
- `module_address`
- `PIR sensor 00 ... [V]`
- `PIR sensor 01 ... [V]`
- `PIR sensor 02 ... [V]`
- `PIR sensor 03 ... [V]`
- `PIR sensor 04 ... [V]`

本実験では、PIR時刻として `timestamp_unix` を優先して使用した。

### 2.2 Theia3D教師データ

Theia3Dの教師データは `skelton/Theia_Sub0.csv` に保存されている。先頭列は `Frame` であり、以降に各jointのXYZ座標がmm単位で格納されている。

Theia側のフレーム時刻は、`skelton/<trial>.tsv` に記録された `TIME_STAMP` と `FREQUENCY` を用いて算出した。

```text
Theia時刻 = TIME_STAMP + Frame / FREQUENCY
```

本データでは `FREQUENCY = 25 Hz` であった。

### 2.3 予測対象

予測対象はTheia3Dのglobal joint座標である。`worldbody` は座標が常に0となるため、予測対象から除外した。最終的な予測対象は19関節 x XYZの57次元である。

## 3. 前処理

### 3.1 25Hzフレーム時刻へのPIR補間

Theia3Dの教師データは25Hzである。一方、PIRの実測サンプリング周期はtrialやモジュールによって異なる可能性がある。そのため、学習サンプルはTheia3Dの25Hzフレーム時刻を基準として作成した。

各PIRモジュールについて、Theiaフレーム時刻に対応するPIR値を線形補間で求め、複数モジュールのチャンネルを結合した。これにより、教師座標をPIR側の低い実測FPSへ落とすのではなく、PIR特徴量をTheia側の25Hzへ合わせた。

### 3.2 特徴量

各フレームで使用した特徴量は375次元である。構成は以下の通りである。

- PIR生値
- 過去lag特徴量
  - `0.00`, `0.04`, `0.08`, `0.16`, `0.32`, `0.64`, `1.00`, `1.50`, `2.00`, `3.00` 秒
- 差分特徴量
  - 現在値と `0.16`, `0.64`, `1.50`, `3.00` 秒前との差
- 時間微分特徴量
  - 各PIRチャンネルの時間方向勾配

### 3.3 時系列窓

深層学習モデルでは、各時刻単体ではなく、過去数秒の特徴量列を入力した。検討した文脈長は2秒、4秒、6秒である。25Hzであるため、各文脈長に対応するフレーム数は以下となる。

| 文脈長 | 入力フレーム数 |
|---:|---:|
| 2秒 | 51 |
| 4秒 | 101 |
| 6秒 | 151 |

各入力系列の最後のフレームに対応するTheia3D座標を予測した。

## 4. モデル

### 4.1 Ridgeベースライン

まず、時系列特徴量を明示的なlag特徴量として展開し、Ridge回帰を学習した。これは深層学習モデルの性能を評価するための線形ベースラインである。

### 4.2 TCN

TCNは1次元畳み込みを用いた時系列モデルである。本実験では、dilationを増加させるResidual Blockを積み重ね、長い時間文脈を局所畳み込みの組み合わせとして扱った。PIR信号は局所的な変化や遅延応答が重要であるため、TCNは本タスクに適していると考えられる。

### 4.3 MLP

MLPは入力系列をflattenし、全結合層で直接座標を回帰するモデルである。実装は単純であるが、文脈長が長くなると入力次元が大きくなり、少数trialでは過学習しやすい。

### 4.4 GRU

GRUは再帰型ニューラルネットワークであり、時系列の状態を逐次的に更新する。PIR信号に含まれる時間的な遅れや履歴依存性を扱う目的で評価した。本実験では双方向GRUを用いた。

### 4.5 Transformer

Transformer Encoderは自己注意機構により系列内の長距離依存を扱うモデルである。PIR特徴量の長期文脈を直接扱える可能性がある一方、データ数が少ない場合は過学習しやすい。

### 4.6 ConvTransformer

ConvTransformerは、局所的な時間畳み込みでPIR信号の短期変化を抽出した後、Transformer Encoderで系列全体の関係を扱うモデルである。TCNとTransformerの中間的な構成として評価した。

### 4.7 予測平均アンサンブル

単体モデルの予測誤差にはモデルごとの差があるため、上位モデルの予測CSVをフレーム単位で揃え、予測座標を平均した。最良のアンサンブルは、TCN 2秒、TCN 4秒、TCN 6秒、GRU 6秒の4モデル平均であった。

## 5. 実験設定

### 5.1 データ分割

trial単位で以下の分割を行った。

| 用途 | trial |
|---|---|
| 学習 | `002`, `003`, `004` |
| テスト | `005` |

深層学習では、学習trial内の末尾15%程度を検証データとして用い、early stoppingに使用した。

### 5.2 学習設定

主な学習設定は以下である。

- 損失関数: `SmoothL1Loss(beta=0.5)`
- optimizer: `AdamW`
- scheduler: `ReduceLROnPlateau`
- early stopping: 検証lossに基づく
- GPU: RTX 4070 Ti SUPER
- 実行デバイス: `cuda`

### 5.3 評価指標

以下の指標を用いた。

- 全座標RMSE
  - 57次元のXYZ座標値全体に対するRMSE
- 全座標MAE
  - 57次元のXYZ座標値全体に対するMAE
- 平均関節誤差
  - 各フレームで19関節の3次元距離誤差を平均した値
- P95平均関節誤差
  - フレーム平均関節誤差の95パーセンタイル
- R2
  - 全global座標に対する決定係数

## 6. 結果

### 6.1 モデル比較

| 順位 | モデル | 種類 | 文脈長 [s] | テスト行数 | RMSE [mm] | MAE [mm] | 平均関節誤差 [mm] | P95平均関節誤差 [mm] | R2 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Ensemble_TCN2_TCN4_TCN6_GRU6 | 予測平均アンサンブル |  | 6613 | 240.1 | 138.0 | 312.4 | 873.0 | 0.963 |
| 2 | Ensemble_TCN2_TCN4_TCN6_GRU6_MLP2 | 予測平均アンサンブル |  | 6613 | 240.8 | 138.1 | 313.4 | 879.4 | 0.963 |
| 3 | Ensemble_TCN4_TCN6_GRU6 | 予測平均アンサンブル |  | 6613 | 244.7 | 142.4 | 322.7 | 866.3 | 0.961 |
| 4 | Ensemble_TCN4_GRU6 | 予測平均アンサンブル |  | 6613 | 247.3 | 145.0 | 330.2 | 886.1 | 0.960 |
| 5 | TCN_4s | TCN | 4.0 | 6663 | 260.5 | 153.4 | 347.9 | 901.2 | 0.957 |
| 6 | GRU_6s | GRU | 6.0 | 6613 | 266.6 | 158.6 | 365.2 | 916.3 | 0.954 |
| 7 | TCN_2s | TCN | 2.0 | 6713 | 267.8 | 156.2 | 357.3 | 971.1 | 0.954 |
| 8 | TCN_6s | TCN | 6.0 | 6613 | 269.6 | 156.5 | 355.3 | 934.9 | 0.953 |
| 9 | GRU_4s | GRU | 4.0 | 6663 | 289.0 | 177.2 | 409.7 | 932.1 | 0.946 |
| 10 | MLP_2s | MLP | 2.0 | 6713 | 293.8 | 169.7 | 389.0 | 1054.8 | 0.945 |
| 11 | Transformer_4s | Transformer | 4.0 | 6663 | 338.7 | 199.7 | 459.0 | 1182.2 | 0.927 |
| 12 | ConvTransformer_4s | ConvTransformer | 4.0 | 6663 | 355.0 | 209.5 | 482.9 | 1294.0 | 0.919 |
| 13 | MLP_4s | MLP | 4.0 | 6663 | 409.0 | 233.0 | 532.0 | 1459.6 | 0.893 |
| 14 | Ridge | Ridge回帰 |  | 6763 | 1184.1 | 829.7 | 1889.9 | 3219.5 | 0.103 |

### 6.2 ベースラインとの比較

RidgeベースラインのRMSEは1184.1 mmであった。最良アンサンブルのRMSEは240.1 mmであり、RMSE削減率は79.7%であった。

```text
RMSE削減率 = (1184.1 - 240.1) / 1184.1 = 79.7%
```

この結果から、PIR信号とTheia3D座標の対応関係は線形モデルだけでは十分に表現できず、時間文脈を扱う非線形モデルが有効であることが示唆される。

## 7. 考察

### 7.1 TCNの有効性

単体モデルではTCNが最も良い性能を示した。TCN 4秒モデルはRMSE 260.5 mmであり、GRU 6秒の266.6 mmよりも良好であった。PIR信号は、人物の移動に伴う局所的な時間変化やセンサ応答の遅れを含むため、dilated convolutionによって短期から中期の文脈を扱うTCNが適していたと考えられる。

### 7.2 文脈長の影響

TCNでは、2秒、4秒、6秒の文脈を比較した。単体モデルとしては4秒文脈が最良であった。6秒文脈では入力系列が長くなり、利用できる有効サンプル数が減るため、性能はわずかに悪化した。一方で、6秒TCNはアンサンブルに含めると性能向上に寄与したため、単体性能だけでなく誤差傾向の違いが重要である。

### 7.3 GRUの役割

GRUは4秒よりも6秒で性能が改善した。再帰型モデルは長い履歴を内部状態として扱えるため、より長い文脈でPIR応答の遅れを拾えた可能性がある。ただし、平均関節誤差ではTCN 4秒より大きく、単体最良にはならなかった。

### 7.4 Transformer系モデルの課題

Transformer 4秒とConvTransformer 4秒は、TCNやGRUに比べて性能が低かった。現在のデータは4 trialのみであり、Transformer系モデルの表現力に対して学習データが少ないため、過学習しやすかったと考えられる。より多くのtrial、別日データ、複数被験者データが得られれば、Transformer系の再評価には価値がある。

### 7.5 アンサンブルの効果

最良結果は、TCN 2秒、TCN 4秒、TCN 6秒、GRU 6秒の予測平均アンサンブルで得られた。単体最良のTCN 4秒よりもRMSEが20.4 mm改善した。これは、各モデルが異なる文脈長や時間構造を捉えており、誤差が完全には一致していないためと考えられる。

## 8. 可視化

3Dビューアを実装し、以下を確認できるようにした。

- global座標での真値と予測値
- pelvis基準の相対座標での真値と予測値
- モデルごとの予測結果の切り替え
- フレームごとの平均関節誤差
- 最悪関節とその誤差
- 現在の画角でのWebM動画書き出し

ビューアでは画面を7:3に分割し、左側にglobal座標、右側にpelvis基準の相対座標を表示する。これにより、絶対位置のずれと身体姿勢の相対的なずれを分けて観察できる。

## 9. 限界

本実験には以下の限界がある。

- テストは単一trial `005` のみである。
- 学習trialは3本であり、モデルの汎化性能を十分に評価するには少ない。
- 同一環境・同一センサ配置での評価であり、別環境への汎化は未検証である。
- 被験者差、服装、室温、センサ配置変更に対する頑健性は未評価である。
- 予測対象はTheia3Dのglobal座標であり、Theia3D自体の推定誤差は別途考慮していない。

## 10. 今後の課題

今後は以下を実施することで、論文としてより強い実験にできる。

- trial単位の交差検証
- 別日収録データでの評価
- 複数被験者での評価
- センサ配置変更に対する頑健性評価
- global座標と相対座標を同時に損失へ入れるmulti-task学習
- velocityやbone length制約を用いた物理的に自然な予測
- 不確実性推定による信頼度付き骨格推定

## 11. 再現手順

### 11.1 25Hz整列データ作成

```bash
python PIR_gpu/data/train_pir_to_global_bones.py \
  --trimmed-dir PIR_gpu/data \
  --output-dir PIR_gpu/model_output/baseline_ridge_v3 \
  --pir-time-source auto \
  --sync-mode timestamps
```

### 11.2 TCN 4秒モデルの学習

```bash
python PIR_gpu/data/train_pir_sequence_model.py \
  --dataset PIR_gpu/model_output/baseline_ridge_v3/aligned_dataset.npz \
  --output-dir PIR_gpu/model_output/sequence_tcn4s_gpu \
  --model tcn \
  --test-trials 005 \
  --context-sec 4.0 \
  --epochs 120 \
  --patience 16 \
  --batch-size 192 \
  --hidden-dim 192 \
  --layers 6 \
  --dropout 0.1 \
  --lr 0.002 \
  --weight-decay 0.0001 \
  --device cuda
```

### 11.3 アンサンブル

```bash
python PIR_gpu/data/ensemble_prediction_csvs.py \
  --inputs \
    PIR_gpu/model_output/sequence_tcn_gpu/test_predictions.csv \
    PIR_gpu/model_output/sequence_tcn4s_gpu/test_predictions.csv \
    PIR_gpu/model_output/sequence_tcn6s_gpu/test_predictions.csv \
    PIR_gpu/model_output/sequence_gru6s_gpu/test_predictions.csv \
  --names tcn2s tcn4s tcn6s gru6s \
  --output-dir PIR_gpu/model_output/ensemble_tcn2_tcn4_tcn6_gru6
```

### 11.4 比較ビューア

```bash
python PIR_gpu/data/create_prediction_viewer.py \
  --output-dir PIR_gpu/model_output/comparison_viewer \
  --model Ridge=PIR_gpu/model_output/baseline_ridge_v3/test_predictions.csv \
  --model TCN_4s=PIR_gpu/model_output/sequence_tcn4s_gpu/test_predictions.csv \
  --model Ensemble_TCN2_TCN4_TCN6_GRU6=PIR_gpu/model_output/ensemble_tcn2_tcn4_tcn6_gru6/test_predictions.csv
```

```bash
cd PIR_gpu/model_output/comparison_viewer
python -m http.server 8766
```

ブラウザで以下を開く。

```text
http://127.0.0.1:8766/index.html
```

## 12. 結論

PIRセンサ時系列からTheia3D骨格座標を推定するための一連の前処理、学習、評価、可視化パイプラインを構築した。Theia3Dの25Hzフレーム時刻にPIR特徴量を補間することで、教師座標の時間解像度を保った学習データを作成した。モデル比較では、TCN系モデルが単体で最も高い性能を示し、さらにTCNとGRUの予測平均アンサンブルによってRMSE 240.1 mm、平均関節誤差312.4 mmを達成した。

この結果は、PIR信号から3次元骨格座標を推定する問題において、局所的な時間畳み込みと複数文脈長の組み合わせが有効であることを示している。ただし、現段階では単一の未使用テストtrialでの評価であり、今後はtrial単位交差検証、追加収録、別環境評価によって汎化性能を検証する必要がある。
