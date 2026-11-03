# RunPod 60개 trial 전체 조합 — activation 포함

6개 모델 조합 × 10개 trial. 각 행은 Optuna의 `optuna_trials.csv`와 해당 trial의 `resolved_config.yaml`을 대조했다. 모든 trial을 각각 나열하며, 원본 YAML에 명시되지 않은 생성자 기본값은 아래에 별도로 구분했다. 실수 값은 표에서 유효숫자 5자리로 표시했으며 [CSV](../results/runpod_tcn_20261101/60개_trial_전체조합_activation포함.csv)는 원본 정밀도를 보존한다.

각 표: CNN act=CNN activation, Seq act=시퀀스 activation, 분류 act=분류기 activation, MIL act=attention pooling 활성화, 분류 drop=분류기 dropout, Seq drop=시퀀스 dropout. `—`는 해당 시퀀스에 별도 activation 설정이 없음을 뜻한다. BiLSTM은 내부 gate에 sigmoid/tanh를 사용한다. AP는 validation 기준이며 독립 test가 아니다.

## 전체 공통 설정

- Adam; 최대 100 epoch; early stopping patience 10, min_delta 0.0001; seed 42; 환자당 최대 128개 30초 파형 구간; 판정 기준 0.5. train에는 원본과 증강 기록 포함.
- 분류기: hidden 32, activation GELU. Attention MIL: hidden 32, activation GELU, dropout 0.0. CNN activation은 모든 실험에서 GELU.
- Transformer와 Attention LSTM FCN의 설정된 activation은 GELU. BiLSTM에는 외부 activation 설정이 없고 LSTM 내부 sigmoid/tanh 게이트가 사용된다.
- 첫 TCN + Transformer의 trial 000~007만 batch 1/workers 0이고, 나머지는 batch 6/workers 4.
- 각 trial의 `resolved_config.yaml`에는 모든 생성자 기본값이 명시되지는 않는다. 현재 코드 및 실험 문서 기준 기본값: ResNet width 32·blocks (2,2), TCN dilations (1,2,4), Transformer embedding 64·heads 4·feedforward 128, LSTM hidden 32·recurrent layers 1, Attention LSTM FCN kernels (3,5). 과거 실행 코드와 현재 파일의 비트 단위 동일 여부는 이 표만으로 별도 검증하지 않았다.
- LSTM recurrent layer가 1개라 recurrent 층 사이 dropout은 적용되지 않는다. Attention LSTM FCN은 FCN 경로, BiLSTM은 시퀀스 출력에 설정 dropout을 적용한다.

## TCN + Transformer

| Trial | LR | WD | CNN act | TCN hidden / kernel | Seq act | Seq drop | Transformer 층 | 분류 act / drop | MIL act / drop | Batch / workers | Val AP |
|---:|---:|---:|---|---:|---|---:|---:|---|---|---:|---:|
| 000 | 0.00035747 | 0.00071145 | GELU | 64 / 5 | GELU | 0.1 | 1 | GELU / 0 | GELU / 0 | 1 / 0 | 0.1467 |
| 001 | 0.0001866 | 8.1795e-06 | GELU | 16 / 5 | GELU | 0 | 1 | GELU / 0 | GELU / 0 | 1 / 0 | 0.1554 |
| 002 | 0.00078962 | 3.2477e-06 | GELU | 16 / 3 | GELU | 0.1 | 1 | GELU / 0.3 | GELU / 0 | 1 / 0 | 0.1300 |
| 003 | 0.00095198 | 8.6126e-06 | GELU | 16 / 3 | GELU | 0 | 2 | GELU / 0.1 | GELU / 0 | 1 / 0 | 0.1418 |
| 004 | 0.00037508 | 6.517e-06 | GELU | 64 / 5 | GELU | 0 | 1 | GELU / 0 | GELU / 0 | 1 / 0 | 0.1004 |
| 005 | 0.0011935 | 0.00020597 | GELU | 16 / 5 | GELU | 0.1 | 1 | GELU / 0.1 | GELU / 0 | 1 / 0 | 0.1294 |
| 006 | 0.00015019 | 0.00013795 | GELU | 32 / 5 | GELU | 0.1 | 2 | GELU / 0.3 | GELU / 0 | 1 / 0 | 0.1608 |
| 007 | 0.00023347 | 1.7027e-05 | GELU | 64 / 3 | GELU | 0 | 1 | GELU / 0 | GELU / 0 | 1 / 0 | 0.1316 |
| 008 | 0.0015584 | 0.00048784 | GELU | 64 / 5 | GELU | 0 | 2 | GELU / 0 | GELU / 0 | 6 / 4 | 0.1573 |
| 009 | 0.0003002 | 3.6006e-05 | GELU | 16 / 3 | GELU | 0.1 | 2 | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.1232 |

## ResNet + Attention LSTM FCN

| Trial | LR | WD | CNN act | TCN hidden / kernel | Seq act | Seq drop | Transformer 층 | 분류 act / drop | MIL act / drop | Batch / workers | Val AP |
|---:|---:|---:|---|---:|---|---:|---:|---|---|---:|---:|
| 000 | 0.00035747 | 0.00071145 | GELU | — | GELU | 0.3 | — | GELU / 0 | GELU / 0 | 6 / 4 | 0.2380 |
| 001 | 0.00077254 | 0.00013311 | GELU | — | GELU | 0 | — | GELU / 0.1 | GELU / 0 | 6 / 4 | 0.1715 |
| 002 | 0.00028145 | 3.7521e-05 | GELU | — | GELU | 0.3 | — | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.1323 |
| 003 | 0.00047171 | 0.00022674 | GELU | — | GELU | 0.1 | — | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.3211 |
| 004 | 0.00012476 | 0.00070252 | GELU | — | GELU | 0.1 | — | GELU / 0 | GELU / 0 | 6 / 4 | 0.1568 |
| 005 | 0.00015145 | 3.0587e-05 | GELU | — | GELU | 0 | — | GELU / 0.1 | GELU / 0 | 6 / 4 | 0.1546 |
| 006 | 0.00064203 | 3.5856e-06 | GELU | — | GELU | 0.3 | — | GELU / 0 | GELU / 0 | 6 / 4 | 0.1889 |
| 007 | 0.00013512 | 3.8721e-06 | GELU | — | GELU | 0.1 | — | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.1816 |
| 008 | 0.00026 | 4.2471e-05 | GELU | — | GELU | 0 | — | GELU / 0.1 | GELU / 0 | 6 / 4 | 0.1471 |
| 009 | 0.0001019 | 0.0002795 | GELU | — | GELU | 0.1 | — | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.1671 |

## TCN + Attention LSTM FCN

| Trial | LR | WD | CNN act | TCN hidden / kernel | Seq act | Seq drop | Transformer 층 | 분류 act / drop | MIL act / drop | Batch / workers | Val AP |
|---:|---:|---:|---|---:|---|---:|---:|---|---|---:|---:|
| 000 | 0.00035747 | 0.00071145 | GELU | 64 / 5 | GELU | 0.1 | — | GELU / 0 | GELU / 0 | 6 / 4 | 0.1500 |
| 001 | 0.0002059 | 3.5114e-06 | GELU | 64 / 5 | GELU | 0.3 | — | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.1145 |
| 002 | 0.00019722 | 3.489e-05 | GELU | 64 / 3 | GELU | 0.3 | — | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.1258 |
| 003 | 0.00044685 | 2.3234e-06 | GELU | 32 / 5 | GELU | 0.1 | — | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.2491 |
| 004 | 0.002442 | 0.0004836 | GELU | 64 / 3 | GELU | 0 | — | GELU / 0.1 | GELU / 0 | 6 / 4 | 0.1825 |
| 005 | 0.00063333 | 2.6471e-06 | GELU | 16 / 3 | GELU | 0.1 | — | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.1690 |
| 006 | 0.00033845 | 2.2264e-06 | GELU | 64 / 3 | GELU | 0 | — | GELU / 0 | GELU / 0 | 6 / 4 | 0.1434 |
| 007 | 0.0011312 | 0.00019158 | GELU | 16 / 3 | GELU | 0 | — | GELU / 0.1 | GELU / 0 | 6 / 4 | 0.1562 |
| 008 | 0.0021907 | 5.596e-06 | GELU | 32 / 3 | GELU | 0.1 | — | GELU / 0.1 | GELU / 0 | 6 / 4 | 0.1200 |
| 009 | 0.00018862 | 0.00047608 | GELU | 16 / 5 | GELU | 0 | — | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.1866 |

## ResNet + BiLSTM

| Trial | LR | WD | CNN act | TCN hidden / kernel | Seq act | Seq drop | Transformer 층 | 분류 act / drop | MIL act / drop | Batch / workers | Val AP |
|---:|---:|---:|---|---:|---|---:|---:|---|---|---:|---:|
| 000 | 0.00035747 | 0.00071145 | GELU | — | — (LSTM gate) | 0.3 | — | GELU / 0 | GELU / 0 | 6 / 4 | 0.1392 |
| 001 | 0.00077254 | 0.00013311 | GELU | — | — (LSTM gate) | 0 | — | GELU / 0.1 | GELU / 0 | 6 / 4 | 0.1650 |
| 002 | 0.00028145 | 3.7521e-05 | GELU | — | — (LSTM gate) | 0.3 | — | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.2257 |
| 003 | 0.00047171 | 0.00022674 | GELU | — | — (LSTM gate) | 0.1 | — | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.1670 |
| 004 | 0.00012476 | 0.00070252 | GELU | — | — (LSTM gate) | 0.1 | — | GELU / 0 | GELU / 0 | 6 / 4 | 0.1493 |
| 005 | 0.00015145 | 3.0587e-05 | GELU | — | — (LSTM gate) | 0 | — | GELU / 0.1 | GELU / 0 | 6 / 4 | 0.2051 |
| 006 | 0.00064203 | 3.5856e-06 | GELU | — | — (LSTM gate) | 0.3 | — | GELU / 0 | GELU / 0 | 6 / 4 | 0.2210 |
| 007 | 0.00013512 | 3.8721e-06 | GELU | — | — (LSTM gate) | 0.1 | — | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.2519 |
| 008 | 0.00026 | 4.2471e-05 | GELU | — | — (LSTM gate) | 0 | — | GELU / 0.1 | GELU / 0 | 6 / 4 | 0.1795 |
| 009 | 0.0001019 | 0.0002795 | GELU | — | — (LSTM gate) | 0.1 | — | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.2133 |

## TCN + BiLSTM

| Trial | LR | WD | CNN act | TCN hidden / kernel | Seq act | Seq drop | Transformer 층 | 분류 act / drop | MIL act / drop | Batch / workers | Val AP |
|---:|---:|---:|---|---:|---|---:|---:|---|---|---:|---:|
| 000 | 0.00035747 | 0.00071145 | GELU | 64 / 5 | — (LSTM gate) | 0.1 | — | GELU / 0 | GELU / 0 | 6 / 4 | 0.1207 |
| 001 | 0.0002059 | 3.5114e-06 | GELU | 64 / 5 | — (LSTM gate) | 0.3 | — | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.1586 |
| 002 | 0.00019722 | 3.489e-05 | GELU | 64 / 3 | — (LSTM gate) | 0.3 | — | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.1356 |
| 003 | 0.00044685 | 2.3234e-06 | GELU | 32 / 5 | — (LSTM gate) | 0.1 | — | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.1725 |
| 004 | 0.002442 | 0.0004836 | GELU | 64 / 3 | — (LSTM gate) | 0 | — | GELU / 0.1 | GELU / 0 | 6 / 4 | 0.1825 |
| 005 | 0.00063333 | 2.6471e-06 | GELU | 16 / 3 | — (LSTM gate) | 0.1 | — | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.1409 |
| 006 | 0.00033845 | 2.2264e-06 | GELU | 64 / 3 | — (LSTM gate) | 0 | — | GELU / 0 | GELU / 0 | 6 / 4 | 0.1589 |
| 007 | 0.0011312 | 0.00019158 | GELU | 16 / 3 | — (LSTM gate) | 0 | — | GELU / 0.1 | GELU / 0 | 6 / 4 | 0.1268 |
| 008 | 0.0021907 | 5.596e-06 | GELU | 32 / 3 | — (LSTM gate) | 0.1 | — | GELU / 0.1 | GELU / 0 | 6 / 4 | 0.1602 |
| 009 | 0.00018862 | 0.00047608 | GELU | 16 / 5 | — (LSTM gate) | 0 | — | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.1900 |

## ResNet + Transformer

| Trial | LR | WD | CNN act | TCN hidden / kernel | Seq act | Seq drop | Transformer 층 | 분류 act / drop | MIL act / drop | Batch / workers | Val AP |
|---:|---:|---:|---|---:|---|---:|---:|---|---|---:|---:|
| 000 | 0.00035747 | 0.00071145 | GELU | — | GELU | 0.3 | 2 | GELU / 0 | GELU / 0 | 6 / 4 | 0.2341 |
| 001 | 0.00010725 | 0.00081232 | GELU | — | GELU | 0.3 | 1 | GELU / 0 | GELU / 0 | 6 / 4 | 0.1624 |
| 002 | 0.00080127 | 2.6211e-06 | GELU | — | GELU | 0 | 1 | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.1645 |
| 003 | 0.00078962 | 3.2477e-06 | GELU | — | GELU | 0 | 1 | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.1175 |
| 004 | 0.00015145 | 3.0587e-05 | GELU | — | GELU | 0 | 1 | GELU / 0.1 | GELU / 0 | 6 / 4 | 0.1256 |
| 005 | 0.0027052 | 0.00021154 | GELU | — | GELU | 0 | 2 | GELU / 0 | GELU / 0 | 6 / 4 | 0.3157 |
| 006 | 0.00037508 | 6.517e-06 | GELU | — | GELU | 0.3 | 2 | GELU / 0 | GELU / 0 | 6 / 4 | 0.2340 |
| 007 | 0.0013826 | 3.9459e-06 | GELU | — | GELU | 0.1 | 1 | GELU / 0.1 | GELU / 0 | 6 / 4 | 0.1723 |
| 008 | 0.0018833 | 7.4113e-05 | GELU | — | GELU | 0.1 | 1 | GELU / 0 | GELU / 0 | 6 / 4 | 0.1939 |
| 009 | 0.00015019 | 0.00013795 | GELU | — | GELU | 0.1 | 2 | GELU / 0.3 | GELU / 0 | 6 / 4 | 0.1565 |
