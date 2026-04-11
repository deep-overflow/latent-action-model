# Experiment: LAM Flow vs Baseline — Linear Probing Comparison

## Purpose
- **Hypothesis**: Optical flow prediction을 decoder target으로 사용하면, pixel reconstruction 대비 더 의미 있는 latent action을 학습할 수 있다.
- **근거**: Pixel reconstruction은 배경, 조명 등 action과 무관한 정보를 복원하는 데 capacity를 소비하지만, optical flow는 순수한 motion 정보만 담고 있어 latent action이 dynamics에 집중하도록 유도한다.
- **검증 방법**: LAM의 32-dim latent action에서 실제 로봇 29-dim delta action으로의 linear regression R²를 비교한다. R²가 높을수록 latent action이 실제 action과 선형적으로 일치함을 의미한다.

## Setup

### Model Architecture
- **LAM (Latent Action Model)**: VAE 기반
  - Encoder: SpatioTemporalTransformer (2-frame RGB 입력 → 32-dim latent z)
  - Decoder: SpatioTransformer (latent z + frame_t → target 예측)
  - Model dim: 768, 16 enc/dec blocks, 12 heads, patch size 16
  - Input resolution: 240 x 320
  - Total params: 267M

### Training Variants

| 항목 | Baseline (lam_medium) | Flow (lam_medium_flow) |
|---|---|---|
| Decoder target | RGB pixel reconstruction | Optical flow (2-ch) |
| Decoder output | 3-ch (sigmoid) | 2-ch (no activation) |
| GT source | 입력 프레임 그대로 | SEA-RAFT precomputed + cycle consistency |
| Loss | MSE (pixel) + β·KL | Masked MSE (flow) + β·KL |
| Frame skip | random 1-4 | **1 (강제)** |
| β (KL weight) | 1e-6 | 1e-6 |
| Batch size | 40/device × 2 GPU × 2 grad_accum = 160 | 동일 |
| LR | 2.5e-5 (AdamW) | 동일 |
| Precision | fp16-mixed | 동일 |
| GPU | 2x Blackwell RTX PRO 5000 | 동일 |

### Dataset
- **GR-1 Robot Teleop**: 997 episodes, ego-view 카메라 (20Hz)
- 경로: `/media/data1/chan/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot`
- Samples per epoch: 100,000

### Optical Flow GT
- **Model**: SEA-RAFT (spring-M config, Tartan-C-T-TSKH pretrained)
- **Cycle consistency**: forward + backward flow 계산, |f_AB + warp(f_BA, f_AB)| < 1.0 px인 픽셀만 reliable
- **저장 형식**: per-video directory에 `flow.npy` (float16) + `mask.npy` (bool), mmap 지원
- **Mask 통과율**: ~98-99% (대부분의 픽셀이 reliable)

### Training Duration

| Model | Epochs | Steps | Wall Time |
|---|---|---|---|
| Baseline | 8 (epoch 0-7) | 5,000 | ~4h |
| Flow | 35 (epoch 0-34) | 21,875 | ~17h |

### Linear Probing Setup
- **Feature extraction**: LAM encoder의 z_rep (32-dim, deterministic μ in eval mode)
- **Target**: 29-dim normalized delta action (per-frame joint position difference)
  - left_arm (7), right_arm (7), left_hand (6), right_hand (6), waist (3)
  - Min-max normalized to [-1, 1] using dataset stats
- **Probe**: `nn.Linear(32, 29)`, Adam lr=1e-3, batch 512
- **Input standardization**: z_rep을 train mean/std로 정규화 (baseline과 flow의 z scale 차이 보정)
- **Early stopping**: patience=50 epochs (max 500)
- **Data**: 500 episodes, frame_stride=4 → 21,841 samples (17,472 train / 4,369 val)
- **Metric**: R² (coefficient of determination), zero-variance dims (waist_2) 제외

## Results (Facts)

### Matched Comparison (8 epochs each)

#### Overall

| Metric | Baseline (ep7) | Flow (ep7) |
|---|---|---|
| Val MSE | 0.000443 | **0.000433** |
| **Overall R²** | 0.0091 | **0.0385** |
| Probe convergence | epoch 264 | epoch 148 |

#### Per-Group R²

| Joint Group | Baseline (ep7) | Flow (ep7) | Δ |
|---|---|---|---|
| left_arm (7 joints) | 0.035 | **0.141** | +0.106 |
| right_arm (7 joints) | 0.063 | **0.202** | +0.139 |
| left_hand (6 joints) | -0.013 | -0.011 | ~0 |
| right_hand (6 joints) | 0.002 | **0.022** | +0.020 |
| waist (2 joints) | **0.427** | 0.420 | ~0 |

#### Per-Joint R² (주요 차이)

| Joint | Baseline | Flow | Δ |
|---|---|---|---|
| left_arm_0 | 0.108 | **0.468** | +0.360 |
| left_arm_3 | 0.071 | **0.295** | +0.224 |
| right_arm_0 | 0.094 | **0.542** | +0.448 |
| right_arm_1 | 0.335 | **0.450** | +0.115 |
| right_arm_2 | 0.213 | **0.283** | +0.070 |
| right_arm_3 | 0.080 | **0.316** | +0.236 |
| waist_0 | 0.605 | **0.616** | +0.011 |

### Flow LAM Epoch Scaling

#### Overall

| Metric | Flow ep7 | Flow ep20 | Flow ep34 |
|---|---|---|---|
| Val MSE | 0.000433 | 0.000475 | 0.000573 |
| **Overall R²** | 0.0385 | 0.0377 | 0.0379 |
| Probe convergence | epoch 148 | epoch 105 | epoch 41 |

#### Per-Group R²

| Joint Group | Flow ep7 | Flow ep20 | Flow ep34 |
|---|---|---|---|
| left_arm | 0.141 | 0.183 | **0.199** |
| right_arm | 0.202 | 0.249 | **0.256** |
| left_hand | -0.011 | -0.008 | **0.002** |
| right_hand | 0.022 | 0.016 | 0.020 |
| waist | 0.420 | **0.488** | 0.482 |

#### Full 4-way Comparison (Per-Group R²)

| Joint Group | baseline ep7 | flow ep7 | flow ep20 | flow ep34 |
|---|---|---|---|---|
| left_arm | 0.035 | 0.141 | 0.183 | **0.199** (5.7×) |
| right_arm | 0.063 | 0.202 | 0.249 | **0.256** (4.1×) |
| left_hand | -0.013 | -0.011 | -0.008 | 0.002 |
| right_hand | 0.002 | 0.022 | 0.016 | 0.020 |
| waist | 0.427 | 0.420 | **0.488** | 0.482 |
| **Overall** | 0.0091 | 0.0385 | 0.0377 | 0.0379 |

*(괄호 안 숫자: vs baseline 배수)*

### Flow LAM Training Convergence

| Epoch | Train Loss | EPE (px) |
|---|---|---|
| 0 | 0.5648 | 0.547 |
| 1 | 0.2326 | 0.309 |
| 5 | 0.1137 | 0.224 |
| 10 | 0.0785 | 0.183 |
| 13 | 0.0684 | 0.168 |
| 20 | ~0.05 | ~0.15 |
| 34 | ~0.04 | ~0.13 |

## Analysis (Opinion)

### Flow LAM Epoch 효과 (ep7 vs ep20 vs ep34)
1. **Overall R²은 거의 평탄** (0.0385 → 0.0377 → 0.0379): probing 관점에서 flow LAM은 ep7만으로 충분히 학습됨. 14배 더 학습해도 전체 선형 가능성은 거의 그대로.
2. **하지만 group별로는 미세한 점진적 개선**:
   - left_arm: 0.141 → 0.183 → **0.199** (꾸준히 증가)
   - right_arm: 0.202 → 0.249 → **0.256** (꾸준히 증가)
   - waist: 0.420 → **0.488** → 0.482 (ep20에서 peak, ep34는 약간 감소)
3. **MSE는 ep34에서 오히려 증가** (0.000433 → 0.000573): 과적합 또는 latent 분포가 더 복잡해져 linear probe로 일부 손실. 다만 R²은 거의 동일한 것을 보면 latent 정보 자체는 여전히 유효.
4. **Probe 수렴 속도가 빨라짐** (148 → 105 → 41 epochs): 잘 학습된 flow latent일수록 linear probe가 빠르게 수렴.
5. **시사점**: flow LAM은 적은 학습량(epoch 7)으로도 latent action 품질이 거의 plateau에 도달. 추가 학습의 marginal value는 작음.

### Flow가 Baseline보다 좋은 이유
1. **Optical flow는 순수한 motion 정보**: pixel reconstruction은 texture, 조명, 배경 등 action과 무관한 시각 정보를 복원하는 데 decoder capacity를 소비한다. Flow는 오직 "무엇이 얼마나 움직였는가"만 인코딩하므로, latent action이 dynamics에 더 집중한다.
2. **팔 관절에서 차이가 큰 이유**: 팔의 움직임은 ego-view에서 큰 optical flow를 생성하고, flow prediction 학습 시 이 motion을 잘 캡처한다. Pixel reconstruction은 팔의 외형(텍스처)도 같이 복원해야 해서 motion 정보가 희석된다.
3. **Hand에서 차이 없는 이유**: 손가락 움직임은 (1) 매우 작은 visual change를 만들고 (2) ego-view에서 잘 안 보이며 (3) self-occlusion이 심해서 flow든 pixel이든 latent에 인코딩하기 어렵다.
4. **Waist에서 비슷한 이유**: 허리 회전은 전체 화면의 큰 변화를 만들어 두 방식 모두 쉽게 캡처한다.

### R² 자체가 낮은 이유 (둘 다 < 0.1)
1. **Delta action의 variance가 매우 작음** (std ~0.005): 20Hz로 캡처된 데이터에서 연속 프레임 간 joint 변화가 미미
2. **Visual change → joint delta 관계가 비선형**: 같은 joint delta라도 팔 위치에 따라 다른 visual motion 생성 (forward kinematics)
3. **Linear probe의 한계**: MLP probe를 사용하면 R²가 더 높을 가능성
4. **32-dim bottleneck**: latent dim이 29-dim action에 비해 크지만, VAE의 information bottleneck이 action 관련 정보를 모두 통과시키지 않을 수 있음

### Suggested Next Steps
1. **Frame skip 확대**: skip=1 대신 skip=4-8로 delta action을 크게 만들어 signal-to-noise ratio 향상
2. **MLP probe**: 1-2 hidden layer MLP로 비선형 regression 시도 → linear probe 한계 극복
3. **더 큰 latent dim**: 32 → 64 또는 128로 확장하여 information capacity 증가
4. **Action-conditioned generation**: latent action을 downstream VLA policy에 직접 통합하여 실제 task 성능 평가
5. **KL weight 조정**: β=1e-6은 매우 작아서 posterior가 거의 제약 없음 → β 키우면 더 structured latent이 될 수 있음
