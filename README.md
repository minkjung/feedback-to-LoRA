# Feedback-to-LoRA

자연어 교정 피드백을 hypernetwork로 직접 LoRA weight delta로 변환하는 시스템.
scalar reward 없이 텍스트가 곧 gradient가 되는 구조.

```
유저: "회사 우편번호 뭐야?"
모델: "12345입니다"
유저: "아니잖아 54321이잖아"
→ hypernetwork forward 1회 (~0.1초) → LoRA 생성 → 모델에 merge
→ 이후 같은 질문 + 관련 질문에 교정 반영됨
```

## 구성

| 역할 | 모델 |
|------|------|
| Target model | Gemma 4 E4B (frozen) |
| Teacher | Gemma 4 E4B (frozen, feedback 포함 prompt) |
| Hypernetwork backbone | Gemma 4 E2B (fine-tune) |
| LoRA projection head | MLP (~50M, scratch) |

## 설치

```bash
pip install -e .
```

## 실행 순서

```bash
# Step 0: E4B SimpleQA Verified 오답률 측정 (1~2시간)
python scripts/step0_probe.py

# Step 1: 합성 데이터 생성 (GPT-5 Nano, ~$15)
python scripts/step1_synth.py

# Step 2: hypernetwork 학습 — Gemma E2B backbone (A100, 3~4일)
python scripts/step2_train.py

# Step 3: 평가 (Correction / Generalization / ICL / Doc-to-LoRA baseline)
python scripts/step3_eval.py

# Step 4: ablation — Perceiver from scratch 재학습 + 평가 (2~3일)
python scripts/step4_ablation.py
```

## 디렉토리

```
feedback-to-lora/
├── configs/config.yaml
├── src/                # core modules
├── scripts/            # step0~4
├── data/{raw,processed,splits}/
└── outputs/{checkpoints,results}/
```

## 평가 지표

1. **Correction Accuracy** — 교정 후 같은 질문 정답률
2. **Generalization Rate** — 교정 후 관련 질문 정답률

## Baseline

- ICL: feedback을 prompt에 넣음 (upper bound)
- Feedback-to-LoRA (ours)
- Doc-to-LoRA: feedback을 document로 취급

## 참고

- Doc-to-LoRA: https://github.com/SakanaAI/doc-to-lora
- Text-to-LoRA: https://github.com/SakanaAI/text-to-lora
- SimpleQA Verified: https://www.kaggle.com/benchmarks/deepmind/simpleqa-verified
- Gemma 4 model card: https://ai.google.dev/gemma/docs/core/model_card_4
