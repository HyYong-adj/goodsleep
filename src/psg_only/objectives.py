"""클래스 불균형용 대체 목적함수.

현재 기본 손실은 inverse-sqrt weighted CE 다. 여기에 REM 가중치를 더 키우는 시도는
이미 두 번 기각됐다(Phase 1 A4, choihy-e010): REM recall 은 0.247 -> 0.335 로 올랐으나
precision 이 떨어져 Macro-F1 은 하락했다. **가중치 조정은 결정 경계를 옮길 뿐
표현의 분리도를 바꾸지 못한다.**

여기 두 목적함수는 서로 다른 지점을 건드린다.
  - focal    : 쉬운 샘플(대부분 Light)의 기여를 줄여 어려운 소수 클래스에 집중
  - supcon   : 같은 단계 epoch 끼리 embedding 을 모으고 다른 단계는 밀어냄 -> 분리도 자체
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor,
               gamma: float = 2.0, label_smoothing: float = 0.0) -> torch.Tensor:
    """Class-weighted focal loss. gamma=0 이면 weighted CE 와 동일하다.

    (1-p_t)^gamma 로 이미 잘 맞히는 샘플의 기여를 줄인다. 본 데이터는 82%가 Light 라
    CE 의 대부분이 이미 쉬운 Light 에서 나온다.
    """
    mask = targets != -100
    if not mask.any():
        return logits.sum() * 0.0
    logits, targets = logits[mask], targets[mask]
    log_probability = F.log_softmax(logits, dim=-1)
    if label_smoothing > 0:
        classes = logits.shape[-1]
        true = F.one_hot(targets, classes).float()
        smooth = true * (1 - label_smoothing) + label_smoothing / classes
        per_sample_ce = -(smooth * log_probability).sum(-1)
    else:
        per_sample_ce = F.nll_loss(log_probability, targets, reduction="none")
    probability = log_probability.gather(1, targets.unsqueeze(1)).squeeze(1).exp()
    sample_weight = weight[targets]
    loss = sample_weight * (1.0 - probability).pow(gamma) * per_sample_ce
    # weighted CE 와 같은 정규화(가중치 합으로 나눔)를 써서 loss scale 을 비교 가능하게 유지한다.
    return loss.sum() / sample_weight.sum().clamp_min(1e-8)


def supcon_loss(embeddings: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None = None,
                temperature: float = 0.1) -> tuple[torch.Tensor, dict]:
    """Supervised contrastive (InfoNCE 계열) 손실.

    같은 수면 단계 epoch 을 positive 로 묶는다. anchor 별 손실에 클래스 가중치를 곱해
    희소 클래스 anchor 가 82% 를 차지하는 Light 에 묻히지 않게 한다.

    positive 가 없는 anchor(배치 안에 같은 클래스가 자기뿐)는 건너뛴다. REM 은 4.9% 라
    배치가 작으면 positive 를 못 찾는 일이 잦으므로, 실제로 몇 개가 쓰였는지 함께 돌려준다.
    """
    mask = targets != -100
    stats = {"anchors": 0, "used": 0, "rem_anchors": 0}
    if mask.sum() < 2:
        return embeddings.sum() * 0.0, stats
    z = F.normalize(embeddings[mask].float(), dim=-1)
    y = targets[mask]
    count = z.shape[0]
    stats["anchors"] = int(count)
    stats["rem_anchors"] = int((y == 1).sum())  # canonical index 1 = REM

    similarity = (z @ z.T) / temperature
    self_mask = torch.eye(count, dtype=torch.bool, device=z.device)
    similarity = similarity.masked_fill(self_mask, float("-inf"))
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)

    positive = (y.unsqueeze(0) == y.unsqueeze(1)) & ~self_mask
    positive_count = positive.sum(1)
    usable = positive_count > 0
    stats["used"] = int(usable.sum())
    if not usable.any():
        return embeddings.sum() * 0.0, stats

    # log_probability 의 대각선은 -inf 다(self 제외). bool 로 곱하면 -inf * 0 = NaN 이 되므로
    # 곱하지 말고 non-positive 자리를 0 으로 덮은 뒤 더한다.
    positive_logprob = log_probability.masked_fill(~positive, 0.0)
    per_anchor = -positive_logprob.sum(1)[usable] / positive_count[usable]
    if weight is not None:
        anchor_weight = weight[y][usable]
        return (per_anchor * anchor_weight).sum() / anchor_weight.sum().clamp_min(1e-8), stats
    return per_anchor.mean(), stats
