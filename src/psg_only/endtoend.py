"""Phase 3: Conformer epoch encoder + BiLSTM 을 한 번에 학습한다.

동결 2단계(B0 학습 -> freeze -> embedding 캐시 -> B1 학습)에서는 encoder 가
"고립된 30초를 분류하는 데 유용한 특징"만 배우고 멈춘다. 문맥 속에서만 의미 있는
특징은 배울 수 없다. 참조 트랙의 temporal 이득(+0.127)이 본 트랙(+0.070)보다 큰 것이
이 병목과 일관된다.

여기서는 같은 두 구성요소를 공동 최적화한다. 구조는 그대로 두고 **학습 방식만** 바꾼다.
"""
from __future__ import annotations

import torch
from torch import nn

from .conformer import ConformerEpochModel
from .windows import context_options


class EndToEndModel(nn.Module):
    """[B, S, 1, 48, 1499] -> 중앙 target_epochs 개의 4-class logits."""

    def __init__(self, layers=2, num_heads=4, ff_dim=768, conv_kernel_size=31, dropout=0.2,
                 hidden_dim=128, lstm_layers=2, input_epochs=40, target_epochs=20,
                 post_transformer_layers=0, post_transformer_heads=4, post_transformer_ff=1024,
                 in_channels=1):
        super().__init__()
        context_options(dict(input_epochs=input_epochs, target_epochs=target_epochs))
        self.input_epochs, self.target_epochs = input_epochs, target_epochs
        self.left_context = (input_epochs - target_epochs) // 2
        self.in_channels = int(in_channels)
        self.encoder = ConformerEpochModel(layers, num_heads, ff_dim, conv_kernel_size, dropout, in_channels=self.in_channels)
        self.norm = nn.LayerNorm(192)
        self.lstm = nn.LSTM(192, hidden_dim, num_layers=lstm_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0)
        # BiLSTM 뒤에 쌓는 Transformer encoder (0 이면 없음 = 기존 parent).
        # BiLSTM 은 지역적/순차적 문맥을, Transformer 는 창 전체의 전역 관계를 맡는다.
        # choihy-e005 에서 BiLSTM 을 Transformer 로 **교체**한 것은 실패했으나(-0.027),
        # 적층은 다른 구조이므로 별도 실험으로 다룬다.
        width = hidden_dim * 2
        self.post_transformer = None
        if post_transformer_layers > 0:
            if width % post_transformer_heads:
                raise ValueError("post_transformer_heads must divide 2*hidden_dim.")
            # BiLSTM 출력에 이미 순서 정보가 있으나, attention 은 순서를 못 보므로 위치 임베딩을 준다.
            self.post_position = nn.Parameter(torch.zeros(input_epochs, width))
            nn.init.normal_(self.post_position, std=0.02)
            self.post_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(width, post_transformer_heads, post_transformer_ff,
                                           dropout=dropout, batch_first=True, norm_first=True),
                num_layers=post_transformer_layers, norm=nn.LayerNorm(width))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 4))

    def forward(self, mel, input_valid=None, return_embedding=False):
        if mel.ndim != 5 or mel.shape[1] != self.input_epochs or mel.shape[2:] != (self.in_channels, 48, 1499):
            raise ValueError(f"Expected [B,{self.input_epochs},{self.in_channels},48,1499], got {tuple(mel.shape)}.")
        batch, sequence = mel.shape[:2]
        # 시퀀스를 batch 축으로 펼쳐 encoder 를 한 번에 통과시킨다.
        embedding = self.encoder.encode(mel.reshape(batch * sequence, self.in_channels, 48, 1499))
        embedding = embedding.reshape(batch, sequence, 192)
        if input_valid is not None:
            embedding = embedding.masked_fill(~input_valid.unsqueeze(-1), 0.0)
        output, _ = self.lstm(self.norm(embedding))
        if self.post_transformer is not None:
            output = output + self.post_position.unsqueeze(0)
            # 패딩 위치는 attention 에서 제외한다. 빼면 야간 경계의 0 벡터가 전역 표현을 오염시킨다.
            padding_mask = None if input_valid is None else ~input_valid
            output = self.post_transformer(output, src_key_padding_mask=padding_mask)
        logits = self.head(output[:, self.left_context:self.left_context + self.target_epochs])
        if return_embedding:
            # SupCon 은 시간 모델을 거치기 전 epoch 표현에 건다. 시간 문맥으로 정답을
            # 맞히는 우회를 막고 encoder 자체가 분리도를 갖게 하려는 것이다.
            target_embedding = embedding[:, self.left_context:self.left_context + self.target_epochs]
            return logits, target_embedding
        return logits


def endtoend_from_config(config):
    model = config["model"]
    required = {"type", "embedding_dim", "layers", "num_heads", "ff_dim", "conv_kernel_size",
                "dropout", "hidden_dim", "lstm_layers", "post_transformer_layers",
                "post_transformer_heads", "post_transformer_ff", "in_channels"}
    if set(model) != required or model["type"] != "endtoend_conformer_bilstm" or model["embedding_dim"] != 192:
        raise ValueError("Unknown/missing end-to-end model options or embedding dimension.")
    for key in ("layers", "num_heads", "ff_dim", "conv_kernel_size", "hidden_dim", "lstm_layers",
                "post_transformer_layers", "post_transformer_heads", "post_transformer_ff", "in_channels"):
        if type(model[key]) is not int:
            raise ValueError("End-to-end dimensions must be integers.")
    data = context_options(config["data"])
    return EndToEndModel(
        layers=model["layers"], num_heads=model["num_heads"], ff_dim=model["ff_dim"],
        conv_kernel_size=model["conv_kernel_size"], dropout=float(model["dropout"]),
        hidden_dim=model["hidden_dim"], lstm_layers=model["lstm_layers"],
        input_epochs=data["input_epochs"], target_epochs=data["target_epochs"],
        post_transformer_layers=model["post_transformer_layers"],
        post_transformer_heads=model["post_transformer_heads"],
        post_transformer_ff=model["post_transformer_ff"],
        in_channels=model["in_channels"],
    )
