import torch
import torch.nn as nn
import torch.nn.functional as F
from tst.encoder import Encoder
from tst.decoder import Decoder

import math


class Transformer(nn.Module):
    def __init__(self,
                 d_input: int,
                 d_output: int,
                 d_channel: int,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 dropout: float = 0.2,
                 pe: bool = False,
                 mask: bool = False):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_input = d_input
        self._d_channel = d_channel
        self._d_model = d_model
        self._pe = pe

        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      dropout=dropout,
                                                      mask=mask) for _ in range(N)])

        # self.layers_decoding = nn.ModuleList([Decoder(d_model,
        #                                               q,
        #                                               v,
        #                                               h,
        #                                               dropout=dropout) for _ in range(N)])

        self._embedding_input = nn.Linear(self._d_channel, d_model) #seqLEN
        self._embedding_channel = nn.Linear(self._d_input, d_model) #feature

        self._gate = nn.Linear(d_model * d_channel+d_model * d_input, 2)
        self._dropout = nn.Dropout(p=dropout)
        # self._linear_input = nn.Linear(d_model * d_input, d_output)#seqLEN
        self._linear = nn.Linear(d_model * d_channel+d_model * d_input, d_output)#feature+seqLen
        # self._linear = nn.Linear(d_model + d_model, d_output)#feature

        # self._linear = nn.Linear(d_model, d_output)#一维feature


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x = x.expand(x.shape[0], x.shape[1], self._d_channel)

        # x = x.transpose(-1,-2) #单维数据集转换

        encoding_input = self._embedding_input(x)

        # 位置编码
        if self._pe:
            pe = torch.ones_like(encoding_input[0])
            position = torch.arange(0, self._d_input).unsqueeze(-1)
            # position = torch.arange(0, self._d_input).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000)/self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)

            encoding_input = encoding_input + pe

        # Encoding stack
        for layer in self.layers_encoding:
            encoding_input = layer(encoding_input)





        encoding_channel = self._embedding_channel(x.transpose(-1,-2))

        # 位置编码
        # if self._pe:
        #     pe = torch.ones_like(encoding_channel[0])
        #     position = torch.arange(0, self._d_channel).unsqueeze(-1)
        #     # position = torch.arange(0, self._d_input).unsqueeze(-1)
        #     temp = torch.Tensor(range(0, self._d_model, 2))
        #     temp = temp * -(math.log(10000) / self._d_model)
        #     temp = torch.exp(temp).unsqueeze(0)
        #     temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
        #     pe[:, 0::2] = torch.sin(temp)
        #     pe[:, 1::2] = torch.cos(temp)
        #
        #     encoding_channel = encoding_channel + pe

        # Encoding stack
        for layer in self.layers_encoding:
            encoding_channel = layer(encoding_channel)


        # 三维变两维
        encoding_input = encoding_input.reshape(encoding_input.shape[0], -1)
        encoding_channel = encoding_channel.reshape(encoding_channel.shape[0], -1)

        # encoding_input = encoding_input.mean(dim=1)
        # encoding_channel = encoding_channel.mean(dim=1)
        # output = encoding.mean(dim=1)

        gate = F.softmax(self._gate(torch.cat((encoding_channel, encoding_input), dim=1)), dim=1)

        encoding = torch.cat((encoding_channel * gate[:, 0:1], encoding_input * gate[:, 1:2]), dim=1)

        # d_model -> output
        encoding = self._dropout(encoding)
        output = self._linear(encoding)

        return output
