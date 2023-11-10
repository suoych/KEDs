# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from typing import Tuple, Union

import os
import json
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pdb

def find_first_nonzero_indices(tensor):
    is_nonzero = tensor != 0
    very_large_value = torch.max(tensor) + 1
    tensor_with_large_value = torch.where(is_nonzero, tensor, very_large_value)
    first_nonzero_indices = torch.argmin(tensor_with_large_value, dim=1).tolist()
    return first_nonzero_indices

class CrossAttention(nn.Module):
    def __init__(self, q_dim,k_dim,v_dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == q_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(q_dim, inner_dim, bias = True)
        self.to_k = nn.Linear(k_dim, inner_dim , bias=True)
        self.to_v = nn.Linear(v_dim, inner_dim , bias = True)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, q_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        #self.self_attn = ResidualAttentionBlock(dim,heads)

    def forward(self, q, k, v):
        b, n, _, h = *k.shape, self.heads
        b_q, n_q, _, h = *q.shape, self.heads

        q = self.to_q(q)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        k = self.to_k(k)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)

        v = self.to_v(v)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)

        #out = self.self_attn(out)

        return out

class CrossFormer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, num_layers = 1, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self._num_layers = num_layers
        layer_list =[]
        for _ in range(self._num_layers):
            layer_list.append(CrossAttention(
                    q_dim,
                    k_dim,
                    v_dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                ))

        self.cross_layers = nn.ModuleList(layer_list)

    def forward(self, q, k, v):
        for layer in self.cross_layers:
            q = layer(q, k, v)
        return q



class IM2TEXT(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())            
            dim = middle_dim
            layers.append(nn.Sequential(*block))        
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)

class T2I(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())            
            dim = middle_dim
            layers.append(nn.Sequential(*block))        
        self.layers = nn.Sequential(*layers)
        #self.ln_final = LayerNorm(output_dim)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        x = self.fc_out(x)
        #x = self.ln_final(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor,cross_layers=None,kv_features=None,collect_ind=None,mid_feature=False,text_feature=None, img2text=None):
        if mid_feature:
            result_list = []
            for block in self.resblocks:
                x = block(x)
                result_list.append(x.permute(1,0,2))
            return x, result_list
        elif cross_layers is not None and kv_features is not None:
            # textual invert
            for i in range(len(self.resblocks)):
                if i != len(self.resblocks) - 1 and i<=5:
                    eos = x[collect_ind]
                    eos = cross_layers(eos.unsqueeze(1), kv_features)
                    x[collect_ind] = x[collect_ind] + eos.squeeze(1)
                    x = self.resblocks[i](x)
                else:
                    x = self.resblocks[i](x)
            return x
        elif text_feature is not None:
            # visual invert
            for i in range(len(self.resblocks)):
                if i == len(self.resblocks) - 6 :
                    #x = x.permute(1,0,2)
                    #x = torch.cat([x[:, 0].unsqueeze(1)+text_feature, x[:, 1:]+text_feature], dim=1)
                    #x = torch.cat([x[:, 0].unsqueeze(1), text_feature, x[:, 1:]], dim=1)
                    #x = x.permute(1,0,2)
                    
                    x = x.permute(1,0,2)
                    x = torch.cat([x[:, 0].unsqueeze(1) + img2text(x,text_feature).squeeze(1), x[:, 1:]], dim=1)
                    #x[:,0] = x[:,0] + img2text(text_feature,x).squeeze(1)
                    x = x.permute(1,0,2)
                    x = self.resblocks[i](x)
                else:
                    #with torch.no_grad():
                    x = self.resblocks[i](x)
            return x
        else:
            return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, attn_mask=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads,attn_mask=attn_mask)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor,mid_feature=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        if mid_feature:
            x,result_list = self.transformer(x,mid_feature=mid_feature)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_post(x[:, 0, :])
            if self.proj is not None:
                x = x @ self.proj
            return x, result_list
        else:
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_post(x[:, 0, :])
            if self.proj is not None:
                x = x @ self.proj
            return x
        

    def get_tokens(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 extra_transformer_layers: int = 0,
                 share_projection_layer: bool = True,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.share_projection_layer = share_projection_layer
        self.has_extra = True if extra_transformer_layers > 0 else False

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )
            #self.visual_mask = VisualTransformer(
            #    input_resolution=image_resolution,
            #    patch_size=vision_patch_size,
            #    width=vision_width,
            #    layers=vision_layers,
            #    heads=vision_heads,
            #    output_dim=embed_dim,
            #    attn_mask=self.build_visual_attention_mask()
            #)
        self.transformer_width = transformer_width
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        if extra_transformer_layers > 0:
            self.extra_transformer = Transformer(
                width=transformer_width,
                layers=extra_transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask()
            )
            self.extra_ln_final = LayerNorm(transformer_width)

        self.vocab_size = vocab_size
        self.end_id = self.vocab_size -1
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        if not share_projection_layer:
            self.extra_text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
        if hasattr(self, 'extra_text_projection'):
            nn.init.normal_(self.extra_text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_visual_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(257, 257)
        mask.fill_(float("-inf"))
        num_tokens_to_pick = 1 # we randomly keep some tokens 
        pick_positions = torch.randperm(256)[:num_tokens_to_pick]
        pick_positions = pick_positions + 1 # pick from all except cls
        
        for i in pick_positions:
            for j in range(257):
                mask[j][i] = 0
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image,mid_feature=False,mask_token=False):
        if mid_feature:
            return self.visual(image.type(self.dtype),mid_feature=mid_feature)
        elif mask_token:
            return self.visual(image.type(self.dtype))
        else:
            return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        collect_ind = text == self.end_id 
        collect_ind = collect_ind.nonzero()[:, 1]
        x = x[torch.arange(x.size(0)), collect_ind] @ self.text_projection
        return x
    
    def get_text_tokens(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        collect_ind = text == self.end_id 
        collect_ind = collect_ind.nonzero()[:, 1]
        #x = x[torch.arange(x.size(0)), collect_ind] @ self.text_projection
        return x, collect_ind
    
    def get_text_mid_cross_feature(self, text, img_tokens, cross_layers):
        b_size = img_tokens.size(0)
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        collect_ind = text == self.end_id 
        collect_ind = collect_ind.nonzero()[:, 1]
        #x = torch.cat([x[:, :collect_ind[0]], img_tokens, x[:, collect_ind[0]:-1]], dim=1)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, cross_layers,img_tokens,collect_ind[0])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)    
        #x = x[torch.arange(x.size(0)), collect_ind+1] @ self.text_projection
        x = x[torch.arange(x.size(0)), collect_ind] @ self.text_projection # we don't plus one cause we did not concat the img token
        #pdb.set_trace()
        return x
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def get_visual_composed_features(self, text_feature, images,img2text):
        """
        Map text eos token to visual space.
        """
        text_feature = text_feature.unsqueeze(1)
        #text_features, collect_ind = model.get_text_tokens(text)
        x = self.visual.conv1(images)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        
        # adopt random masking from MAE
        x_masked, mask, ids_restore = self.random_masking(x[:,1:,:], 1)
        #x = x[:,0,:].unsqueeze(1)
        x = torch.cat([x[:,0,:].unsqueeze(1), x_masked], dim=1)

        #pdb.set_trace()
        #x = torch.cat([x[:, 0].unsqueeze(1), text_feature, x[:, 1:].mean(dim=1, keepdim=True)], dim=1) # average pooling
        #x = torch.cat([x[:, 0].unsqueeze(1), text_feature, x[:, 1:]], dim=1)
        #x = torch.cat([x[:, 0].unsqueeze(1), text_feature], dim=1)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual.transformer(x,text_feature=text_feature,img2text=img2text)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.visual.ln_post(x[:, 0, :])

        if self.visual.proj is not None:
            x = x @ self.visual.proj
        return x
    
    def get_visual_composed_features_eval(self, text_feature, images):
        """
        Map text eos token to visual space.
        """
        text_feature = text_feature.unsqueeze(1)
        #text_features, collect_ind = model.get_text_tokens(text)
        x = self.visual.conv1(images)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        x_ori = x 
        
        #pdb.set_trace()
        #x = torch.cat([x[:, 0].unsqueeze(1), text_feature, x[:, 1:]], dim=1)
        #x = torch.cat([x[:, 0].unsqueeze(1), text_feature], dim=1)
        #x = x[:, 0].unsqueeze(1)
        x_masked, mask, ids_restore = self.random_masking(x[:,1:,:], 1)

        #x = x[:,0,:].unsqueeze(1)
        x = torch.cat([x[:,0,:].unsqueeze(1), x_masked], dim=1)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual.transformer(x,text_feature=None)#text_feature)
        #x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.visual.ln_post(x[:, 0, :])

        if self.visual.proj is not None:
            x = x @ self.visual.proj

        t_masked, mask, ids_restore = self.random_masking(x_ori[:,1:,:], 1)

        #x = x[:,0,:].unsqueeze(1)
        t = torch.cat([x_ori[:,0,:].unsqueeze(1), t_masked], dim=1)

        #pdb.set_trace()

        t = t.permute(1, 0, 2)  # NLD -> LND
        t = self.visual.transformer(t,text_feature=text_feature)
        #x = self.visual.transformer(x)
        t = t.permute(1, 0, 2)  # LND -> NLD

        t = self.visual.ln_post(t[:, 0, :])

        if self.visual.proj is not None:
            t = t @ self.visual.proj

        return t #0.075 * x + 0.925 * t
    
    """
    def encode_text_img_cross_attn(self, text, img_tokens):
        b_size = img_tokens.size(0)
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        collect_ind = text == self.end_id 
        collect_ind = collect_ind.nonzero()[:, 1]
        img_tokens = img_tokens.view(b_size, 1, -1)
        x = torch.cat([x[:, :collect_ind[0]], img_tokens, x[:, collect_ind[0]:-1]], dim=1)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = self.cross(torch.cat([x,img_tokens],dim=1))
        
        x = x[torch.arange(x.size(0)), collect_ind+1] @ self.text_projection
        return x  
    """
        
    def encode_text_img(self, text, img_tokens):
        b_size = img_tokens.size(0)
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        collect_ind = text == self.end_id 
        collect_ind = collect_ind.nonzero()[:, 1]
        #pdb.set_trace()
        #img_tokens = img_tokens.view(b_size, 1, -1)
        #x = torch.cat([x[:, :collect_ind[0]], img_tokens, x[:, collect_ind[0]:-1]], dim=1)
        #img_tokens = img_tokens.view(b_size, 3, -1)
        
        #x = torch.cat([x[:, :collect_ind[0]], img_tokens, x[:, collect_ind[0]:-3]], dim=1)
        x = torch.cat([x[:, :collect_ind[0]], img_tokens, x[:, collect_ind[0]:-2]], dim=1)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)    
        #x = x[torch.arange(x.size(0)), collect_ind+1] @ self.text_projection
        x = x[torch.arange(x.size(0)), collect_ind+2] @ self.text_projection
        return x              
    
    def encode_text_img_vis(self, text, img_tokens, split_ind=4):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        collect_ind = text == self.end_id 
        collect_ind = collect_ind.nonzero()[:, 1]
        new_x = []
        for i, sample in enumerate(x):
            ind_insert = text[i] == split_ind
            sample = sample.view(1, x.size(1), -1)            
            if isinstance(img_tokens, tuple):
                indexes = ind_insert.nonzero()
                for i, index in enumerate(indexes):
                    img = img_tokens[i].view(1, 1, -1)
                    sample = torch.cat([sample[:, :index], img, sample[:, index+1:]], dim=1)
            else:
                img_tokens = img_tokens.view(1, 1, -1)
                ind_insert = ind_insert.nonzero()[0]
                sample = torch.cat([sample[:, :ind_insert], img_tokens, sample[:, ind_insert+1:]], dim=1)                
            new_x.append(sample)
        x = torch.cat(new_x, dim=0)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)    
        x = x[torch.arange(x.size(0)), collect_ind] @ self.text_projection
        return x

    def encode_text_img_retrieval(self, text, img_tokens, split_ind=4, repeat=True):
        # text.shape = [1, n_ctx]
        # img_tokens.shape = [batch_size, d_model]        
        if isinstance(img_tokens, tuple):
            b_size = img_tokens[0].shape[0]
        else:
            b_size = img_tokens.shape[0]
        if repeat:            
            text = text.repeat(b_size, 1)
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        collect_ind = text == self.end_id 
        collect_ind = collect_ind.nonzero()[:, 1]
        ind_insert = text[0] == split_ind   
        if isinstance(img_tokens, tuple):
            indexes = ind_insert.nonzero()
            for i, index in enumerate(indexes):
                #img = img_tokens[i].view(b_size, 1, -1)
                x = torch.cat([x[:, :index], img_tokens[i], x[:, index+1:]], dim=1)
        else:
            #img_tokens = img_tokens.view(b_size, 1, -1)
            ind_insert = ind_insert.nonzero()[0]
            #x = torch.cat([x[:, :ind_insert], img_tokens, x[:, ind_insert+1:]], dim=1)
            #x = torch.cat([x[:, :ind_insert], img_tokens, x[:, ind_insert+1:-3]], dim=1)
            if img_tokens.shape[1] == 2:
                x = torch.cat([x[:, :ind_insert], img_tokens, x[:, ind_insert+1:-1]], dim=1)
            else:
                x = torch.cat([x[:, :ind_insert], img_tokens, x[:, ind_insert+1:-2]], dim=1) # 3tokens
            #x = torch.cat([x[:, :ind_insert], img_tokens, x[:, ind_insert+1:-1]], dim=1)
        #x = torch.cat([x, torch.zeros_like(x).cuda()[:, :1, :]], dim=1)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)    
        #x = x[torch.arange(x.size(0)), collect_ind] @ self.text_projection
        #x = x[torch.arange(x.size(0)), collect_ind+3] @ self.text_projection
        if img_tokens.shape[1] == 2:
            x = x[torch.arange(x.size(0)), collect_ind+1] @ self.text_projection
        else:
            x = x[torch.arange(x.size(0)), collect_ind+2] @ self.text_projection
        #x = x[torch.arange(x.size(0)), collect_ind+1] @ self.text_projection
        return x
   
    def encode_text_img_train(self, text, img_tokens, split_ind=4, repeat=True):
        # text.shape = [1, n_ctx]
        # img_tokens.shape = [batch_size, d_model]        
        b_size = img_tokens.shape[0]
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        #pdb.set_trace()
        collect_ind = text == self.end_id 
        collect_ind = collect_ind.nonzero()[:, 1]
        #ind_insert = text[0] == split_ind   

        #flexible insert index
        #ind_insert = text == split_ind 
        ind_insert = text[0] == split_ind 
        ind_insert = ind_insert.nonzero()[0]
        #ind_insert = ind_insert.nonzero()

        #ind_insert = find_first_nonzero_indices(ind_insert)
        #ind_insert = torch.cat(ind_insert)
        #pdb.set_trace()

        #for i in range(x.shape[0]):
        #    x[i,ind_insert[i]:ind_insert[i]+3,:] = img_tokens[i]
        #pdb.set_trace()
        
        #img_tokens = img_tokens.view(b_size, 1, -1)
        #ind_insert = ind_insert.nonzero()[0]

        #x = torch.cat([x[:, :ind_insert], img_tokens, x[:, ind_insert+2:]], dim=1) # only one token
        x = torch.cat([x[:, :ind_insert], img_tokens, x[:, ind_insert+3:]], dim=1)
        #x = torch.cat([x[:, :ind_insert], img_tokens, x[:, ind_insert+1:-2]], dim=1)
        #x = torch.cat([x, torch.zeros_like(x).cuda()[:, :1, :]], dim=1)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)    
        x = x[torch.arange(x.size(0)), collect_ind] @ self.text_projection
        return x
   
    def forward(self, image, text, extra=False):
        if image is None:
            if extra:
                return self.encode_text_extra(text)
            else:
                return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)
        image_features = self.encode_image(image)
        if extra:
            text_features = self.encode_text_extra(text)
        else:
            text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return image_features, text_features, self.logit_scale.exp()


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict) #,strict=False) # manually load weights for masked visual transformers
    #with torch.no_grad():
    #    for a_param, b_param in zip(model.visual.parameters(), model.visual_mask.parameters()):
    #        b_param.copy_(a_param)
    return model.eval()