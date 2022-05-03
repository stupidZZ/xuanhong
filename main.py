from dataclasses import dataclass
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
from torch import nn
import torch.optim as optim

from swin_v2 import SwinTransformerV2
from swin_v2_seq_atten import SwinTransformerV2SeqAtten
pretrain_model = "/data/home/zhez/models/swin_v2.pth"

def train_one_step(model, opt, data):
        opt.zero_grad()
        out = model(data)
        model_loss = (out ** 2).sum()
        model_loss.backward()
        opt.step()

        return out

model = SwinTransformerV2(
        patch_size=4,
        in_chans=3,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        use_checkpoint=True,
        relative_coords_table_type='norm8_log_192to224',
        checkpoint_blocks=[255,0,255,0] # 255: Apply checkpoint for all blocks of the stage; 0: Do not apply checkpoint for the stage.
)
model.init_weights(pretrain_model) # load pre-train model
model.cuda()


model_seq = SwinTransformerV2SeqAtten(
        patch_size=4,
        in_chans=3,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        use_checkpoint=True,
        relative_coords_table_type='norm8_log_192to224',
        checkpoint_blocks=[255,255,255,255],
        head_chunk_size=[4, 8, 16, 32], 
)
model_seq.init_weights(pretrain_model)
model_seq.cuda()


model_opt = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
model_seq_opt = optim.SGD(model_seq.parameters(), lr=1e-5, momentum=0.9)

mock_data = torch.rand((16, 3, 224, 224)).cuda()
out = train_one_step(model, model_opt, mock_data)
seq_out = train_one_step(model_seq, model_seq_opt, mock_data)

diff = (seq_out - out).abs().max()
assert diff > 1e-5


