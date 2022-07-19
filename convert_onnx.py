from dataclasses import dataclass
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
from torch import nn
import torch.optim as optim
import torch.onnx
import torch.utils.model_zoo as model_zoo

import torch.nn as nn
import torch.nn.init as init

from swin_v2_onnx import SwinTransformerV2

pretrain_model = "/data/home/zhez/models/swin_v2.pth"


model = SwinTransformerV2(
        patch_size=4,
        in_chans=3,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        # depths=[1,1,1,1],
        # num_heads=[1, 8, 16, 32],
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        relative_coords_table_type='norm8_log_192to224',
)
# model.init_weights(pretrain_model) # load pre-train model
# model.cuda()


# set the model to inference mode
model.eval()

# Input to the model
mock_data = torch.rand((16, 3, 224, 224))

# Export the model
torch.onnx.export(model,               # model being run
                  mock_data,                         # model input (or a tuple for multiple inputs)
                  "swinv2.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=14,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
model.cuda()

mock_data_2 = torch.rand((1, 3, 224, 224))
torch_out = model(mock_data_2.cuda())
import onnx

onnx_model = onnx.load("swinv2.onnx")
onnx.checker.check_model(onnx_model)

import onnxruntime
import numpy as np

ort_session = onnxruntime.InferenceSession("swinv2.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(mock_data_2)}
# ort_inputs = {ort_session.get_inputs()[0].name: mock_data}

ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
