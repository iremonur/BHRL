import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
import torch


# Load the configuration file
config_file = '/home/ionur2/Desktop/MSc_THESIS/BHRL/configs/vot/BHRL.py'
cfg = mmcv.Config.fromfile(config_file)

# Build the model
model = build_detector(cfg.model, train_cfg=None, test_cfg=None)

# Load the weights (optional)
checkpoint_file = '/home/ionur2/Desktop/MSc_THESIS/BHRL/checkpoints/model_split3.pth'

print(model)

x = torch.randn(1, 8)
y = model(x)

make_dot(y.mean(), params=dict(model.named_parameters()))