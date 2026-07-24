

from urllib.request import urlopen
from PIL import Image
import timm
import torch


model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
model = model.eval()
print(model)