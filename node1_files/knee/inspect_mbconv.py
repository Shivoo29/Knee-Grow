import torch
import timm
import inspect
from timm.models.maxxvit import MbConvBlock

print("MbConvBlock.forward source:")
print(inspect.getsource(MbConvBlock.forward))
