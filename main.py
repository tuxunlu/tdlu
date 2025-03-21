import torch

img_encoder = torch.load("/Users/nicklu/Documents/Git/tdlu/mgh_mammo_MIRAI_Base_May20_2019.p",
                         map_location='cpu', weights_only=False)
