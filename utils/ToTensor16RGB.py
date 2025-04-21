import numpy as np
import torch
from PIL import Image

class ToTensor16RGB:
    """Convert a PIL I;16 image to a 3×H×W float32 tensor in [0,1]."""
    def __call__(self, pic: Image.Image) -> torch.Tensor:
        # pic.mode == "I;16"
        # torch.set_printoptions(precision=32)
        arr16 = np.array(pic, dtype=np.uint16)          # H×W, 0…65535
        rgb16 = np.stack([arr16, arr16, arr16], axis=2) # H×W×3
        t = torch.from_numpy(rgb16).float()             # torch.uint16→float32
        t = t.permute(2,0,1) / 65535.0                  # 3×H×W in [0,1]
        return t