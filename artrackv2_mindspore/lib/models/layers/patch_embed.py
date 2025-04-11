import sys
sys.path.append("/home/baiyifan/code/AR2_mindspore_cp/2stage")
import mindspore.nn as nn
from mindspore import ops
from lib.models.timm import to_2tuple


class PatchEmbed(nn.Cell):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,pad_mode='valid',has_bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def construct(self, x):
        # allow different input size
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = ops.flatten(x,start_dim=2)
            x = x.swapaxes(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
