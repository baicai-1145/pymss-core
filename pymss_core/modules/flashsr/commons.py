# Adapted via FlashSR; see NOTICE.txt and incl_licenses/ for source and licenses.
# Changes: relocated the helpers into pymss_core.modules.flashsr.

import math
import torch
from torch.nn import functional as F


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)
