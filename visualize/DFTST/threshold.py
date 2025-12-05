# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:17:00 2024

@author: alyadham
"""

import torch

def soft_threshold(x, T):
    return torch.copysign(torch.abs(x)-T, x)
    