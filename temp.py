# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 13:48:55 2018

@author: Christian
"""

import numpy as np

z = np.arange(0, 6, 1)
ans = np.sqrt(z[0:3]**2 + z[3:]**2)
print(ans)