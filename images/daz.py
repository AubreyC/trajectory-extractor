# -*- coding: utf-8 -*-

# Wrap angle into -pi, pi
def daz(x):
    pi = np.pi
    x = x - np.floor(x/(2*pi)) *2 *pi
    if x >= pi:
        x = x- 2*pi
    return x
