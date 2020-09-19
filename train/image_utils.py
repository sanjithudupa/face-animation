import numpy as np

def applyMask(b, m):
    h = b.shape[0]
    w = b.shape[1]

    rgb = np.zeros((h, w, 4), np.uint8)

    for y in range(0, h):
        for x in range(0, w):
            if(m[y, x][0] == 255):
                rgb[y, x][0] = b[y, x][0]
                rgb[y, x][1] = b[y, x][1]
                rgb[y, x][2] = b[y, x][2]
                rgb[y, x][3] = 255
            else:
                rgb[y, x][3] = 0
    
    return rgb

def overlay(b, o):
    h = o.shape[0]
    w = o.shape[1]

    for y in range(h):
        for x in range(w):
            if(o[y, x][3] > 252 and not(o[y, x][0] > 250 or o[y, x][1] > 250 or o[y, x][2] > 250)):
                b[y, x] = o[y, x]