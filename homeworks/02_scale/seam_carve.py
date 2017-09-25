# coding: utf-8
import numpy as np
import cv2


def energy(img):
    brightness = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0].astype(np.float32)
    derivative_x = np.vstack((brightness[1:2] - brightness[:1],
                              brightness[2:] - brightness[:-2],
                              brightness[-1:] - brightness[-2:-1]))
    derivative_y = np.hstack((brightness[:,1:2] - brightness[:,:1],
                              brightness[:,2:] - brightness[:,:-2],
                              brightness[:,-1:] - brightness[:,-2:-1]))
    return np.sqrt(derivative_x ** 2 + derivative_y ** 2)


# In[120]:


def seam_carve(img, mode="horizontal shrink", mask=None):
    direction, action = mode.split()
    if direction == "vertical":
        img = img.transpose(1, 0, 2)
        if mask is not None:
            mask = mask.transpose(1, 0)
    
    cum = energy(img)
    if mask is not None:
        delta = mask.shape[0] * mask.shape[1] * 256
        cum[mask == 1] += delta
        cum[mask == -1] -= delta
        mask = list(mask)
    for i in range(1, cum.shape[0]):
        for j in range(1, cum.shape[1] - 1):
            cum[i, j] += np.min(cum[i - 1, j - 1: j + 2])
        cum[i, 0] += np.min(cum[i - 1, :2])
        cum[i, -1] += np.min(cum[i - 1, -2:])
    
    # Build and carve seam simultaneously
    seam = []
    seam_y = np.argmin(cum[-1])
    img = list(img)
    if action == "shrink":
        img[-1] = np.concatenate((img[-1][:seam_y], img[-1][seam_y + 1:]), axis=0)
        if mask is not None:
            mask[-1] = np.concatenate((mask[-1][:seam_y], mask[-1][seam_y + 1:]), axis=0)
    else:
        img[-1] = np.concatenate((img[-1][:seam_y + 1],
                                  img[-1][seam_y: seam_y + 2].mean(axis=0).reshape(1, -1).astype(img[-1].dtype),
                                  img[-1][seam_y + 1:]),
                                 axis=0)
        if mask is not None:
            mask[-1] = np.concatenate((mask[-1][:seam_y + 1],
                                       mask[-1][seam_y: seam_y + 2].mean(axis=0).reshape(1,).astype(mask[-1].dtype),
                                       mask[-1][seam_y + 1:]),
                                      axis=0)
    seam.append(np.zeros_like(img[-1][:,0], dtype=np.uint8))
    seam[-1][min(seam_y, seam[-1].shape[0] - 1)] = 1
        
    for i in range(cum.shape[0] - 2, -1, -1):
        seam_y = np.argmin(cum[i, max(0, seam_y - 1): min(cum.shape[1], seam_y + 2)]) + max(0, seam_y - 1)
        if action == "shrink":
            img[i] = np.concatenate((img[i][:seam_y],
                                     img[i][seam_y + 1:]),
                                    axis=0)
            if mask is not None:
                mask[i] = np.concatenate((mask[i][:seam_y], mask[i][seam_y + 1:]), axis=0)
        else:
            img[i] = np.concatenate((img[i][:seam_y + 1],
                                     img[i][seam_y: seam_y + 2].mean(axis=0).reshape(1, -1).astype(img[-1].dtype),
                                     img[i][seam_y + 1:]),
                                    axis=0)
            if mask is not None:
                mask[i] = np.concatenate((mask[i][:seam_y + 1],
                                          mask[i][seam_y: seam_y + 2].mean(axis=0).reshape(1, ).astype(mask[i].dtype),
                                          mask[i][seam_y + 1:]),
                                         axis=0)
        seam.append(np.zeros_like(img[i][:,0], dtype=np.uint8))
        seam[-1][min(seam_y, seam[-1].shape[0] - 1)] = 1

    img = np.array(img)
    if mask is not None:
        mask = np.array(mask)
    seam = np.array(seam[::-1])
        
    if direction == "vertical":
        img = img.transpose(1, 0, 2)
        if mask is not None:
            mask = mask.transpose(1, 0)
        seam = seam.transpose(1, 0)

    return img, mask, seam
