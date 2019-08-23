import numpy as np
from astropy.io import fits
from numpy.fft import fft2, fftshift
import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt


# swaps 2,4 and 1,3 quadrants of a square matix
# ==============================================
def transform(a):
    quar = a.shape[0]//2

    for i in range(quar):
        for j in range(quar):
            a[i][j],a[i+quar][j+quar] =  a[i+quar][j+quar],a[i][j]
            
    for i in range(quar):
        for j in range(quar):
            a[i+quar][j], a[i][j+quar] = a[i][j+quar],a[i+quar][j]

    return a
# ==============================================

size = 1000
radius = 50

y,x = np.ogrid[-size: size+1, -size: size+1]
mask = x**2+y**2 <= radius**2
mask = 254*mask.astype(int)

fits.writeto("image.fits",mask,overwrite=True)

fft = np.abs(fft2(fftshift(mask)))
fft = transform(fft)

fits.writeto("fft.fits",fft,overwrite=True)

plt.plot(fft[fft.shape[0]//2])
plt.show()