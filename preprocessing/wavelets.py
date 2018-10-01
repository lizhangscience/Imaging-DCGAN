import numpy as np
import pylab as pl
import pywt


path = '../data/hst_calibration.npy'


raw = np.load(path)

print(np.shape(raw))

wp = pywt.wavedec2(data=raw[0,:,:], wavelet='db2', mode='symmetric', level=2)

print(np.shape(wp))
print(np.shape(wp[0]))
print(np.shape(wp[1]))
print(np.shape(wp[2]))

print(wp[2][0])

pl.figure()
pl.imshow(wp[0])
pl.colorbar()

pl.figure()
pl.imshow(wp[1][0])
pl.colorbar()


pl.figure()
pl.imshow(wp[1][1])
pl.colorbar()


pl.figure()
pl.imshow(wp[1][2])
pl.colorbar()


pl.figure()
pl.imshow(wp[2][0])
pl.colorbar()


pl.figure()
pl.imshow(wp[2][1])
pl.colorbar()


pl.figure()
pl.imshow(wp[2][2])
pl.colorbar()


for i in range(1,3):
    wp[i][0][:] = 0.0
    wp[i][1][:] = 0.0
    wp[i][2][:] = 0.0


out = pywt.waverec2(wp, wavelet='db2', mode='symmetric',)
print(out)

pl.figure()
pl.imshow(raw[0,:,:])
pl.title('original')

pl.figure()
pl.imshow(out)
pl.title('recon')


pl.show()