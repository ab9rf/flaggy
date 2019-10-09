# python

import numpy

from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA

def clamp (x):
	i = int(round(float(x+0.5)*256))
	i = max(0,min(255,i))
	return i

def genflag (pca, vec, siz, fn):
	(minh, minw) = siz
	arr = pca.inverse_transform(vec)
	arr = numpy.fromiter( [ clamp(x) for x in numpy.ravel(arr)], numpy.int8 )
	arr = (arr.reshape((minh,minw,3)))
	im = Image.fromarray(arr, mode='RGB')
	im.save(fn)


folder = Path ('e:/kelly/projects/flags-master/flags-master/png/256')

files = list(folder.iterdir())

outf = Path ('.', 'flags')

if not outf.exists(): outf.mkdir()

minw = 256
minh = 128

stdsize = (minw,minh)
rowlen = minw*minh*3

mat = numpy.empty((0,rowlen), numpy.int8)

print (rowlen)
for f in files:
	with Image.open(f) as im:
		im2 = im.convert('RGB').resize(stdsize, resample=Image.BILINEAR)
		arr = numpy.asarray(im2).reshape(1,rowlen) / 256 - 0.5
		mat = numpy.vstack((mat, arr))

nc = 3

pca = PCA(n_components=0.999,svd_solver='full')
pca.fit(mat)

tr = pca.transform(mat)
nc = pca.n_components_
print (nc)

mean = numpy.mean(mat,0)
#print (mean.shape)
#gray = numpy.full((1,rowlen), 0.5)

pmax = numpy.amax(tr,1)
pmin = numpy.amin(tr,1)
pscale = pmax - pmin
base = pca.transform(mean.reshape((1,rowlen)))

(trr, trc) = tr.shape

genflag (pca, base, (minh, minw), outf / (f'flag-0.png'))

for k in range(0,nc):
	for (c,j) in [('m',-1),('p',1)]:
		vec = numpy.copy(base)
		vec[0,k] = vec[0,k] + pscale[k] * j
		genflag(pca, vec, (minh, minw), outf / (f'flag-{k}{c}.png'))

