# python

import numpy
import math

from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

import pickle
from joblib import dump, load

def clamp (x):
	i = int(round(float(x+0.5)*256))
	i = max(0,min(i,255))
	return i

def genflag (mlp, vec, siz, fn):
	(minh, minw) = siz
	arr = numpy.fromiter( [ clamp(x) for x in numpy.ravel(mlp.predict(vec))], numpy.int8 ).reshape((minh,minw,3))
	im = Image.fromarray(arr, mode='RGB')
	im.save(fn)

folder = Path ('e:/kelly/projects/flags-master/flags-master/png/512')

outf = Path ('.', 'flags2')
regenf = Path('.', 'regen')

if not outf.exists(): outf.mkdir()
if not regenf.exists(): regenf.mkdir()

minw = 256
minh = 128

stdsize = (minw,minh)
rowlen = minw*minh*3

mat = numpy.empty((0,rowlen), numpy.int8)
flaglist = []

def readimg(f):
	with Image.open(f) as im:
		name = f.stem
		im2 = im.convert('RGB').resize(stdsize, resample=Image.BILINEAR)
		arr = numpy.asarray(im2).reshape(1,rowlen) / 256 - 0.5
		return arr
	
mlp_pickle = Path('.', 'saved_mlp.pickle')
if (not mlp_pickle.exists()):
	for f in folder.iterdir():
		arr = readimg(f)
		mat = numpy.vstack((mat, arr))
			
	pca = PCA(n_components=0.99,svd_solver='full')
	pca.fit(mat)
	featmat = pca.transform(mat)
	(ns,nf) = featmat.shape
			
	mlp = MLPRegressor(verbose=True, hidden_layer_sizes=(ns,) )
	mlp.fit (featmat, mat)
	dump((mlp, pca, featmat), mlp_pickle)
else:
	(mlp, pca, featmat) = load(mlp_pickle)

(ns,nf) = featmat.shape
mean = numpy.mean(featmat,0)
#print (mean.shape)
#gray = numpy.full((1,rowlen), 0.5)

pmax = numpy.amax(featmat,1)
pmin = numpy.amin(featmat,1)
pscale = pmax - pmin
base = numpy.copy(mean).reshape((1,nf))

for f in folder.iterdir():
	name = f.name
	arr = readimg(f)
	arr = pca.transform(arr)
	genflag (mlp, arr, (minh, minw), regenf / name)
	

genflag (mlp, base, (minh, minw), outf / (f'flag-0.png'))

for k in range(0,nf):
	for (c,j) in [('m',-1),('p',1)]:
		vec = numpy.copy(base)
		vec[0,k] = vec[0,k] + pscale[k] * j
		genflag(mlp, vec, (minh, minw), outf / (f'flag-{k}{c}.png'))

