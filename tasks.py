from utils import *
from tsne import *

# import des donnees
images = load_data(path='./dataset')

# t-SNE
nsamples, nx, ny, nz = images.shape

data = images.reshape((nsamples, nx*ny*nz))

embedded_data = tsne(X=data, perplexity=30.0)

artists = imscatter(embedded_data[:,0], embedded_data[:,1], images, ax=None, zoom=0.1)

# fig, ax = plt.subplots()
# ax.plot(embedded_data[:,0], embedded_data[:,1])
# ax.add_artist(artists)
plt.show()

