import numpy as np
import matplotlib.pyplot as plt
import sklearn.manifold as TSNE

def plot_tsne(features, labels, save_filename, save_eps=False):
    ''' Plot TSNE figure. Set save_eps=True if you want to save a .eps file.
    '''
    features = TSNE(n_components=2, init='pca', random_state=0).fit_transform(features)
    x_min, x_max = np.min(features, 0), np.max(features, 0)
    data = (features - x_min) / (x_max - x_min)
    del features
    for i in range(data.shape[0]):
        colors = plt.cm.tab20.colors
        plt.scatter(data[i, 0], data[i, 1], color=colors[labels[i]])
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title('T-SNE')
    if save_eps:
        plt.savefig('tsne.eps', dpi=600, format='eps')
    plt.savefig(save_filename, dpi=600)
    plt.show()


if __name__ == '__main__':
    outm = []
    y_batch = []
    for i in range(9):
        outmt = np.load(file='./featuresForTSNE/outm'+ str(i) +'.npy')
        yt = np.load(file='./featuresForTSNE/y_batch'+str(i)+'.npy')
        outmt = np.reshape(outmt, (64, -1, 3, outmt.shape[-1]))
        outmt = np.reshape(outmt, (64, -1, outmt.shape[-1]))
        y_batch = np.append(y_batch, yt)
    plot_tsne(outm, m)
    plot_tsne(outm, y_batch)

