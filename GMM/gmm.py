# 
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy import linalg
from sklearn import mixture
import matplotlib as mpl

class DataGeneration:
    data = []
    def generateMultiVarNormal(self,  _mu = [0, 1], _cov = [ [.5,0], [0, 2]], _samples_size = 100):
        self.data = np.random.multivariate_normal(_mu,_cov,_samples_size,'raise')
    def combineData(self, _new_input = []):
        return np.append(self.data,_new_input,0)
    
def showData(_data):
    if _data.shape[1]>2:
        print ('Only works for 2D data points')
        return
    x = _data[:,0]
    y = _data[:,1]
    
    plt.plot(x,y, 'x')
    plt.axis('equal')
    plt.show()
        
if __name__ == '__main__':
    # Generate two 2D Gaussian distributions
    np.random.seed(0)
    dist1 = DataGeneration()
    dist2 = DataGeneration()
    dist1.generateMultiVarNormal(_mu=[0, 1], _cov=[[4, 1], [1, 2]])
    dist2.generateMultiVarNormal(_mu=[1.5, 1], _cov = [[3,0], [0,1]])
    combined = dist1.combineData(dist2.data)
    #TODO: Put it working for cases with _cov_xy = 0s
    #TODO: Explore	
    bic = []
    lowest_bic = np.infty
    n_components_range = range(1, 7)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(combined)
            bic.append(gmm.bic(combined))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm   
    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)
    
    # Plot the winner
    splot = plt.subplot(2, 1, 2)
    Y_ = clf.predict(combined)
    for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                               color_iter)):
        
        v, w = linalg.eigh(cov)
        if not np.any(Y_ == i):
            continue
        plt.scatter(combined[Y_ == i, 0], combined[Y_ == i, 1], .8, color=color)
    
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180. * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(.5)
        splot.add_artist(ell) 
#    plt.scatter(dist1.data[:,0],dist1.data[:,1], color='k')
#    plt.scatter(dist2.data[:,0],dist2.data[:,1], color='y')
#    showData(combined)
#    