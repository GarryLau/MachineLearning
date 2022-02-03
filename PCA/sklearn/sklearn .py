
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy

iris = datasets.load_iris()

x = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_r = pca.fit(x).transform(x)
cov = pca.get_covariance()
# X_r = pca.fit_transform(x)
print("explained variance ratio (first two components): %s" % str(pca.explained_variance_ratio_))

#fout = open(r'D:\2.Work\0.LearnML\0.PCA\pca\data\sklearn_pca_iris.txt', 'w',encoding='UTF-8')
#for i in range(len(X_r)):
#    fout.write(str(X_r[i][0])+', '+str(X_r[i][1])+'\n')
#fout.close()

numpy.savetxt(r'D:\2.Work\0.LearnML\0.PCA\pca\data\sklearn_pca_iris.txt', X_r, fmt='%.12f', delimiter=', ', newline='\n')

plt.figure()
colors = ["navy", "turquoise", "darkorange"]
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA of IRIS dataset")
plt.show()


