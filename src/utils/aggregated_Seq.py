import math
import numpy as np
import scipy
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from scipy.linalg import fractional_matrix_power
from sklearn.metrics import accuracy_score


def kernel(xi, uj):
    # uj = uj[0]
    k_ans = 0
    for i in range(0, len(xi)):
        k_ans = k_ans + math.exp(-(np.absolute(xi[i] - uj[i])))
    return k_ans


def compute_p_nearest(xi, p, anchors):
    distances = []
    ans = []
    counter = 0
    for anchor in anchors:
        ans.append(counter)
        counter = counter + 1
        distances.append(np.sum(np.subtract((xi, anchor) ** 2)))
    ind = np.argsort(distances)
    ans = np.array(ans)
    ans = ans[ind]
    return (ans[0:p])


def seqkm(k, Images):
    print("SeqKM start")
    M = Images
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(M)
    print("Kmeans++ done")
    centers, label = kmeans.cluster_centers_, kmeans.labels_
    print("SeqKM done")
    return label, centers


def ssvd(z):
    print("SSVD start")
    zt = np.transpose(z)
    s = np.matmul(zt, z)
    print(s.shape)
    sigma, B = np.linalg.eig(s)
    print(B.shape)
    print(sigma.shape)
    sigma = np.diag(sigma)
    print(sigma.shape)
    sigma = fractional_matrix_power(sigma, 0.5)
    sigma_inverse = fractional_matrix_power(sigma, -1)
    print(sigma_inverse.shape)
    R = np.matmul(B, sigma_inverse)
    A = np.matmul(z, R)
    # return show_result(z, A, sigma, B)
    return A, sigma, B


def seqsc(x, k, m):
    print("SeqSC start")
    my_x = x
    label_all, anchors = seqkm(m, my_x)
    p = 5
    d = [0] * m
    z = np.zeros((len(x), m))
    for i in range(0, len(x)):
        ux = compute_p_nearest(my_x[i], p, anchors)
        sum_k = 0
        for j in ux:
            z[i][j] = kernel(my_x[i], anchors[j])
            sum_k += z[i][j]
            d[j] = d[j] + z[i][j]
        for j in ux:
            z[i][j] /= sum_k
    d = np.diag(d)
    d = scipy.linalg.fractional_matrix_power(d, -0.5)
    z_bar = np.matmul(z, d)
    A, sigma, B = ssvd(z_bar)
    A_my = []
    for block in A:
        A_my.append(block[1:k + 1])
    label_all, centers = seqkm(k, A_my)
    print("SeqSC done")
    return label_all, centers, anchors


def guiseqsc(k, n, m, f):
    data = load_digits()
    X_train = data.data
    y_train = data.target
    X_train = X_train[0:n]
    y_train = y_train[0:n]
    z = []
    for i in range(10):
        z.append(0)
    for y in y_train:
        z[y] = z[y] + 1
    if (f > 0):
        print("apply filter")
    label_all, centers, anchors = seqsc(X_train, k, m)
    return X_train, y_train, label_all


x, y, label = guiseqsc(10, 100, 20, 1)
print(label)
print(accuracy_score(y, label))
