from toolkits import read_images
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.gmm import GMM
from pyod.models.pca import PCA
from pyod.models.hbos import HBOS
import cv2

band_choices = {
    'optic': [3, 2, 1],
    'fire': [5, 4, 2],
    'veg': [4, 3, 2],
    'l7_optic': [2, 1, 0],
    'l7_veg': [3, 2, 1],
    'l7_fire': [4, 3, 2],
    'l8': [5, 4, 3, 2, 1],
    'l7': [4, 3, 2, 1, 0],
}

def cosine_baseline(prior_path, detection_path, pband_choice, dband_choice):
    p_img = read_images(prior_path, band_choice=band_choices[pband_choice])
    d_img = read_images(detection_path, band_choice=band_choices[dband_choice])
    h, w, _ = d_img.shape
    # normalize p_img
    p_flat = np.reshape(p_img, (p_img.shape[0] * p_img.shape[1], p_img.shape[2]))
    p_deno = np.linalg.norm(p_flat, ord=2, axis=1)
    p_deno = np.expand_dims(p_deno, axis=-1)
    p_l2norm = p_flat / p_deno
    # normalize d_img
    d_flat = np.reshape(d_img, (d_img.shape[0] * d_img.shape[1], d_img.shape[2]))
    d_deno = np.linalg.norm(d_flat, ord=2, axis=1)
    d_deno = np.expand_dims(d_deno, axis=-1)
    d_l2norm = d_flat / d_deno
    # compute cosine similarity
    cos_anomaly = 1 - np.sum(np.multiply(p_l2norm, d_l2norm), axis=1).reshape(h, w)
    cos_anomaly = np.nan_to_num(cos_anomaly, nan=0.)
    cos_anomaly = (cos_anomaly - np.min(cos_anomaly)) / (np.max(cos_anomaly) - np.min(cos_anomaly))
    #plt.imshow(cos_anomaly, cmap='jet')
    #plt.show()
    np.save('./cosine_baseline_result.npy', cos_anomaly)

def euclidean_baseline(prior_path, detection_path, pband_choice, dband_choice):
    #print(prior_path)
    p_img = read_images(prior_path, band_choice=band_choices[pband_choice])
    d_img = read_images(detection_path, band_choice=band_choices[dband_choice])

    # compute metrics
    eu_anomaly = np.sqrt(np.sum(((p_img - d_img)**2), axis=-1))
    eu_anomaly = np.nan_to_num(eu_anomaly, nan=0.)
    eu_anomaly = (eu_anomaly - np.min(eu_anomaly)) / (np.max(eu_anomaly) - np.min(eu_anomaly))
    #plt.imshow(eu_anomaly, cmap='jet')
    #plt.show()
    np.save('./euclidean_baseline_result.npy', eu_anomaly)

def knn_anomaly(prior_path, detection_path, pband_choice, dband_choice):
    p_img = read_images(prior_path, band_choice=band_choices[pband_choice])
    d_img = read_images(detection_path, band_choice=band_choices[dband_choice])
    h, w, _ = d_img.shape
    # reshape images
    p_flat = np.reshape(p_img, (p_img.shape[0] * p_img.shape[1], p_img.shape[2]))
    d_flat = np.reshape(d_img, (d_img.shape[0] * d_img.shape[1], d_img.shape[2]))
    # prior computation
    clf = KNN(method='largest', n_neighbors=5)
    clf.fit(p_flat)
    score = clf.predict_proba(d_flat, method='linear')
    score = np.array(score)
    score_a = np.reshape(score[:, 1], (h, w))
    #plt.imshow(score_a, cmap='jet')
    #plt.show()
    np.save('./knn_result.npy', score_a)

def ifr_anomaly(prior_path, detection_path, pband_choice, dband_choice):
    p_img = read_images(prior_path, band_choice=band_choices[pband_choice])
    d_img = read_images(detection_path, band_choice=band_choices[dband_choice])
    h, w, _ = d_img.shape
    # reshape images
    p_flat = np.reshape(p_img, (p_img.shape[0] * p_img.shape[1], p_img.shape[2]))
    d_flat = np.reshape(d_img, (d_img.shape[0] * d_img.shape[1], d_img.shape[2]))
    # prior computation
    clf = IForest()
    clf.fit(p_flat)
    score = clf.predict_proba(d_flat, method='linear')
    score = np.array(score)
    score_a = np.reshape(score[:, 1], (h, w))
    #plt.imshow(score_a, cmap='jet')
    #plt.show()
    np.save('./ifr_result.npy', score_a)

def gmm_anomaly(prior_path, detection_path, pband_choice, dband_choice):
    p_img = read_images(prior_path, band_choice=band_choices[pband_choice])
    d_img = read_images(detection_path, band_choice=band_choices[dband_choice])
    h, w, _ = d_img.shape
    # reshape images
    p_flat = np.reshape(p_img, (p_img.shape[0] * p_img.shape[1], p_img.shape[2]))
    d_flat = np.reshape(d_img, (d_img.shape[0] * d_img.shape[1], d_img.shape[2]))
    # prior computation
    clf = GMM()
    clf.fit(p_flat)
    score = clf.predict_proba(d_flat, method='linear')
    score = np.array(score)
    score_a = np.reshape(score[:, 1], (h, w))
    #plt.imshow(score_a, cmap='jet')
    #plt.show()
    np.save('./gmm_result.npy', score_a)

def pca_anomaly(prior_path, detection_path, pband_choice, dband_choice):
    p_img = read_images(prior_path, band_choice=band_choices[pband_choice])
    d_img = read_images(detection_path, band_choice=band_choices[dband_choice])
    h, w, _ = d_img.shape
    # reshape images
    p_flat = np.reshape(p_img, (p_img.shape[0] * p_img.shape[1], p_img.shape[2]))
    d_flat = np.reshape(d_img, (d_img.shape[0] * d_img.shape[1], d_img.shape[2]))
    # prior computation
    clf = PCA()
    clf.fit(p_flat)
    score = clf.predict_proba(d_flat, method='linear')
    score = np.array(score)
    score_a = np.reshape(score[:, 1], (h, w))
    #plt.imshow(score_a, cmap='jet')
    #plt.show()
    np.save('./pca_result.npy', score_a)

def hbos_anomaly(prior_path, detection_path, pband_choice, dband_choice):
    p_img = read_images(prior_path, band_choice=band_choices[pband_choice])
    d_img = read_images(detection_path, band_choice=band_choices[dband_choice])
    h, w, _ = d_img.shape
    # reshape images
    p_flat = np.reshape(p_img, (p_img.shape[0] * p_img.shape[1], p_img.shape[2]))
    d_flat = np.reshape(d_img, (d_img.shape[0] * d_img.shape[1], d_img.shape[2]))
    # prior computation
    clf = HBOS()
    clf.fit(p_flat)
    score = clf.predict_proba(d_flat, method='linear')
    score = np.array(score)
    score_a = np.reshape(score[:, 1], (h, w))
    #plt.imshow(score_a, cmap='jet')
    #plt.show()
    np.save('./hbos_result.npy', score_a)

if __name__ == '__main__':
    d_img = r'D:\bnu\esad_2023\demo_programs\dataset2\co_mudslide\cliped\co_post.tif'
    p_img = r'D:\bnu\esad_2023\demo_programs\dataset2\co_mudslide\cliped\co_pre.tif'
    methods = [cosine_baseline, euclidean_baseline, gmm_anomaly, pca_anomaly, hbos_anomaly, ifr_anomaly]
    for method in methods:
        method(p_img, d_img, 'l7', 'l8')
    #gmm_anomaly(p_img, d_img, 'l8', 'l8')