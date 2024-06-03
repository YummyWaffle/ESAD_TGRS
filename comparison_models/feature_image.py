from toolkits import construct_feature_extractor, deep_features, read_images
from feature_filters import channel_mean_image
import numpy as np
import matplotlib.pyplot as plt
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

def cosine_embedding(prior_path, detection_path, pband_choice, dband_choice):
    p_img = read_images(prior_path, band_choice=band_choices[pband_choice])
    d_img = read_images(detection_path, band_choice=band_choices[dband_choice])
    h, w, _ = d_img.shape

    # construct predictor
    predictor = construct_feature_extractor()

    # extract features
    det_feat = deep_features(d_img, predictor, layer=-1)
    det_feat = channel_mean_image(det_feat, near_radius=2)
    pri_feat = deep_features(p_img, predictor, layer=-1)
    pri_feat = channel_mean_image(pri_feat, near_radius=2)
    fh, fw, _ = det_feat.shape

    # cosine baseline
    p_flat = np.reshape(pri_feat, (pri_feat.shape[0] * pri_feat.shape[1], pri_feat.shape[2]))
    p_deno = np.linalg.norm(p_flat, ord=2, axis=1)
    p_deno = np.expand_dims(p_deno, axis=-1)
    p_l2norm = p_flat / p_deno

    d_flat = np.reshape(det_feat, (det_feat.shape[0] * det_feat.shape[1], det_feat.shape[2]))
    d_deno = np.linalg.norm(d_flat, ord=2, axis=1)
    d_deno = np.expand_dims(d_deno, axis=-1)
    d_l2norm = d_flat / d_deno

    cos_anomaly = 1 - np.sum(np.multiply(p_l2norm, d_l2norm), axis=1).reshape(fh, fw)
    cos_anomaly = np.nan_to_num(cos_anomaly, nan=0.)
    cos_anomaly = (cos_anomaly - np.min(cos_anomaly)) / (np.max(cos_anomaly) - np.min(cos_anomaly))
    plt.imshow(cos_anomaly, cmap='jet')
    plt.show()
    np.save('./cosine_embdedding_result.npy', cos_anomaly)

def euclidean_embedding(prior_path, detection_path, pband_choice, dband_choice):
    p_img = read_images(prior_path, band_choice=band_choices[pband_choice])
    d_img = read_images(detection_path, band_choice=band_choices[dband_choice])
    h, w, _ = d_img.shape

    # construct predictor
    predictor = construct_feature_extractor()

    # extract features
    det_feat = deep_features(d_img, predictor, layer=-1)
    det_feat = channel_mean_image(det_feat, near_radius=2)
    pri_feat = deep_features(p_img, predictor, layer=-1)
    pri_feat = channel_mean_image(pri_feat, near_radius=2)
    fh, fw, _ = det_feat.shape

    # compute metrics
    eu_anomaly = np.sqrt(np.sum(((det_feat - pri_feat) ** 2), axis=-1))
    eu_anomaly = np.nan_to_num(eu_anomaly, nan=0.)
    eu_anomaly = (eu_anomaly - np.min(eu_anomaly)) / (np.max(eu_anomaly) - np.min(eu_anomaly))
    plt.imshow(eu_anomaly, cmap='jet')
    plt.show()
    np.save('./euclidean_embdedding_result.npy', eu_anomaly)

if __name__ == '__main__':
    d_img = r'D:\bnu\esad_2023\demo_programs\dataset2\florence_flood\cliped\flr_post.tif'
    p_img = r'D:\bnu\esad_2023\demo_programs\dataset2\florence_flood\cliped\paired_aut.tif'
    euclidean_embedding(p_img, d_img, 'fire', 'fire')