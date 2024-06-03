from toolkits import construct_feature_extractor, deep_features, read_images
from feature_filters import channel_mean_image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

band_choices = {
    'optic': [3, 2, 1],
    'fire': [5, 4, 2],
    'veg': [4, 3, 2],
    'l7_veg': [3, 2, 1],
}

def cosine_embedding(prior_path, detect_path, season, band_choice='fire'):
    # read image
    band_choice = band_choices[band_choice]
    p_img = read_images(prior_path, band_choice=band_choice)
    d_img = read_images(detect_path, band_choice=band_choice)
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
    cos_anomaly = (cos_anomaly - np.min(cos_anomaly)) / (np.max(cos_anomaly) - np.min(cos_anomaly)) * 255
    cos_anomaly = cos_anomaly.astype(np.uint8)
    plt.imshow(cos_anomaly, cmap='jet')
    #plt.show()
    ret, cos_anomaly = cv2.threshold(cos_anomaly, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cos_anomaly = cv2.resize(cos_anomaly, (h, w))
    cv2.imwrite('./cemb_'+season+'.jpg', cos_anomaly)

def euclidean_embedding(prior_path, detect_path, season, band_choice='fire'):
    # read image
    band_choice = band_choices[band_choice]
    p_img = read_images(prior_path, band_choice=band_choice)
    d_img = read_images(detect_path, band_choice=band_choice)
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
    eu_anomaly = (eu_anomaly - np.min(eu_anomaly)) / (np.max(eu_anomaly) - np.min(eu_anomaly)) * 255
    eu_anomaly = eu_anomaly.astype(np.uint8)
    plt.imshow(eu_anomaly, cmap='jet')
    #plt.show()
    ret, eu_anomaly = cv2.threshold(eu_anomaly, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    eu_anomaly = cv2.resize(eu_anomaly, (h, w))
    cv2.imwrite('./eemb_'+season+'.jpg', eu_anomaly)

if __name__ == '__main__':
    seasons = ['spr', 'sum', 'aut', 'win']
    d_img = r'D:\bnu\esad\demo_programs\dataset2\karymsky_volcano\cliped\karymsky_post.tif'
    for season in tqdm(seasons):
        p_img = r'D:\bnu\esad\demo_programs\dataset2\karymsky_volcano\cliped\paired_'+season+'.tif'
        euclidean_embedding(p_img, d_img, season)