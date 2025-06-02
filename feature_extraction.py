# --
# feature extraction

import numpy as np
import torch
import skimage


def extract_hog_features(cfg, data, model_class):
  """
  extract hog features
  """

  # skip if not required
  if not cfg['training']['use_extracting_hog']: return data

  # only for svm and random forest
  use_one_dim = model_class.__name__ == 'SVMModel' or model_class.__name__ == 'RandomForestModel'

  # data preparation
  x, y = data[0], data[1]
  sid = None if not len(data) == 3 else data[2]
  
  # kwargs
  kwargs = {'orientations': 9, 'pixels_per_cell': [4, 4], 'cells_per_block': [3, 3], 'block_norm': 'L2-Hys', 'visualize': True, 'transform_sqrt': False, 'feature_vector': True, 'channel_axis': 0}

  # no batch dimension
  if len(x.shape) == 3:
    features, hog_image = skimage.feature.hog(x.numpy(), **kwargs)
    features = torch.from_numpy(features)
    hog_image = torch.from_numpy(hog_image[np.newaxis, :])
    return features if use_one_dim else hog_image, y, sid

  # hog for batch dimension
  features, hog_image = zip(*[skimage.feature.hog(xi.numpy(), **kwargs) for xi in x])
  features = torch.from_numpy(np.array([i for i in features]))
  hog_image = torch.from_numpy(np.array([i for i in hog_image])[:, np.newaxis, :])

  return features if use_one_dim else hog_image, y, sid