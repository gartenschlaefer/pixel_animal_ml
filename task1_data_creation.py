# --
# task 1: data creation

# In task 1 you should create images as described in the assignment sheet.
# This code provides you with a function to downsample the data.
# Fill the sections with # *** your code her *** #

import numpy as np
from pathlib import Path


def load_image_from_file(file):
  """
  load an image from a file, make it a numpy array, remove the alpha channel if there exists one
  and make the color channel dimension on first position of the array, e.g.: [color channel x height x width]
  useful function: imageio.v3.imread, numpy.transpose
  """

  # ***
  # your code here

  raise NotImplementedError()

  # image
  img = None

  #
  # ***

  return img


def get_files_from_path(path, file_ext='.png'):
  """
  get files from dataset path
  note: make sure that the search of glob is recursive '**/' might be useful
  useful functions: Path('path').glob
  """

  # ***
  # your code here

  raise NotImplementedError()
  
  # get files
  files = None

  #
  # ***

  return files


def downsample_files(file_path, save_path, file_ext='.png'):
  """
  use this to create downsampled versions of existing images
  the function will downsample the files in file_path and create an upsampled version of it (to view the file),
  after downsampling place the files (in save_path) in the dataset folder under the correct class name
  note that the files must have equal size e.g. 400x400 pixels
  """

  import cv2

  # get files
  files = get_files_from_path(file_path, file_ext=file_ext)

  # upsample path
  upsample_path = Path(save_path) / 'upsample/'

  # create path
  if not Path(save_path).is_dir(): Path(save_path).mkdir()
  if not upsample_path.is_dir(): upsample_path.mkdir()

  # process files
  for file_num, file in enumerate(files):

    # read image
    img = cv2.imread(file)

    # info message
    print("{} downsample: {} with shape {}".format(file_num, file, img.shape))

    # continuous rescaling
    while True:

      # half shape
      new_shape = tuple([int(s * 0.5) for s in img.shape[1::-1]]) if img.shape[0] >= 64 else (32, 32)

      # interpolation
      interpolation = cv2.INTER_AREA if img.shape[0] >= 64 else cv2.INTER_CUBIC

      # subtract the blurred image from the original
      high_pass = cv2.subtract(img, cv2.bilateralFilter(img, 5, 75, 75))

      # add the high-pass image back to the original
      img = cv2.addWeighted(img, 1.0, high_pass, 1.0, 0)

      # resize
      img = cv2.resize(img, new_shape, interpolation=interpolation)

      # break the loop
      if img.shape[0] <= 32: break

    # file name
    out_file_name = '{}.png'.format(file.stem)

    # save image
    cv2.imwrite(Path(save_path) / out_file_name, img)

    # resize
    r_img = cv2.resize(img, tuple([int(s * 10) for s in img.shape[1::-1]]), interpolation=cv2.INTER_NEAREST)

    # save image
    cv2.imwrite(upsample_path / out_file_name, r_img)



if __name__ == '__main__':
  """
  task 1: data creation
  """

  import yaml
  import matplotlib.pyplot as plt

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # specify dataset path in config.yaml
  dataset_path = cfg['task1']['dataset_path']

  # downsample files, you need to save them to the dataset path by yourself
  downsample_files(cfg['task1']['downsample_path'], save_path=cfg['task1']['save_downsampled_path'])

  # get files
  files = get_files_from_path(dataset_path)

  # go through each image and plot it
  for file_num, file in enumerate(files):

    # load image
    img = load_image_from_file(file)

    # assertions
    assert isinstance(img, np.ndarray) and img.shape == (3, 32, 32)

    # plot each image individually
    plt.figure()
    plt.imshow(img.transpose(1, 2, 0))
    plt.title(file.parent.stem + '/' + file.stem)
    plt.show()

  # assertion
  assert file_num + 1 >= 6, 'You have to add files to the test dataset.'

  print("Everything successful!")
