# --
# task 2: datamodule

# In this task you will create a datamodule for machine learning models.
# The datamodule stores the data and feeds the model with small chunks (batches).
# Also implement the augmentation function
# Fill the sections with # *** your code her *** #

import numpy as np
import torch
import torchvision
from torchvision.transforms import v2
from pathlib import Path


class CIFAR10AnimalDatamodule(torch.utils.data.Dataset):
  """
  cifar-10 datamodule from torchvision, here we extract the training and validation data and choose only 3 classes ['dog', 'frog', 'horse'],
  nothing to do here
  """

  def __init__(self, root_path='./ignore/data/'):

    # super constructor
    super().__init__()

    # members
    self.features = None
    self.targets = None
    self.sample_ids = None
    self.length = None
    self.label_dict = None
    self.cache_info = None
    self.classes = None

    # the datamodule
    cifar10_datamodule = torchvision.datasets.CIFAR10(root=root_path, train=True, download=True, transform=v2.Compose([v2.ToImage()]))

    # cifar only with 3 animals
    label_map = {5: 0, 6: 1, 7: 2}
    self.classes = ['dog', 'frog', 'horse']

    # get features and targets
    self.features, self.targets, self.sample_ids = zip(*[(img, label_map[label], sid) for sid, (img, label) in enumerate(cifar10_datamodule) if label in label_map.keys()])

    # to torch
    self.sample_ids = torch.tensor(self.sample_ids)

    # length
    assert(len(self.features) == len(self.targets))

    # dataset length
    self.length = len(self.targets)


  def __len__(self):
    return self.length


  def __getitem__(self, idx):
    return self.features[idx], self.targets[idx], self.sample_ids[idx]


  def get_labels_from_class_number_list(self, y): 
    return [self.classes[i] for i in y]



class PixelAnimalDatamodule(torch.utils.data.Dataset):
  """
  pixel animal datamodule, use this to store your created data from task 1
  """

  def __init__(self, cfg):

    # your implemented functions from task 1
    from task1_data_creation import get_files_from_path, load_image_from_file

    # super constructor
    super().__init__()

    # arguments
    self.cfg = cfg

    # members
    self.length = None
    self.label_dict = None
    self.cache_info = None

    # specify dataset path in config.yaml
    dataset_path = self.cfg['dataset_path']

    # this you have already implemented in the previous task
    files = get_files_from_path(dataset_path)

    # store file names
    self.file_names = [Path(f).stem for f in files]

    # number of files
    num_files = len(files)

    # assertions
    assert num_files

    # get class from filename
    get_class_from_file = lambda f: Path(f).parent.stem

    # label dictionary
    self.label_dict = {c: i for i, c in enumerate(np.unique([get_class_from_file(f) for f in files]))}

    # classes
    self.classes = list(self.label_dict.keys())

    # allocate memory space
    self.features = np.empty(shape=(num_files, 3, 32, 32), dtype=np.uint8)
    self.targets = np.empty(shape=(num_files, 1), dtype=np.uint8)
    self.sample_ids = np.empty(shape=(num_files, 1), dtype=np.uint32)

    # features are the image files
    # targets are the class name, but indexed as integers (self.label_dict is useful)
    # sample_ids are the id of each data sample

    # ***
    # your code here

    raise NotImplementedError()

    # # go through each file
    # for i, file in enumerate(files):
    #   self.features...
    #   self.targets...
    #   self.sample_ids...

    #
    # ***

    # assertions
    assert len(self.features) == len(self.targets)

    # length of dataset
    self.length = len(self.targets)


  def __len__(self):
    return self.length


  def __getitem__(self, idx):
    """
    this is called from an iterator and returns the requested data indexed by idx
    """
    return torch.from_numpy(self.features[idx]), torch.from_numpy(np.squeeze(self.targets[idx])), torch.from_numpy(np.squeeze(self.sample_ids[idx]))


  def get_filename_from_sid(self, sid): 
    return np.array(self.file_names)[sid]


  def get_labels_from_class_number_list(self, y): 
    return [self.classes[i] for i in y]



def augment_data(data, num_augmentations=0):
  """
  data augmentation,
  extend the data with augmentations of itself,
  the augmentations should have
    - Gaussian noise
    - random crop
    - horizontal flip
    - vertical flip
    - greyscale (randomly applied)
    - adjust sharpness
  you are free to extend further augmentation transforms
  """

  # get data
  x, y, sid = data

  # ***
  # your code here

  raise NotImplementedError()

  # select some of the following transforms provided by pytorch
  # you can also add some more if you want to

  # transform compositions
  transform = v2.Compose([
    #v2.ToDtype(dtype=torch.float32, scale=True), v2.GaussianNoise(mean=0.0, sigma=0.05), v2.ToDtype(dtype=torch.uint8, scale=True),
    #v2.RandomCrop(size=(x.shape[2:]), padding=8),
    #v2.RandomResizedCrop(size=(x.shape[1:]), scale=(0.5, 1.5), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
    #v2.RandomHorizontalFlip(p=0.5),
    #v2.FiveCrop(size=(x.shape[1:])),
    #v2.RandomVerticalFlip(p=0.2),
    #v2.RandomAffine(degrees=(0, 0), translate=(0.1, 0.3), scale=(0.8, 1.0))
    #v2.RandomRotation(degrees=(0, 180)),
    #v2.ElasticTransform(alpha=25.0),
    #v2.RandomApply(torch.nn.ModuleList([v2.Grayscale(num_output_channels=x.shape[1])]), p=0.2),
    #v2.ColorJitter(brightness=0.4, hue=0.1),
    #v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
    #v2.RandomInvert(),
    #v2.RandomPosterize(bits=2),
    #v2.RandomSolarize(threshold=125.0),
    #v2.RandomAdjustSharpness(sharpness_factor=2),
    #v2.RandomAutocontrast(),
    #v2.RandomEqualize(),
    #v2.ToDtype(dtype=torch.float32, scale=True),
    ])

  # 
  # ***

  # add augmentations
  x_all = torch.cat((x, *[transform(x) for i in range(num_augmentations)]))
  y_all = torch.cat((y, *[y for i in range(num_augmentations)]))
  sid_all = torch.cat((sid, *[sid for i in range(num_augmentations)])) if sid is not None else None

  # augmented data
  data_augmented = x_all, y_all, sid_all

  return data_augmented



if __name__ == '__main__':
  """
  task 2: datamodule
  """

  import yaml
  from plots import plot_2dmatrix

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # datamodule
  cifar10_animal_datamodule = CIFAR10AnimalDatamodule(**cfg['task2']['cifar10_animal_datamodule']['kwargs'])
  pixel_animal_datamodule = PixelAnimalDatamodule(cfg['task2']['pixel_animal_datamodule'])

  # loader
  train_dataloader = torch.utils.data.DataLoader(cifar10_animal_datamodule, **{'batch_size': 32})
  test_dataloader = torch.utils.data.DataLoader(pixel_animal_datamodule, **{'batch_size': 32})

  # train data
  data = next(iter(train_dataloader))

  # 2d matrix
  plot_2dmatrix(data[0], num_rows=4, aspect='equal', sample_labels=cifar10_animal_datamodule.get_labels_from_class_number_list(data[1]), title='Data')

  # do some data augmentation
  data_augmented = augment_data(data, num_augmentations=1)

  # 2d matrix
  plot_2dmatrix(data_augmented[0], num_rows=6, aspect='equal', sample_labels=cifar10_animal_datamodule.get_labels_from_class_number_list(data_augmented[1]), title='Data with Augmentations')

  # test data
  data = next(iter(test_dataloader))

  # data augmentation
  data = augment_data(data, num_augmentations=4)

  # 2d matrix
  plot_2dmatrix(data[0], num_rows=3, aspect='equal', sample_labels=pixel_animal_datamodule.get_labels_from_class_number_list(data[1]), title='Test Data')

  print("Everything successful!")