# --
# task 3: model

# In this task you will create a simple cnn model and test it.
# Note that for the next task we need to define several functions within the cnn model class in order to run the full pipeline.
# Fill the sections with # *** your code her *** #

import yaml
import numpy as np
import torch
import importlib
from pathlib import Path


class AnimalCNNModel(torch.torch.nn.Module):
  """
  animal cnn model
  """

  def __init__(self, cfg, **kwargs):

    # arguments
    self.cfg = cfg
    self.kwargs = kwargs

    # update via kwargs
    [self.cfg.update({k: v}) for k, v in self.kwargs.items() if k in self.cfg.keys()]

    # assertions
    assert len(self.cfg['input_shape']) == 3, "wrong input shape: {}, should have 3 dimensions: c x m x n".format(self.cfg['input_shape'])

    # parent init
    super().__init__()

    # members
    self.criterion = torch.nn.CrossEntropyLoss()
    self.optimizer = None
    self.device = torch.device(self.cfg['device']['device_name'] if torch.cuda.is_available() and not self.cfg['device']['use_cpu'] else 'cpu')
    self.save_path = Path(self.cfg['save_path']) if 'save_path' in self.cfg.keys() else Path('./')
    self.model_file_path = self.save_path / self.cfg['model_file_name'] if 'model_file_name' in self.cfg.keys() else 'model.pth'
    self.best_validation_loss = 9999.9
    self.current_patience = 0
    self.current_refinement = 0
    self.is_early_stopping_fulfilled = False

    # create save path
    if not self.save_path.is_dir(): self.save_path.mkdir(parents=True)

    # print device
    if self.cfg.get('verbose'): print("Animal CNN Model on device: {}".format(self.device) + ("\nGPU: {}".format(torch.cuda.get_device_name(self.device)) if torch.cuda.is_available() and not self.cfg['device']['use_cpu'] else ""))

    # --
    # network structure

    # conv layer 1
    self.layer1 = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels=self.cfg['input_shape'][0], out_channels=16, kernel_size=(16, 16), stride=(1, 1)),
      torch.nn.ReLU(),
      )

    # ***
    # your code here

    raise NotImplementedError()

    # create a new layer with 16 in and 32 out channels, kernelsize 8x8 and stride 1, 1
    # add a dropout layer: torch.nn.Dropout2d with p=0.25
    # add a max pool layer: torch.nn.MaxPool2d
    # make sure a torch.nn.ReLU is the activation function of he layer at the end

    # conv layer 2
    self.layer2 = None

    # 
    # ***

    # get shape of conv layers
    with torch.no_grad(): flattened_shape = self.layer2(self.layer1(torch.randn((1,) + tuple(self.cfg['input_shape'])))).data.shape

    # output layer
    self.output_layer = torch.nn.Sequential(
      torch.nn.Flatten(),
      torch.nn.Linear(flattened_shape.numel(), flattened_shape.numel()//2),
      torch.nn.Dropout(p=0.5),
      torch.nn.ReLU(),
      torch.nn.Linear(flattened_shape.numel()//2, self.cfg['num_classes']),
      torch.nn.Softmax(dim=1),
      )

    # setup
    self.setup()


  def setup(self):
    """
    setup
    """

    # model to device
    self.to(device=self.device)

    # epoch trainings
    if 'training_update' in self.cfg.keys():
      self.current_patience = self.cfg['training_update']['num_patience']
      self.current_refinement = 0

    # no optimizer specified
    if self.cfg.get('optimizer') is None:
      print("No optimizer specified... use Adam")
      self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, betas=(0.9, 0.999))
      return

    # optimizer class
    optimizer_class = getattr(importlib.import_module(self.cfg['optimizer']['module']), self.cfg['optimizer']['attr'])

    # optimizer
    self.optimizer = optimizer_class(self.parameters(), **self.cfg['optimizer']['kwargs'])


  def forward(self, x):
    """
    forward pass
    """

    # 1. conv layer
    h = self.layer1(x)
    #print(h.shape)

    # ***
    # your code here

    raise NotImplementedError()

    # forward layer2 and the output layer
    y = None

    #
    # ***

    return y


  def train_step(self, data):
    """
    training step
    """

    # reset optimizer
    self.optimizer.zero_grad()

    # get data
    x = data[0].to(device=self.device, dtype=torch.float32)
    y = data[1].to(device=self.device)

    # forward
    y_hat = self.forward(x)

    # loss
    loss = self.criterion(y_hat, y)

    # backwards
    loss.backward()

    # parameter update
    self.optimizer.step()

    return loss.item()


  def validation_step(self, data):
    """
    validation step
    """

    # no gradients here
    with torch.no_grad():

      # get data
      x = data[0].to(device=self.device, dtype=torch.float32)
      y = data[1].to(device=self.device)

      # forward 
      y_hat = self.forward(x)

      # loss
      loss = self.criterion(y_hat, y)

      # prediction
      y_pred = torch.argmax(y_hat, axis=-1).cpu().numpy()

    return y_pred, loss.item()


  def predict(self, x):
    """
    prediction
    """

    # make sure it is eval mode
    self.eval()

    # no gradients here
    with torch.no_grad():

      # get data
      x = x.to(device=self.device, dtype=torch.float32)

      # forward 
      y_hat = self.forward(x)

      # prediction
      y_pred = torch.argmax(y_hat, axis=-1).cpu().numpy()

    return y_pred


  def training_update(self, epoch, validation_loss):
    """
    once an epoch ended, update model training
    """

    # is not defined
    if not 'training_update' in self.cfg.keys(): return

    # validation loss gets better
    if validation_loss < self.best_validation_loss:
      self.best_validation_loss = validation_loss
      self.current_patience = self.cfg['training_update']['num_patience']
      if not 'use_early_stopping_criteria' in self.cfg.keys(): return
      if not self.cfg['use_early_stopping_criteria']: return
      self.save()
      return

    # only do something if no patience
    if self.current_patience > 0: 
      self.current_patience -= 1
      print("be patient...{}".format(self.current_patience))
      return

    # still refining?
    if self.current_refinement >= self.cfg['training_update']['num_refinements']: return

    # increase refinement and restart patience
    self.current_refinement += 1
    self.current_patience = self.cfg['training_update']['num_patience']

    # assert
    assert self.current_refinement and (self.current_refinement <= self.cfg['training_update']['num_refinements'])

    # new learning rate
    new_lr = self.cfg['optimizer']['kwargs']['lr'] * (self.cfg['training_update']['lr_scale_factor'] ** self.current_refinement)

    # update learning rate
    for param_group in self.optimizer.param_groups: param_group['lr'] = new_lr

    # message
    print("patience reached more refinement [cur step: {}] -> lower lr to: [{}]".format(self.current_refinement, new_lr))


  def early_stopping_criteria(self):
    """
    early stopping criteria
    """
    if not 'use_early_stopping_criteria' in self.cfg.keys(): return False
    return self.current_refinement >= self.cfg['training_update']['num_refinements'] and self.current_patience <= 0 and self.cfg['use_early_stopping_criteria']


  def analytics(self, show_plot=False, **kwargs):
    """
    analytics of the model
    """

    from plots import plot_2dmatrix

    # load model if early stopping was used
    if self.early_stopping_criteria(): 
      print("load early stopping")
      self.load(self.model_file_path)

    # analyze weights
    w = self.get_weights_of_first_layer_conv()

    # normalize weights
    w = np.clip((w + np.abs(np.min(w))) / np.max(np.abs(w + np.min(w))), 0.0, 1.0)

    # plot
    plot_2dmatrix(w, aspect='equal', title='First layer convolutional filter kernels of the model', num_rows=4, show_plot=show_plot, save_path=self.save_path / 'first_conv_kernels_{}.png'.format(self.model_file_path.stem))

    # second conv layer weights
    w = self.layer2[0].weight.data.cpu().numpy()
    w = np.sum(w, axis=1)
    w = np.clip((w + np.abs(np.min(w))) / np.max(np.abs(w + np.min(w))), 0.0, 1.0)

    # plot
    plot_2dmatrix(w, aspect='equal', title='Second layer convolutional filter kernels of the model', num_rows=4, show_plot=show_plot, save_path=self.save_path / 'second_conv_kernels_{}.png'.format(self.model_file_path.stem))


  def save(self):
    """
    save model
    """
    torch.save(self.state_dict(), self.model_file_path)


  def load(self, model_file):
    """
    load model
    """
    self.load_state_dict(torch.load(model_file, map_location=self.device))


  def get_weights_of_first_layer_conv(self):
    """
    get weights
    """
    return self.layer1[0].weight.data.cpu().numpy()


  def get_save_path(self): return self.save_path
  def get_model_file_path(self): return self.model_file_path



if __name__ == '__main__':
  """
  task 3: model
  """

  # datamodules
  from task2_datamodule import CIFAR10AnimalDatamodule
  from task2_datamodule import PixelAnimalDatamodule

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # datamodule
  cifar10_animal_datamodule = CIFAR10AnimalDatamodule(**cfg['task2']['cifar10_animal_datamodule']['kwargs'])
  pixel_animal_datamodule = PixelAnimalDatamodule(cfg['task2']['pixel_animal_datamodule'])

  # sample input shape
  input_shape = next(iter(cifar10_animal_datamodule))[0].shape

  # prints
  print("classes: ", cifar10_animal_datamodule.classes)
  print("input shape: ", input_shape)

  # model
  model = AnimalCNNModel({**cfg['task3']['model']['animal_cnn_model'], **{'input_shape': input_shape, 'num_classes': len(cifar10_animal_datamodule.classes)}})

  # loader
  train_loader = torch.utils.data.DataLoader(cifar10_animal_datamodule, **{'batch_size': 32})
  test_loader = torch.utils.data.DataLoader(pixel_animal_datamodule, **{'batch_size': 6})

  # set to train mode
  model.train()

  # train loader
  for data in train_loader: 

    # fit data
    loss = model.train_step(data)
    print("Training loss: [{:.6}]".format(loss))

    # stop after first batch (just to evaluate if the model works)
    break

  # set to evaluation mode
  model.eval()

  # test loader
  for data in test_loader: 

    # test data
    y_pred, loss = model.validation_step(data)
    print("Validation loss: [{:.6}], accuracy: [{}]".format(loss, np.mean(y_pred == data[1].numpy())))

  # training update (just for testing)
  model.training_update(epoch=0, validation_loss=loss)

  # model analytics
  model.analytics(show_plot=True)

  # save model
  model.save()

  # new model
  loaded_model = AnimalCNNModel({**cfg['task3']['model']['animal_cnn_model'], **{'model_file_name': 'loaded_model', 'input_shape': input_shape, 'num_classes': len(cifar10_animal_datamodule.classes)}})
  loaded_model.load(model.get_model_file_path())

  # model analytics
  loaded_model.analytics(show_plot=True)

  print("Everything successful!")