# --
# model collection

import torch
import numpy as np
import importlib
import pickle
from pathlib import Path


class FundamentalModel():
  """
  fundamental model base class
  """

  def __init__(self, cfg, **kwargs):

    # arguments
    self.cfg = cfg
    self.kwargs = kwargs

    # update via kwargs
    [self.cfg.update({k: v}) for k, v in self.kwargs.items() if k in self.cfg.keys()]

    # members
    self.model = None
    self.save_path = Path(self.cfg['save_path']) if 'save_path' in self.cfg.keys() else Path('./')
    self.model_file_path = self.save_path / self.cfg['model_file_name'] if 'model_file_name' in self.cfg.keys() else 'model.pth'

    # setup
    self.setup()


  def __call__(self, x):
    """
    call
    """
    return self.predict(x)


  def setup(self):
    """
    setup model, overwrite
    """
    pass


  def predict(self, x):
    """
    predict, overwrite
    """
    return None


  def forward(self, x):
    """
    forward, overwrite
    """
    return None


  def fit(self, x, y):
    """
    fit data, overwrite
    """
    pass


  def train_step(self, data):
    """
    training step, return loss
    """
    return 0.0


  def validation_step(self, data):
    """
    validation step, return prediciton and loss
    """
    return 0, 0.0


  def training_update(self, epoch, validation_loss):
    """
    once an epoch ended, update model training
    """
    pass


  def early_stopping_criteria(self):
    """
    early stopping criteria
    """
    return False


  def crossentropyloss(self, x, y):
    """
    cross entropy loss, no batch considered
    """
    return -np.log((np.exp(x[y]) / np.sum([np.exp(x[c]) for c in range(len(x))])))


  def save(self):
    """
    save model
    """
    pass


  def load(self, file_name):
    """
    load model,
    """
    pass


  def info(self):
    """
    info
    """
    pass


  def train(self):
    """
    train setup
    """
    pass


  def eval(self):
    """
    eval setup
    """
    pass


  def analytics(self, **kwargs):
    """
    analytics
    """
    pass


  def get_save_path(self): return self.save_path



class RandomModel(FundamentalModel):
  """
  Random Model
  """

  def __init__(self, cfg, **kwargs):

    # super constructor
    super().__init__(cfg, **kwargs)

    # must have number of classes
    self.num_classes = self.cfg['num_classes']


  def predict(self, x):
    """
    predict
    """

    # forward
    y_hat = self.forward(x)

    # predict
    y_pred = np.argmax(y_hat, axis=-1)

    return y_pred


  def forward(self, x):
    """
    forward
    """

    # random class
    random_classes = [np.random.randn(self.num_classes) for i in range(len(x))]

    # softmax classes -> probabilities
    random_classes = np.array([np.exp(r) / np.sum(np.exp(r)) for r in random_classes])

    return random_classes


  def validation_step(self, data):
    """
    validation step
    """

    # get infos
    x, y = data[0], data[1]

    # forward 
    y_hat = self.forward(x)

    # loss
    loss = np.mean([self.crossentropyloss(y_hat_i, yi) for y_hat_i, yi in zip(y_hat, y)]) if len(y.shape) else self.crossentropyloss(y_hat, y)

    # predict
    y_pred = np.argmax(y_hat, axis=-1)

    return y_pred, loss


  def info(self):
    """
    info
    """
    print("Hi, this is the random model")


  def save(self):
    """
    save model
    """
    pickle.dump(self.model, open(self.model_file_path, 'wb'))


  def load(self, file_path):
    """
    load model,
    """
    self.model = pickle.load(open(file_path, 'rb'))



class SVMModel(FundamentalModel):
  """
  svm model
  """

  def __init__(self, cfg, **kwargs):

    # super constructor
    super().__init__(cfg, **kwargs)


  def setup(self):
    """
    setup model
    """

    from sklearn.ensemble import RandomForestClassifier

    # get module class
    SVC = getattr(importlib.import_module(self.cfg['module']), self.cfg['attr'])

    # create instance
    self.model = SVC(**self.cfg['kwargs'])


  def train_step(self, data):
    """
    training step
    """

    # small data conversion
    x = data[0].numpy() if torch.is_tensor(data[0]) else data[0]
    y = data[1].numpy().ravel() if torch.is_tensor(data[1]) else data[1]

    # fit data
    self.fit(x, y)

    return 0.0


  def validation_step(self, data):
    """
    validation step
    """

    # small data conversion
    x = data[0].numpy() if torch.is_tensor(data[0]) else data[0]
    y = data[1].numpy().ravel() if torch.is_tensor(data[1]) else data[1]

    # forward 
    y_pred = self.predict(x)

    return y_pred, 0.0


  def fit(self, x, y):
    """
    fit data
    """
    self.model.fit(x, y)


  def predict(self, x):
    """
    predict data example
    """
    return self.model.predict(x)


  def save(self):
    """
    save model
    """
    pickle.dump(self.model, open(self.model_file_path, 'wb'))


  def load(self, file_path):
    """
    load model,
    """
    self.model = pickle.load(open(file_path, 'rb'))



class RandomForestModel(FundamentalModel):
  """
  random forest model
  """

  def __init__(self, cfg, **kwargs):

    # super constructor
    super().__init__(cfg, **kwargs)


  def setup(self):
    """
    setup model
    """

    from sklearn.ensemble import RandomForestClassifier

    # model init
    self.model = RandomForestClassifier(**self.cfg['kwargs'])


  def train_step(self, data):
    """
    training step
    """

    # small data conversion
    x = data[0].numpy() if torch.is_tensor(data[0]) else data[0]
    y = data[1].numpy().ravel() if torch.is_tensor(data[1]) else data[1]

    # fit data
    self.fit(x, y)

    return 0.0


  def validation_step(self, data):
    """
    validation step
    """

    # small data conversion
    x = data[0].numpy() if torch.is_tensor(data[0]) else data[0]
    y = data[1].numpy().ravel() if torch.is_tensor(data[1]) else data[1]

    # forward 
    y_pred = self.predict(x)

    return y_pred, 0.0


  def fit(self, x, y):
    """
    fit data
    """
    self.model.fit(x, y)


  def predict(self, x):
    """
    predict data example
    """
    return self.model.predict(x)


  def save(self):
    """
    save model
    """
    pickle.dump(self.model, open(self.model_file_path, 'wb'))


  def load(self, file_path):
    """
    load model,
    """
    self.model = pickle.load(open(file_path, 'rb'))



if __name__ == '__main__':
  """
  model collection
  """

  import yaml
  import sklearn

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # iris dataset
  iris = sklearn.datasets.load_iris()

  # data
  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(iris.data, iris.target, test_size=0.25)

  # random model
  random_model = RandomModel({'num_classes': len(iris.target_names)})

  # train step
  random_model.train_step((x_test, y_test))

  # validation step
  y_hat, loss = random_model.validation_step((x_test, y_test))

  print("random model on iris:")
  print("pred: ", y_hat)
  print("acc: ", np.mean(y_hat == y_test))

  # init model
  svm_model = SVMModel(cfg['svm_model'])

  # fit model
  svm_model.train_step((x_train, y_train))
  y_hat, loss = svm_model.validation_step((x_test, y_test))

  print("svm model on iris:")
  print("pred: ", y_hat)
  print("acc: ", np.mean(y_hat == y_test))

  # init model
  random_forest_model = RandomForestModel(cfg['random_forest_model'])

  # fit model
  random_forest_model.train_step((x_train, y_train))
  y_hat, loss = random_forest_model.validation_step((x_test, y_test))

  print("random forest model on iris:")
  print("pred: ", y_hat)
  print("acc: ", np.mean(y_hat == y_test))
