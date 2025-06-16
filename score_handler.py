# --
# score handler

import numpy as np


class ScoreHandler():
  """
  score handler
  """

  def __init__(self, classes):

    # arguments
    self.classes = classes

    # members
    self.num_classes = len(classes)
    self.tp = 0
    self.tn = 0
    self.fn = 0
    self.fp = 0
    self.accuracy = None
    self.precision = None
    self.f1_score = None
    self.recall = None
    self.y_target_collect = []
    self.y_predicted_collect = []
    self.train_loss_collect = []
    self.validation_loss_collect = []

    # epoch scores
    self.train_epoch_loss = []
    self.validation_epoch_loss = []
    self.validation_epoch_acc = []
    self.y_target_last_epoch = []
    self.y_predicted_last_epoch = []

    # best epoch
    self.best_epoch = 0
    self.best_epoch_validation_loss = 9999.9
    self.best_epoch_validation_acc = 0.0
    self.best_epoch_y_target = []
    self.best_epoch_y_predicted = []


  def reset(self):
    """
    reset
    """
    self.tp = 0
    self.tn = 0
    self.fn = 0
    self.fp = 0
    self.accuracy = None
    self.precision = None
    self.f1_score = None
    self.recall = None
    self.y_target_collect = []
    self.y_predicted_collect = []
    self.train_loss_collect = []
    self.validation_loss_collect = []


  def epoch_finished(self, epoch, print_scores=True, print_function=print):
    """
    epoch finished
    """

    # train loww per epoch
    self.train_epoch_loss.append(np.mean(self.train_loss_collect).item())
    self.validation_epoch_acc.append(self.accuracy)
    self.validation_epoch_loss.append(np.mean(self.validation_loss_collect).item())
    self.y_target_last_epoch = [y.item() for y in self.y_target_collect]
    self.y_predicted_last_epoch = [y.item() for y in self.y_predicted_collect]

    # best epoch
    if self.validation_epoch_loss[-1] < self.best_epoch_validation_loss:
      self.best_epoch = epoch
      self.best_epoch_validation_loss = self.validation_epoch_loss[-1]
      self.best_epoch_validation_acc = self.validation_epoch_acc[-1]
      self.best_epoch_y_target = list(self.y_target_last_epoch)
      self.best_epoch_y_predicted = list(self.y_predicted_last_epoch)

    # reset collections for new epoch
    self.reset()

    # print scores
    if not print_scores: return
    print_function("ep: {:04}, train loss: {:.6}, val loss: {:.6}, val acc: {:.4} {}".format(epoch, self.train_epoch_loss[-1], self.validation_epoch_loss[-1], self.validation_epoch_acc[-1], '*' if self.best_epoch == epoch else ''))


  def update_train(self, loss):
    """
    train loss update
    """
    self.train_loss_collect.append(loss)


  def update_validation(self, y_target, y_predicted, loss):
    """
    update scores
    """

    # to numpy
    y_target = np.array(y_target)
    y_predicted = np.array(y_predicted)

    [self.y_target_collect.append(y) for y in y_target]
    [self.y_predicted_collect.append(y) for y in y_predicted]
    self.validation_loss_collect.append(loss)

    # logic
    #y_target, y_predicted, positives_idx, negatives_idx = self.logic_class_to_onehot(y_target, y_predicted)

    positives_idx = y_target == y_predicted
    negatives_idx = y_target != y_predicted

    # true prediction
    self.tp += np.sum(y_predicted[positives_idx] == y_target[positives_idx])
    self.fp += np.sum(y_predicted[positives_idx] != y_target[positives_idx])

    # false prediction
    self.fn += np.sum(y_predicted[negatives_idx] != y_target[negatives_idx])
    self.tn += np.sum(y_predicted[negatives_idx] == y_target[negatives_idx])

    # score updates
    self.accuracy = ((self.tp + self.tn) / np.sum([self.tp, self.fp, self.fn, self.tn])).item()
    self.precision = self.tp / np.sum([self.tp + self.fp]) if (self.tp + self.fp) else 0.0
    self.recall = self.tp / np.sum(self.tp + self.fn) if (self.tp + self.fn) else 0.0
    self.f1_score = (2 * self.precision * self.recall / (self.precision + self.recall)) if (self.precision + self.recall) else 0.0


  def logic_class_to_onehot(self):
    """
    logic for classes that should be one-hot encoded
    """

    # with class labels expand to one-hot
    y_target = self.one_hot_encoding(y_target)
    y_predicted = self.one_hot_encoding(y_predicted)

    # positive and negative indices
    positives_idx = (y_target == True)
    negatives_idx = (y_target == False)

    return y_target, y_predicted, positives_idx, negatives_idx


  def one_hot_encoding(self, c):
    """
    one hot encoding
    """
    return np.eye(self.num_classes)[c.reshape(-1)].astype(np.bool_)


  def info(self):
    """
    info
    """
    print("summary: ")
    [print('{}: {}'.format(k, v)) for k, v in self.get_score_summary_dict().items()]


  def compute_confusion_matrix(self, y_target, y_predicted):
    """
    compute confusion matrix
    """
    import sklearn

    # confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_target, y_predicted)

    # to a string element
    confusion_matrix = ['|'.join([str(row_col_element.item()) for row_col_element in row]) for row in cm]

    return confusion_matrix


  def save(self, save_path, name_addon=''):
    """
    save scores
    """

    import yaml
    from pathlib import Path

    # scores
    score_dict = {
      'a_summary': {
        'classes': ['{}: {}'.format(i, c) for i, c in enumerate(self.classes)],
        'confusion_matrix': self.compute_confusion_matrix(self.best_epoch_y_target, self.best_epoch_y_predicted),
        'best_epoch': self.best_epoch,
        'best_epoch_validation_loss': self.best_epoch_validation_loss,
        'best_epoch_validation_acc': self.best_epoch_validation_acc,
        },
      'train_epoch_loss': self.train_epoch_loss,
      'validation_epoch_loss': self.validation_epoch_loss,
      'validation_epoch_acc': self.validation_epoch_acc,
      }

    with open(str(Path(save_path) / 'scores{}.yaml'.format(name_addon)), 'w') as f: yaml.dump(score_dict, f)


  def get_score_summary_dict(self):
    """
    score summary
    """
    return {'best_epoch': self.best_epoch, 'loss': self.best_epoch_validation_loss, 'acc': self.best_epoch_validation_acc, 'cm': self.compute_confusion_matrix(self.best_epoch_y_target, self.best_epoch_y_predicted)}


  def get_accuracy(self): return self.accuracy
  def get_actual_epoch_train_loss(self): return self.train_epoch_loss[-1]
  def get_actual_epoch_validation_acc(self): return self.validation_epoch_acc[-1]
  def get_actual_epoch_validation_loss(self): return self.validation_epoch_loss[-1]
  def get_train_epoch_loss(self): return self.train_epoch_loss
  def get_validation_epoch_acc(self): return self.validation_epoch_acc
  def get_best_epoch(self): return self.best_epoch
  def get_best_epoch_validation_acc(self): return self.best_epoch_validation_acc
  def get_best_epoch_validation_loss(self): return self.best_epoch_validation_loss
  def get_best_epoch_y_target_and_y_prediction(self): return self.best_epoch_y_target, self.best_epoch_y_predicted


if __name__ == "__main__":
  """
  score handler
  """

  # score handler
  score_handler = ScoreHandler(classes=np.arange(4))

  # labels
  y_target = [0, 1, 0, 0]
  y_predicted = [1, 0, 1, 0]

  # test
  score_handler.update_train(0.0)
  score_handler.update_validation(y_target, y_predicted, loss=0.0)
  score_handler.epoch_finished(epoch=0, print_scores=True)
  score_handler.info()