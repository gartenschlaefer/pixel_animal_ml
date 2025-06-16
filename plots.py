# --
# plots

# Some advanced plots to visualize your data

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path


def get_colormap_from_context(context='none', num_lin_samples=None):
  """
  my colormaps with context
  """
  from palettable.cubehelix import purple_16
  from matplotlib.colors import ListedColormap
  if context == 'confusion': return ListedColormap(purple_16.mpl_colormap.reversed()(np.linspace(0, 0.8, 25 if num_lin_samples is None else int(num_lin_samples))), name='purpletonian')
  return None


def plot_2dmatrix(data, n_padding=2, add_colorbar=False, show_plot=True, save_path=None, **kwargs):
  """
  plot 2d matrix, data: sample x m x n x (c)
  """

  # to numpy
  data = np.array(data)

  # squeeze
  if data.shape[-1] == 1: data = np.squeeze(data, axis=-1)
  if data.shape[1] == 1: data = np.squeeze(data, axis=1)

  # transpose if required
  data = np.array([d.transpose(1, 2, 0) if d.shape[0] == 3 and len(data.shape) == 4 else d for d in data])

  # rgb
  is_rgb = len(data.shape) == 4 and (data.shape[-1] == 3 or data.shape[-1] == 4)

  # matrix specs
  num_rows = kwargs.get('num_rows')
  num_rows = len(data) if num_rows is None else num_rows
  num_rows = len(data) if num_rows > len(data) else num_rows
  num_cols = len(data) // num_rows + int(bool(len(data) % num_rows))
  tile_size = (2 * n_padding + data.shape[1], 2 * n_padding + data.shape[2])

  # create matrix
  matrix = np.zeros((num_rows * tile_size[0], num_cols * tile_size[1]) + (() if not is_rgb else (data.shape[-1], )), dtype=data.dtype)

  # figure
  fig = plt.figure(figsize=[12, 6])

  # create axis
  ax = fig.add_subplot(1, 1, 1)

  # create matrix
  for i, d in enumerate(data):

    # get row index
    row_pos = (i % num_rows) * tile_size[0] + n_padding
    col_pos = (i // num_rows) * tile_size[1] + n_padding

    # copy data
    matrix[row_pos:row_pos+d.shape[0], col_pos:col_pos+d.shape[1]] = d

    # check if there are labels
    if kwargs.get('sample_labels') is None: continue
    if len(kwargs['sample_labels']) <= i: continue
    sample_color = 'white' if kwargs.get('sample_colors') is None else ('white' if len(kwargs['sample_colors']) <= i else kwargs['sample_colors'][i])

    # add text
    ax.text(col_pos, row_pos, "{}".format(kwargs['sample_labels'][i]), color=sample_color, bbox=dict(facecolor='black', pad=1, alpha=0.75)) 

  # kwargs
  imshow_kwargs = {'aspect': 'auto', 'interpolation': 'none', 'cmap': None, 'vmax': None, 'vmin': None }
  [imshow_kwargs.update({k: v}) for k, v in kwargs.items() if k in imshow_kwargs.keys()]

  # image
  im = ax.imshow(matrix, **imshow_kwargs)

  # axis settings
  ax.set_axis_off()

  # title
  if not kwargs.get('title') is None: ax.set_title(kwargs.get('title'))

  # add colorbar
  if add_colorbar:

    # devider for cax
    cax = make_axes_locatable(plt.gca()).append_axes('right', size='2%', pad='2%')

    # colorbar
    color_bar = fig.colorbar(im, cax=cax)

  # save
  if save_path: fig.savefig(save_path, dpi=100)

  # show plot
  if show_plot: plt.show()
  else: plt.close()


def to_short_class_name(y, classes):
  """
  short class name
  """
  class_name = classes[y]
  if class_name == 'dog': return 'dog'
  if class_name == 'horse': return 'hor'
  if class_name == 'frog': return 'fro'
  return class_name


def plot_validationresults(data, y_predicted, classes, num_rows=6, **kwargs):
  """
  validation results
  """

  # to numpy
  y_target = np.array(data[1])
  y_predicted = np.array(y_predicted)

  # sample labels
  sample_labels = ['{}|{}'.format(to_short_class_name(y_t, classes), to_short_class_name(y_p, classes)) for y_p, y_t in zip(y_predicted, y_target)]
  sample_colors = ['green' if y_p == y_t else 'red' for y_p, y_t in zip(y_predicted, y_target)]
  title = 'Validation Data with acc: [{:.2}]'.format(np.mean(y_predicted == y_target))

  # plot
  plot_2dmatrix(data[0], title=title, aspect='equal', num_rows=num_rows, sample_labels=sample_labels, sample_colors=sample_colors, **kwargs)


def plot_confusionmatrix(data, kfold_id, classes=None, accuracy=None, model_class=None, save_path=None, show_plot=False, **kwargs):
  """
  plot confusion matrix
  """

  # packages
  import sklearn

  # get data
  y_target = np.squeeze(data[0])
  y_predicted = np.squeeze(data[1])

  # classes
  classes = classes if not classes is None and not len(np.unique([y_target, y_predicted])) != len(classes) else np.unique([y_target, y_predicted])

  # confusion matrix
  confusion_matrix = sklearn.metrics.confusion_matrix(y_target, y_predicted)

  # max value
  max_value = int(np.max(np.sum(confusion_matrix, axis=1)))

  # figure
  fig = plt.figure(figsize=[12, 6])

  # create axis
  ax = fig.add_subplot(1, 1, 1)

  # kwargs
  imshow_kwargs = {'origin': None, 'aspect': 'auto', 'interpolation': 'none', 'cmap': get_colormap_from_context(context='confusion', num_lin_samples=max_value + 1), 'vmax': max_value, 'vmin': 0}
  [imshow_kwargs.update({k: v}) for k, v in kwargs.items() if k in imshow_kwargs.keys()]

  # image
  im = ax.imshow(confusion_matrix, **imshow_kwargs)

  # text handling
  for y_pred_pos in range(len(classes)):
    for y_true_pos in range(len(classes)):

      # font color and size
      font_color = (('black' if confusion_matrix[y_pred_pos, y_true_pos] < 0.25 * max_value else 'white') if confusion_matrix[y_pred_pos, y_true_pos] else 'white') if y_pred_pos != y_true_pos else ('black' if confusion_matrix[y_pred_pos, y_true_pos] < 0.25 * max_value else 'white')
      fontsize = 8 if len(classes) > 10 else 11
      bbox = None

      # write numbers inside
      ax.text(y_true_pos, y_pred_pos, confusion_matrix[y_pred_pos, y_true_pos], ha='center', va='center', color=font_color, fontsize=fontsize, bbox=bbox)

  # care about labels
  ax.set_title('Confusion Matrix of kfold: {}, acc: [{:.4}] with model: {}'.format(kfold_id, accuracy, model_class.__name__))
  ax.set_xticks(np.arange(len(classes)))
  ax.set_yticks(np.arange(len(classes)))
  ax.set_xticklabels(classes)
  ax.set_yticklabels(classes)
  ax.set_xlabel('Predicted Labels', fontsize=13)
  ax.set_ylabel('True Labels', fontsize=13)

  # devider for cax
  cax = make_axes_locatable(plt.gca()).append_axes('right', size='2%', pad='2%')

  # colorbar
  color_bar = fig.colorbar(im, cax=cax)

  # save
  if save_path: fig.savefig(Path(save_path) / 'confusion_matrix_kfold-{}.png'.format(kfold_id), dpi=100)

  # show plot
  if show_plot: plt.show()
  else: plt.close()


def plot_scores(score_handler, kfold_id, classes, model_class, save_path=None, show_plot=True):
  """
  plot scores
  """

  # confusion matrix of last epoch
  plot_confusionmatrix(score_handler.get_best_epoch_y_target_and_y_prediction(), kfold_id=kfold_id, classes=classes, accuracy=score_handler.get_best_epoch_validation_acc(), model_class=model_class, save_path=save_path, show_plot=show_plot)

  # no epoch scores
  if len(score_handler.get_train_epoch_loss()) <= 1: return

  # figure
  fig = plt.figure(figsize=[12, 8])
  fig.subplots_adjust(hspace=0.5)

  # train loss
  ax = fig.add_subplot(2, 1, 1)
  ax.plot(score_handler.get_train_epoch_loss())
  ax.scatter(score_handler.get_best_epoch(), score_handler.get_train_epoch_loss()[score_handler.get_best_epoch()], marker='*', s=80, color='purple', label='best epoch')
  ax.set_title('Train Loss of kfold: {} with model: {}'.format(kfold_id, model_class.__name__))
  ax.set_ylabel('Mean Loss')
  ax.set_xlabel('Epochs')
  ax.legend()
  ax.grid()

  # validation acc
  ax = fig.add_subplot(2, 1, 2)
  ax.plot(score_handler.get_validation_epoch_acc())
  ax.scatter(score_handler.get_best_epoch(), score_handler.get_validation_epoch_acc()[score_handler.get_best_epoch()], marker='*', s=80, color='purple', label='best epoch')
  ax.set_title('Validation accuracy of kfold: {} with model: {}'.format(kfold_id, model_class.__name__))
  ax.set_ylabel('Accuracy')
  ax.set_xlabel('Epochs')
  ax.set_ylim([0.0, 1.0])
  ax.legend()
  ax.grid()

  # save
  if save_path: fig.savefig(Path(save_path) / 'training_scores_kfold-{}.png'.format(kfold_id), dpi=100)

  # show plot
  if show_plot: plt.show()
  else: plt.close()