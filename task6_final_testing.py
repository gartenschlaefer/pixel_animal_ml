# --
# task 6: final testing

# In this task we will load each model from the experiments and test them on the created test dataset.
# Fill the sections with # *** your code her *** #

import yaml
import numpy as np
import torch
import importlib
from pathlib import Path
from task2_datamodule import PixelAnimalDatamodule
from task2_datamodule import augment_data
from feature_extraction import extract_hog_features
from plots import plot_validationresults


def test_models(cfg):
  """
  test models
  """

  # final path
  if not Path(cfg['task6']['final_path']).is_dir(): Path(cfg['task6']['final_path']).mkdir(parents=True)

  # get all overall scores
  score_files = sorted(list(Path(cfg['task5']['result_path']).glob('**/overall_model_performance.yaml')))

  # get all scores
  all_score_dict = {score_file.parent.stem: yaml.safe_load(open(score_file)) for score_file in score_files}
  all_score_overall_accuracy_dict = {exp_key: {k: v for k, v in exp_score_dict.items() if k == 'overall_accuracy'} for exp_key, exp_score_dict in all_score_dict.items()}

  # save all scores in one file
  yaml.dump(all_score_dict, open(Path(cfg['task6']['final_path']) / 'all_scores.yaml', 'w'))
  yaml.dump(all_score_overall_accuracy_dict, open(Path(cfg['task6']['final_path']) / 'all_scores_overall_accuracy.yaml', 'w'))
  print('Experiment Validation Scores:'), [print('{} with mean acc: [{:.4}]'.format(k, v['overall_accuracy'])) for k, v in all_score_overall_accuracy_dict.items()]


  # ***
  # your code here

  raise NotImplementedError()

  # define the test datamodule (from task 2: PixelAnimalDatamodule)
  datamodule = None

  # 
  # ***


  # model file
  model_files = sorted(list(Path(cfg['task5']['result_path']).glob('**/*.pth')))

  # test
  model_test_eval_dict = {}

  # test each model
  for model_file in model_files:

    # get 
    cfg_copy = yaml.safe_load(open(model_file.parent / 'config_copy.yaml'))

    # model class
    model_class = getattr(importlib.import_module(cfg_copy['model']['module']), cfg_copy['model']['attr'])

    # model init
    model = model_class(*cfg_copy['model']['args'], **cfg_copy['model']['kwargs'])


    # ***
    # your code here

    raise NotImplementedError()

    # load the model_file (model load function)

    # create a dataloader with a large batch_size 
    dataloader = None

    # get all data with one next(iter(dataloader)) call
    data = None

    # augment the test data with 3 augmentations
    data = None

    # call extract_hog_features with the target config
    data_feat = None

    # make a model prediction on data_feat (model predict function)
    y_pred = None

    #
    # ***

    # validation
    plot_validationresults(data, y_pred, classes=datamodule.classes, num_rows=3, save_path=Path(cfg['task6']['final_path']) / (model_file.parent.stem + '_' + model_file.stem + '.png'), show_plot=False)

    # evaluation scores
    model_test_eval_dict.update({str(model_file): {'acc': round(np.mean(y_pred == data[1].numpy()).item(), ndigits=4)}})

  # print
  print("\n# --\n# Model Eval on PixelAnimals: "), [print('{} with acc: [{:.4}]'.format(k, v['acc'])) for k, v in model_test_eval_dict.items()]

  # save scores
  yaml.dump(model_test_eval_dict, open(Path(cfg['task6']['final_path']) / 'model_eval.yaml', 'w'))



if __name__ == '__main__':
  """
  task 6: final testing
  """

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # test models
  test_models(cfg)

  print("Everything successful!")