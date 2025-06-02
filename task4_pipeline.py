# --
# task 4: pipeline

# In this task you will create a machine learning pipeline to train and evaluate different models.
# This pipeline is controlled via a config dictionary, were the datamodule, model, and other parameters are stored.
# Fill the sections with # *** your code her *** #

import yaml
import numpy as np
import torch
import importlib
import torchvision
from torchvision.transforms import v2
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

# local packages
from plots import plot_scores, plot_validationresults
from score_handler import ScoreHandler
from feature_extraction import extract_hog_features
from task2_datamodule import augment_data


def pipeline(cfg, fp_print=print):
  """
  machine learning pipeline, the whole pipeline is controlled via the config dictionary (cfg)
  """

  # create results path
  if not Path(cfg['result_path']).is_dir(): Path(cfg['result_path']).mkdir(parents=True)

  # datamodule
  datamodule_class = getattr(importlib.import_module(cfg['datamodule']['module']), cfg['datamodule']['attr'])
  datamodule = datamodule_class(*cfg['datamodule']['args'], **cfg['datamodule']['kwargs'])

  # model class
  model_class = getattr(importlib.import_module(cfg['model']['module']), cfg['model']['attr'])

  # model class type dependencies
  is_no_incremental_model = model_class.__name__ == 'SVMModel' or model_class.__name__ == 'RandomForestModel'
  if is_no_incremental_model: cfg['training']['use_extracting_hog'] = True

  # sample input shape
  input_shape = extract_hog_features(cfg, next(iter(datamodule)), model_class=model_class)[0].shape

  # config updates, make sure that the config is correct and depends on the datamodule
  cfg['model']['kwargs'].update({'input_shape': list(input_shape), 'num_classes': len(datamodule.classes)})
  original_model_file_name_stem = Path(cfg['model']['args'][0]['model_file_name']).stem

  # make a copy of the config
  yaml.dump(cfg, open(Path(cfg['result_path']) / 'config_copy.yaml', 'w'))

  # cross-validation
  kfold_cross_validator = StratifiedKFold(n_splits=3, shuffle=True)

  # kfold scores, collect for overall score
  kfold_scores = []

  # info messages
  fp_print("# --\n# Datamodule: {}\n# Classes: {}\n# Input shape: {}".format(datamodule_class.__name__, datamodule.classes, input_shape))
  fp_print("# Selected model: {}".format(model_class.__name__))
  fp_print("# Use HOG features: {}".format(cfg['training']['use_extracting_hog']))

  # cross validation
  for kfold_id, (train_idx, test_idx) in enumerate(kfold_cross_validator.split(datamodule, y=datamodule.targets)):

    # kfold split info
    fp_print("\n# --\n# Kfold id: ", kfold_id)

    # filename of the model with fold id
    model_file_name = '{}_fold{}.pth'.format(original_model_file_name_stem, kfold_id)

    # score handler
    score_handler = ScoreHandler(classes=datamodule.classes)

    # model init
    model = model_class(*cfg['model']['args'], **{**cfg['model']['kwargs'], **{'model_file_name': model_file_name}})

    # dataloader kwargs
    dl_kwargs_train = {**cfg['training']['dataloader_kwargs'], **dict({'batch_size': len(train_idx)} if is_no_incremental_model else {})} 
    dl_kwargs_test = {**cfg['training']['dataloader_kwargs'], **dict({'batch_size': len(test_idx)} if is_no_incremental_model else {})} 

    # dataloaders
    train_loader = torch.utils.data.DataLoader(datamodule, sampler=torch.utils.data.SubsetRandomSampler(train_idx), **dl_kwargs_train)
    test_loader = torch.utils.data.DataLoader(datamodule, sampler=torch.utils.data.SubsetRandomSampler(test_idx), **dl_kwargs_test)

    # info message
    fp_print("Training starts...")

    # epochs
    for epoch in range(cfg['training']['num_epochs'] if not is_no_incremental_model else 1):

      # set to train
      model.train()

      # train loader
      for data in train_loader: 


        # ***
        # your code here

        raise NotImplementedError()

        # add data augmentation with augment_data use the number of augmentations from the config file ['training']
        # extract hog features with extract_hog_features
        # do a training step of the model and save its loss
        # save the loss in the score handler .update_train(loss)

        # augment data
        data = None

        # hog features
        data = None

        # train step of the model
        loss = None

        # loss update
        #score_handler...

        # 
        # ***


      # evaluation mode
      model.eval()

      # test loader
      for data in test_loader: 

        # hog features
        data = extract_hog_features(cfg, data, model_class)

        # test data
        y_pred, loss = model.validation_step(data)

        # update score handler
        score_handler.update_validation(y_target=data[1], y_predicted=y_pred, loss=loss)


        # ***
        # your code here

        # just comment this, so the training runs through

        # debug plot
        plot_validationresults(data, y_pred, classes=datamodule.classes)

        # 
        # ***


      # score handler
      score_handler.epoch_finished(epoch, print_scores=True)

      # training update
      model.training_update(epoch, validation_loss=score_handler.get_actual_epoch_validation_loss())

      # early stopping
      if model.early_stopping_criteria(): 
        print("...early stopping!")
        break

    # plots
    plot_scores(score_handler, kfold_id=kfold_id, classes=datamodule.classes, model_class=model_class, save_path=model.get_save_path(), show_plot=cfg['training']['show_plots'])
      
    # score handler
    score_handler.save(save_path=model.get_save_path(), name_addon='_kfold-{}'.format(kfold_id))

    # collect scores for overall score
    kfold_scores.append(score_handler.get_score_summary_dict())

    # save model
    if not model.early_stopping_criteria(): model.save()

    # model analytics
    model.analytics(show_plot=cfg['training']['show_plots'])

  # overall score dict
  overall_score_dict = {'fold_scores': {'fold{}'.format(i): kfold_scores[i] for i in range(len(kfold_scores))}, 'overall_accuracy': np.mean([kfold_score['acc'] for kfold_score in kfold_scores]).item()}

  # overall scores of all folds
  fp_print('\n# --\n# End of Pipeline\n# Overall Scores:')
  [fp_print('{}: {}'.format(k, v)) for k, v in overall_score_dict.items()]

  # write overall scores to file
  yaml.dump(overall_score_dict, open(str(Path(model.get_save_path()) / 'overall_model_performance.yaml'), 'w'))

  return overall_score_dict['overall_accuracy']



if __name__ == '__main__':
  """
  task 4: pipeline
  """

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # start the pipeline
  overall_acc = pipeline(cfg['task4'])

  print("Everything successful!")