# --
# task 5: experiments

# In this task we run the experiments defined in the experiments folder.
# For each experiments we have to print the scores and the training results so that we can review and compare them afterwards.
# In this task you do not have to code but to write config files with the requested structure of the pipeline.
# Create following experiments:
# - cnn model from task3 without augmentation
# - cnn model from task3 with augmentation
# - cnn model from task3 with hog features
# - svm model
# - random forest model
# Note: use the example experiment file in '/experiments/exp_random_model.yaml'

import re
import yaml
import logging
import datetime

from pathlib import Path
from task4_pipeline import pipeline


def prepare_logger(logger, filename='results.log'):
  """
  change logger, e.g. to new path
  """

  # clean up handlers
  for handler in logger.handlers:
    logger.removeHandler(handler)

  # define new handler
  handler = logging.FileHandler(filename, mode='w')
  handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
  logger.addHandler(handler)

  return handler


def run_experiments(experiment_path, result_path, re_experiment_filter=r'.'):
  """
  run experiments
  """

  # custom logger
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.INFO)

  # print and log function
  print_and_log = lambda  *x: (print(*x), logger.info(''.join([str(xi) for xi in x])))

  # get config files
  config_files = sorted(list(Path(experiment_path).glob('*.yaml')))

  # filter config files
  config_files = [f for f in config_files if re.search(re_experiment_filter, Path(f).stem)]

  # collected scores
  collected_scores = {}

  # run through each config file
  for config_file in config_files:

    # yaml config file
    cfg_experiment = yaml.safe_load(open(config_file))

    # experiment result path
    experiment_result_path = str(Path(result_path) / Path(config_file).stem)

    # update experiment config
    cfg_experiment.update({'result_path': experiment_result_path})
    cfg_experiment['model']['kwargs'].update({'save_path': experiment_result_path})

    # create results path
    if not Path(cfg_experiment['result_path']).is_dir(): Path(cfg_experiment['result_path']).mkdir(parents=True)

    # change logger
    prepare_logger(logger, filename=Path(cfg_experiment['result_path']) / 'train.log')

    # info message
    print_and_log("\n# --\n# New Experiment with Config File: ", config_file)
    print_and_log("# Results are stored in: ", cfg_experiment['result_path'])

    # run pipeline
    total_acc = pipeline(cfg_experiment, fp_print=print_and_log)

    # collected scores
    collected_scores.update({str(config_file): total_acc})

  # print and dump scores
  print("\n# --\n# Overall Scores of the experiments:")
  [print("{}: {:.6}".format(k, v)) for k, v in collected_scores.items()]
  yaml.dump({str(datetime.datetime.now()): collected_scores}, open(Path(result_path) / 'overall_results.yaml', 'a'))



if __name__ == '__main__':
  """
  task 5: experiments
  """

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # run experiments
  run_experiments(cfg['task5']['experiments_path'], cfg['task5']['result_path'], re_experiment_filter='.')

  print("Everything successful!")