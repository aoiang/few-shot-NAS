##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
##################################################################
# Regularized Evolution for Image Classifier Architecture Search #
##################################################################
import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets     import get_datasets, SearchDataset
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from nas_201_api  import NASBench201API as API
from models       import CellStructure, get_search_spaces
import json


class Model(object):

  def __init__(self):
    self.arch = None
    self.sim_acc = None
    self.accuracy = None
    
  def __str__(self):
    """Prints a readable version of this bitstring."""
    return '{:}'.format(self.arch)
  

# This function is to mimic the training and evaluatinig procedure for a single architecture `arch`.
# The time_cost is calculated as the total training time for a few (e.g., 12 epochs) plus the evaluation time for one epoch.
# For use_012_epoch_training = True, the architecture is trained for 12 epochs, with LR being decaded from 0.1 to 0.
#       In this case, the LR schedular is converged.
# For use_012_epoch_training = False, the architecture is planed to be trained for 200 epochs, but we early stop its procedure.
#       

def train_and_eval(arch, nasbench_supernet, nas_bench):


  arch_list = arch.tolist(remove_str=None)[0]
  sim_acc = nasbench_supernet[str(arch_list)]
  real_acc = nas_bench[str(arch_list)]
  time_cost = random.random()

  return sim_acc, real_acc, time_cost

def random_architecture_func(max_nodes, op_names):
  # return a random architecture
  def random_architecture():
    genotypes = []
    for i in range(1, max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        op_name  = random.choice( op_names )
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return CellStructure( genotypes )
  return random_architecture


def mutate_arch_func(op_names):
  """Computes the architecture for a child of the given parent architecture.
  The parent architecture is cloned and mutated to produce the child architecture. The child architecture is mutated by randomly switch one operation to another.
  """
  def mutate_arch_func(parent_arch):
    child_arch = deepcopy( parent_arch )
    node_id = random.randint(0, len(child_arch.nodes)-1)
    node_info = list( child_arch.nodes[node_id] )
    snode_id = random.randint(0, len(node_info)-1)
    xop = random.choice( op_names )
    while xop == node_info[snode_id][0]:
      xop = random.choice( op_names )
    node_info[snode_id] = (xop, node_info[snode_id][1])
    child_arch.nodes[node_id] = tuple( node_info )
    return child_arch
  return mutate_arch_func


def regularized_evolution(cycles, population_size, sample_size, time_budget, random_arch, mutate_arch, nas_bench,
                          nasbench_supernet):
  """Algorithm for regularized evolution (i.e. aging evolution).
  
  Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
  Classifier Architecture Search".

  Args:
    cycles: the number of cycles the algorithm should run for.
    population_size: the number of individuals to keep in the population.
    sample_size: the number of individuals that should participate in each tournament.
    time_budget: the upper bound of searching cost

  Returns:
    history: a list of `Model` instances, representing all the models computed
        during the evolution experiment.
  """
  population = collections.deque()
  history, total_time_cost = [], 0  # Not used by the algorithm, only used to report results.
  cur_best_acc = 0.0
  record = []
  # Initialize the population with random models.
  searched_arch = []
  while len(population) < population_size:
    model = Model()
    model.arch = random_arch()
    model.sim_acc, model.accuracy, time_cost = train_and_eval(model.arch, nasbench_supernet, nas_bench)
    population.append(model)
    history.append(model)
    searched_arch.append(model.arch)
    total_time_cost += time_cost

    best_model = max(history, key=lambda i: i.accuracy)
    if best_model.accuracy > cur_best_acc:
      cur_best_acc = best_model.accuracy
      record.append([len(history), cur_best_acc])


  # Carry out evolution in cycles. Each cycle produces a model and removes
  # another.
  while len(history) < cycles:
  # while total_time_cost < time_budget:
    # Sample randomly chosen models from the current population.
    start_time, sample = time.time(), []
    while len(sample) < sample_size:
      # Inefficient, but written this way for clarity. In the case of neural
      # nets, the efficiency of this line is irrelevant because training neural
      # nets is the rate-determining step.
      candidate = random.choice(list(population))
      sample.append(candidate)

    # The parent is the best model in the sample.
    parent = max(sample, key=lambda i: i.sim_acc)


    # Create the child model and store it.
    child = Model()
    child.arch = mutate_arch(parent.arch)
    total_time_cost += time.time() - start_time
    child.sim_acc, child.accuracy, time_cost = train_and_eval(child.arch, nasbench_supernet, nas_bench)
    population.append(child)
    if child.arch not in searched_arch:
      searched_arch.append(child.arch)
      history.append(child)

    best_model = max(history, key=lambda i: i.accuracy)
    if best_model.accuracy > cur_best_acc:
      cur_best_acc = best_model.accuracy
      record.append([len(history), cur_best_acc])
    if cur_best_acc == 94.37:
      break

    # Remove the oldest model.
    population.popleft()
    # with open('/home/yiyangzhao/NAS-Bench-201/nasbench-201-search/REA/results', 'w') as f:
    #   f.write(str(record))
    print(record)
  return history, total_time_cost, record


def main(xargs, nas_bench, nasbench_supernet):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( xargs.workers )
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)


  search_space = get_search_spaces('cell', xargs.search_space_name)
  random_arch = random_architecture_func(xargs.max_nodes, search_space)
  mutate_arch = mutate_arch_func(search_space)
  #x =random_arch() ; y = mutate_arch(x)
  x_start_time = time.time()
  # logger.log('{:} use nas_bench : {:}'.format(time_string(), nas_bench))
  logger.log('-'*30 + ' start searching with the time budget of {:} s'.format(xargs.time_budget))
  history, total_cost, record = regularized_evolution(xargs.ea_cycles, xargs.ea_population, xargs.ea_sample_size, xargs.time_budget, random_arch, mutate_arch, nas_bench, nasbench_supernet)
  logger.log('{:} regularized_evolution finish with history of {:} arch with {:.1f} s (real-cost={:.2f} s).'.format(time_string(), len(history), total_cost, time.time()-x_start_time))

  with open(xargs.save_dir + '/results.txt', 'a') as f:
    f.write(str(record))
    f.write('\n')

  


if __name__ == '__main__':
  parser = argparse.ArgumentParser("Regularized Evolution Algorithm")
  parser.add_argument('--data_path',          type=str,   help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  # channels and number-of-cells
  parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   help='The number of cells in one stage.')
  parser.add_argument('--ea_cycles',          type=int,   help='The number of cycles in EA.')
  parser.add_argument('--ea_population',      type=int,   help='The population size in EA.')
  parser.add_argument('--ea_sample_size',     type=int,   help='The sample size in EA.')
  parser.add_argument('--ea_fast_by_api',     type=int,   help='Use our API to speed up the experiments or not.')
  parser.add_argument('--time_budget',        type=int,   help='The total time cost budge for searching (in seconds).')
  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--arch_nas_dataset',   type=str,   help='The path to load the architecture dataset (tiny-nas-benchmark).')
  parser.add_argument('--arch_supernet_dataset', type=str,
                      help='The path to load the supernet.')
  parser.add_argument('--print_freq',         type=int,   help='print frequency (default: 200)')
  parser.add_argument('--rand_seed',          type=int,   default=-1,   help='manual seed')
  args = parser.parse_args()
  #if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  args.ea_fast_by_api = args.ea_fast_by_api > 0



  with open(args.arch_nas_dataset, 'r') as f:
    nas_bench = json.load(f)
  with open(args.arch_supernet_dataset, 'r') as f:
    nasbench_supernet = json.load(f)
  if args.rand_seed < 0:
    save_dir, all_indexes, num = None, [], 50
    for i in range(num):
      print ('{:} : {:03d}/{:03d}'.format(time_string(), i, num))
      args.rand_seed = random.randint(1, 100000)
      main(args, nas_bench, nasbench_supernet)

  else:
    main(args, nas_bench, nasbench_supernet)
