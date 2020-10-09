##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
#####################################################################################################
# modified from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py #
#####################################################################################################
import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
from torch.distributions import Categorical
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



class Policy(nn.Module):

  def __init__(self, max_nodes, search_space):
    super(Policy, self).__init__()
    self.max_nodes    = max_nodes
    self.search_space = deepcopy(search_space)
    self.edge2index   = {}
    for i in range(1, max_nodes):
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        self.edge2index[ node_str ] = len(self.edge2index)
    self.arch_parameters = nn.Parameter( 1e-3*torch.randn(len(self.edge2index), len(search_space)) )

  def generate_arch(self, actions):
    genotypes = []
    for i in range(1, self.max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        op_name  = self.search_space[ actions[ self.edge2index[ node_str ] ] ]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return CellStructure( genotypes )

  def genotype(self):
    genotypes = []
    for i in range(1, self.max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        with torch.no_grad():
          weights = self.arch_parameters[ self.edge2index[node_str] ]
          op_name = self.search_space[ weights.argmax().item() ]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return CellStructure( genotypes )
    
  def forward(self):
    alphas  = nn.functional.softmax(self.arch_parameters, dim=-1)
    return alphas


class ExponentialMovingAverage(object):
  """Class that maintains an exponential moving average."""

  def __init__(self, momentum):
    self._numerator   = 0
    self._denominator = 0
    self._momentum    = momentum

  def update(self, value):
    self._numerator = self._momentum * self._numerator + (1 - self._momentum) * value
    self._denominator = self._momentum * self._denominator + (1 - self._momentum)

  def value(self):
    """Return the current value of the moving average"""
    return self._numerator / self._denominator


def select_action(policy):
  probs = policy()
  m = Categorical(probs)
  action = m.sample()
  #policy.saved_log_probs.append(m.log_prob(action))
  return m.log_prob(action), action.cpu().tolist()


def train_and_eval(arch, nasbench_supernet, nas_bench):

  arch_list = arch.tolist(remove_str=None)[0]
  sim_acc = nasbench_supernet[str(arch_list)]
  real_acc = nas_bench[str(arch_list)]
  time_cost = random.random()

  return sim_acc, real_acc, time_cost

def main(xargs, nas_bench, nasbench_supernet):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( xargs.workers )
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)
  record = []
  
  search_space = get_search_spaces('cell', xargs.search_space_name)
  policy    = Policy(xargs.max_nodes, search_space)
  optimizer = torch.optim.Adam(policy.parameters(), lr=xargs.learning_rate)
  #optimizer = torch.optim.SGD(policy.parameters(), lr=xargs.learning_rate)
  eps       = np.finfo(np.float32).eps.item()
  baseline  = ExponentialMovingAverage(xargs.EMA_momentum)
  logger.log('policy    : {:}'.format(policy))
  logger.log('optimizer : {:}'.format(optimizer))
  logger.log('eps       : {:}'.format(eps))

  # nas dataset load
  # logger.log('{:} use nas_bench : {:}'.format(time_string(), nas_bench))
  cur_best_acc = 0.0

  # REINFORCE
  # attempts = 0
  x_start_time = time.time()
  logger.log('Will start searching with time budget of {:} s.'.format(xargs.time_budget))
  total_steps, total_costs, trace = 0, 0, []
  for istep in range(xargs.RL_steps):
  # while total_costs < xargs.time_budget:
    start_time = time.time()
    log_prob, action = select_action( policy )
    arch   = policy.generate_arch( action )
    reward, acc, cost_time = train_and_eval(arch, nasbench_supernet, nas_bench)
    trace.append( (reward, arch) )
    if acc > cur_best_acc:
      cur_best_acc = acc
      record.append([len(trace), cur_best_acc])

    if cur_best_acc == 94.37:
      break

    baseline.update(reward)
    # calculate loss
    policy_loss = ( -log_prob * (reward - baseline.value()) ).sum()
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    # accumulate time
    total_costs += time.time() - start_time
    total_steps += 1
    # logger.log('step [{:3d}] : average-reward={:.3f} : policy_loss={:.4f} : {:}'.format(total_steps, baseline.value(), policy_loss.item(), policy.genotype()))
    print(record)



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
  parser.add_argument('--learning_rate',      type=float, help='The learning rate for REINFORCE.')
  parser.add_argument('--RL_steps',           type=int,   help='The steps for REINFORCE.')
  parser.add_argument('--EMA_momentum',       type=float, help='The momentum value for EMA.')
  parser.add_argument('--time_budget',        type=int,   help='The total time cost budge for searching (in seconds).')
  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--arch_nas_dataset', type=str,
                      help='The path to load the architecture dataset (tiny-nas-benchmark).')
  parser.add_argument('--arch_supernet_dataset', type=str,
                      help='The path to load the supernet.')
  parser.add_argument('--print_freq',         type=int,   help='print frequency (default: 200)')
  parser.add_argument('--rand_seed',          type=int,   default=-1,   help='manual seed')
  args = parser.parse_args()
  #if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
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
