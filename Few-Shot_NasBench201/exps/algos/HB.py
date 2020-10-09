##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
###################################################################
# BOHB: Robust and Efficient Hyperparameter Optimization at Scale #
# required to install hpbandster ##################################
# bash ./scripts-search/algos/BOHB.sh -1         ##################
###################################################################
import os, sys, time, random, argparse
from copy import deepcopy
from pathlib import Path
import torch
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config
from datasets     import get_datasets, SearchDataset
from procedures   import prepare_seed, prepare_logger
from log_utils    import AverageMeter, time_string, convert_secs2time
from nas_201_api  import NASBench201API as API
from models       import CellStructure, get_search_spaces
# BOHB: Robust and Efficient Hyperparameter Optimization at Scale, ICML 2018
import ConfigSpace
from hpbandster.optimizers.bohb import BOHB
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers.hyperband import HyperBand
import json



def get_configuration_space(max_nodes, search_space):
  cs = ConfigSpace.ConfigurationSpace()
  #edge2index   = {}
  for i in range(1, max_nodes):
    for j in range(i):
      node_str = '{:}<-{:}'.format(i, j)
      cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(node_str, search_space))
  return cs


def config2structure_func(max_nodes):
  def config2structure(config):
    genotypes = []
    for i in range(1, max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        op_name = config[node_str]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return CellStructure( genotypes )
  return config2structure


class MyWorker(Worker):

  def __init__(self, *args, convert_func=None, dataname=None, nas_bench=None, nasbench_supernet=None, time_budget=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.convert_func   = convert_func
    self._dataname      = dataname
    self._nas_bench     = nas_bench
    self.nasbench_supernet =  nasbench_supernet
    self.time_budget    = time_budget
    self.seen_archs     = []
    self.sim_cost_time  = 0
    self.real_cost_time = 0
    self.is_end         = False
    self.step = 1
    self.best_acc = 0.0
    self.record = []
    with open('/home/yiyangzhao/NAS-Bench-201/nasbench-201-search/nasbench201', 'r') as f:
      self.nasbench201 = json.load(f)

  def get_the_best(self):
    assert len(self.seen_archs) > 0
    best_index, best_acc = -1, None
    for arch_index in self.seen_archs:
      print('current index is', arch_index)
      info = self._nas_bench.get_more_info(arch_index, self._dataname, None, True, True)
      vacc = info['valid-accuracy']
      if best_acc is None or best_acc < vacc:
        best_acc = vacc
        best_index = arch_index
    assert best_index != -1
    return best_index

  def compute(self, config, budget, **kwargs):
    start_time = time.time()
    structure  = self.convert_func( config )
    structure_list = structure.tolist(remove_str=None)[0]
    arch_index = self._nas_bench.query_index_by_arch( structure )




    # cur_vacc = self.nasbench201[str(structure_list)]
    real_acc = self.nasbench201[str(structure_list)]
    cur_vacc = self.nasbench_supernet[str(structure_list)]
    self.real_cost_time += (time.time() - start_time)

    if len(self.seen_archs) <= 3000 and not self.is_end:
      self.seen_archs.append( arch_index )

      if real_acc > self.best_acc:
        self.best_acc = real_acc
        self.record.append([len(self.seen_archs), self.best_acc])

      return ({'loss': 100 - float(cur_vacc),
               'info': {'seen-arch'     : len(self.seen_archs),
                        'sim-test-time' : self.sim_cost_time,
                        'current-arch'  : arch_index}
            })
    else:
      self.is_end = True
      return ({'loss': 100,
               'info': {'seen-arch'     : len(self.seen_archs),
                        'sim-test-time' : self.sim_cost_time,
                        'current-arch'  : None}
            })


def main(xargs, nas_bench, nasbench_supernet, runs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( xargs.workers )
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)


  # nas dataset load
  assert xargs.arch_nas_dataset is not None and os.path.isfile(xargs.arch_nas_dataset)
  search_space = get_search_spaces('cell', xargs.search_space_name)
  cs = get_configuration_space(xargs.max_nodes, search_space)

  config2structure = config2structure_func(xargs.max_nodes)
  hb_run_id = '0'

  NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
  ns_host, ns_port = NS.start()
  num_workers = 1

  #nas_bench = AANASBenchAPI(xargs.arch_nas_dataset)
  #logger.log('{:} Create NAS-BENCH-API DONE'.format(time_string()))
  workers = []
  for i in range(num_workers):
    w = MyWorker(nameserver=ns_host, nameserver_port=ns_port, convert_func=config2structure, dataname="cifar10", nas_bench=nas_bench, nasbench_supernet=nasbench_supernet, time_budget=xargs.time_budget, run_id=hb_run_id, id=i)
    w.run(background=True)
    workers.append(w)

  start_time = time.time()
  bohb = HyperBand(configspace=cs,
            run_id=hb_run_id,
            eta=3, min_budget=12, max_budget=200,
            nameserver=ns_host,
            nameserver_port=ns_port,
            ping_interval=10)
  
  results = bohb.run(xargs.n_iters, min_n_workers=num_workers)

  bohb.shutdown(shutdown_workers=True)
  NS.shutdown()

  real_cost_time = time.time() - start_time

  id2config = results.get_id2config_mapping()
  incumbent = results.get_incumbent_id()
  logger.log('Best found configuration: {:} within {:.3f} s'.format(id2config[incumbent]['config'], real_cost_time))
  best_arch = config2structure(id2config[incumbent]['config'] )




  # info = nas_bench.query_by_arch( best_arch )
  # if info is None: logger.log('Did not find this architecture : {:}.'.format(best_arch))
  # else           : logger.log('{:}'.format(info))
  # logger.log('-'*100)
  #




  with open('/home/yiyangzhao/NAS-Bench-201/nasbench-201-search/nasbench201', 'r') as f:
    nasbench201 = json.load(f)

  arch_list = best_arch.tolist(remove_str=None)[0]
  acc = nasbench201[str(arch_list)]


  # with open('/home/yiyangzhao/NAS-Bench-201/nasbench-201-search/HB/last_results_1', 'a') as f:
  #   f.write(str([runs, acc]))
  #   f.write('\n')

  with open('/home/yiyangzhao/NAS-Bench-201/nasbench-201-search/HB/transfer', 'a') as f:
    f.write(str(workers[0].record))
    f.write('\n')


  logger.log('workers : {:.1f}s with {:} archs'.format(workers[0].time_budget, len(workers[0].seen_archs)))
  logger.close()
  return logger.log_dir, nas_bench.query_index_by_arch( best_arch ), real_cost_time



if __name__ == '__main__':
  parser = argparse.ArgumentParser("Regularized Evolution Algorithm")
  parser.add_argument('--data_path',          type=str,   help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  # channels and number-of-cells
  parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   help='The number of cells in one stage.')
  parser.add_argument('--time_budget',        type=int,   help='The total time cost budge for searching (in seconds).')
  # BOHB
  parser.add_argument('--strategy', default="sampling",  type=str, nargs='?', help='optimization strategy for the acquisition function')
  parser.add_argument('--min_bandwidth',    default=.3,  type=float, nargs='?', help='minimum bandwidth for KDE')
  parser.add_argument('--num_samples',      default=64,  type=int, nargs='?', help='number of samples for the acquisition function')
  parser.add_argument('--random_fraction',  default=.33, type=float, nargs='?', help='fraction of random configurations')
  parser.add_argument('--bandwidth_factor', default=3,   type=int, nargs='?', help='factor multiplied to the bandwidth')
  parser.add_argument('--n_iters',          default=100, type=int, nargs='?', help='number of iterations for optimization method')
  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--arch_nas_dataset',   type=str,   help='The path to load the architecture dataset (tiny-nas-benchmark).')
  parser.add_argument('--print_freq',         type=int,   help='print frequency (default: 200)')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')
  args = parser.parse_args()
  #if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  with open('/home/yiyangzhao/NAS-Bench-201/nasbench-201-search/nasbench201', 'r') as f:
    # nas_bench = json.load(f)
    nas_bench = API(args.arch_nas_dataset)
  with open('/home/yiyangzhao/NAS-Bench-201/nasbench-201-search/transfer', 'r') as f:
    nasbench_supernet = json.load(f)
  # nas_bench = API(args.arch_nas_dataset)
  if args.rand_seed < 0:
    save_dir, all_indexes, num, all_times = None, [], 50, []
    for i in range(num):
      print ('{:} : {:03d}/{:03d}'.format(time_string(), i, num))
      args.rand_seed = random.randint(1, 100000)
      save_dir, index, ctime = main(args, nas_bench, nasbench_supernet, i)
      all_indexes.append( index ) 
      all_times.append( ctime )
    print ('\n average time : {:.3f} s'.format(sum(all_times)/len(all_times)))
    torch.save(all_indexes, save_dir / 'results.pth')
  else:
    main(args, nas_bench, nasbench_supernet, 0)
