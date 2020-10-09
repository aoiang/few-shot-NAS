import hyperopt
from hyperopt import hp
from hyperopt import fmin, tpe
import numpy as np
import os
import json
import collections
import operator
import os, sys, time, glob, random, argparse


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


  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--arch_nas_dataset', type=str,
                      help='The path to load the architecture dataset (tiny-nas-benchmark).')
  parser.add_argument('--arch_supernet_dataset', type=str,
                      help='The path to load the supernet.')
  parser.add_argument('--rand_seed',          type=int,   default=-1,   help='manual seed')
  args = parser.parse_args()
  #if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)

  if not os.path.exists(args.save_dir):
      os.mkdir(args.save_dir)

  with open(args.arch_nas_dataset, 'r') as f:
      dataset = json.load(f)
  with open(args.arch_supernet_dataset, 'r') as f:
      supernet = json.load(f)


  args.rand_seed = random.randint(1, 100000)
  OPS = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
  best_acc = 94.37

  encode_dataset = {}
  encode_supernet = {}

  for network in dataset:
      encode_network = []
      acc = dataset[network]
      network = eval(network)
      for i in range(len(network)):
          for j in range(len(network[i])):
              encode_network.append(network[i][j][0])
      encode_dataset[str(encode_network)] = acc
  dataset = encode_dataset

  for net in supernet:
      encode_network = []
      acc = supernet[net]
      network = eval(net)
      for i in range(len(network)):
          for j in range(len(network[i])):
              encode_network.append(network[i][j][0])
      encode_supernet[str(encode_network)] = acc
  supernet = encode_supernet

  # print(supernet)

  counter = 0
  curt_best = 0
  best_trace = {}

  records = 0


  # define an objective function
  def objective(x):
      global counter
      global curt_best
      global best_trace
      counter += 1

      # network = [int(x[0] + 1), int(x[1]), int(x[2] + 1), int(x[3]), int(x[4] + 1), int(x[5]), int(x[6] + 1), int(x[7]),
      #            int(x[8] + 1), int(x[9])]

      network = [str(x[0]), str(x[1]), str(x[2]), str(x[3]), str(x[4]), str(x[5])]
      val = 0
      real_acc = 0
      network_str = str(network)

      if network_str in dataset:
          # val = dataset[network_str]
          val = supernet[network_str]
          real_acc = dataset[network_str]

      if real_acc > curt_best:
          curt_best = real_acc
          best_trace[network_str] = [counter, real_acc]

      # print(val)

      if real_acc == best_acc or counter >= 3000:
          print("finished, write to file")
          sorted_best_traces = sorted(best_trace.items(), key=operator.itemgetter(1))
          final_results = []
          for item in sorted_best_traces:
              final_results.append(item[1])
              final_results_str = json.dumps(final_results)
          with open(args.save_dir + '/results.txt', "a") as f:
              f.write(final_results_str + '\n')
          os._exit(1)

      return -1 * val


  # define a search space
  space = [
      hp.choice('x0', OPS),
      hp.choice('x1', OPS),
      hp.choice('x2', OPS),
      hp.choice('x3', OPS),
      hp.choice('x4', OPS),
      hp.choice('x5', OPS)

  ]

  # minimize the objective over the space
  best = fmin(objective, space, algo=tpe.suggest, max_evals=15000)