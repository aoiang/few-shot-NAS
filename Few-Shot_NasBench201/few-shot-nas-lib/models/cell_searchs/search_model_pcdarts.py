##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
########################################################
# DARTS: Differentiable Architecture Search, ICLR 2019 #
########################################################
import torch
import torch.nn as nn
from copy import deepcopy
from ..cell_operations import ResNetBasicblock
from .search_cells     import NAS201SearchCell as SearchCell
from .genotypes        import Structure


class TinyNetworkPCDarts(nn.Module):

  def __init__(self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats, edge_to_split=0, edge_op=0,
               pcdarts_prune_ratio=1):
    super(TinyNetworkPCDarts, self).__init__()
    self._C        = C
    self._layerN   = N
    self.max_nodes = max_nodes
    self.stem = nn.Sequential(
                    nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(C))
  
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    C_prev, num_edge, edge2index, split_i, split_j = C, None, None, None, None
    self.cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      if reduction:
        cell = ResNetBasicblock(C_prev, C_curr, 2)
      else:
        cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats, edge_to_split, edge_op,
                          pcdarts_prune_ratio)
        if num_edge is None: num_edge, edge2index, split_i, split_j = cell.num_edges, cell.edge2index, cell.split_i, cell.split_j
        else: assert num_edge == cell.num_edges and edge2index == cell.edge2index and split_i == cell.split_i and split_j == cell.split_j, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
      self.cells.append( cell )
      C_prev = cell.out_dim
    self.op_names   = deepcopy( search_space )
    self._Layer     = len(self.cells)
    self.edge2index = edge2index
    self.split_i = split_i
    self.split_j = split_j
    self.edge_to_split = edge_to_split
    self.edge_op = edge_op
    self.lastact    = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    self.arch_parameters = nn.Parameter( 1e-3*torch.randn(num_edge-1, len(search_space)) )

  def get_weights(self):
    xlist = list( self.stem.parameters() ) + list( self.cells.parameters() )
    xlist+= list( self.lastact.parameters() ) + list( self.global_pooling.parameters() )
    xlist+= list( self.classifier.parameters() )
    return xlist

  def get_alphas(self):
    return [self.arch_parameters]

  def show_alphas(self):
    with torch.no_grad():
      return 'arch-parameters :\n{:}'.format( nn.functional.softmax(self.arch_parameters, dim=-1).cpu() )

  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

  def genotype(self):
    genotypes = []
    for i in range(1, self.max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        if node_str != '{:}<-{:}'.format(self.split_i, self.split_j):
          with torch.no_grad():
            weights = self.arch_parameters[ self.edge2index[node_str] ]
            op_name = self.op_names[ weights.argmax().item() ]
          xlist.append((op_name, j))
        else:
          with torch.no_grad():
            op_name = self.op_names[self.edge_op]
          xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return Structure( genotypes )

  def forward(self, inputs):
    alphas  = nn.functional.softmax(self.arch_parameters, dim=-1)

    feature = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      if isinstance(cell, SearchCell):
        feature = cell.forward_pcdarts(feature, alphas)
      else:
        feature = cell(feature)

    out = self.lastact(feature)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return out, logits
