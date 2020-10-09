##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
# The macro structure is defined in NAS-Bench-201
from .search_model_darts    import TinyNetworkDarts
from .search_model_pcdarts  import TinyNetworkPCDarts
from .search_model_gdas     import TinyNetworkGDAS
from .search_model_setn     import TinyNetworkSETN
from .search_model_enas     import TinyNetworkENAS
from .search_model_random   import TinyNetworkRANDOM
from .rank_model            import UniformRandomSupernet
from .genotypes             import Structure as CellStructure, architectures as CellArchitectures
# NASNet-based macro structure
from .search_model_gdas_nasnet import NASNetworkGDAS
from .search_model_darts_nasnet import NASNetworkDARTS


nas201_super_nets = {'DARTS-V1': TinyNetworkDarts,
                     "DARTS-V2": TinyNetworkDarts,
                     "PCDARTS":  TinyNetworkPCDarts,
                     "GDAS": TinyNetworkGDAS,
                     "SETN": TinyNetworkSETN,
                     "ENAS": TinyNetworkENAS,
                     "RANDOM": TinyNetworkRANDOM,
                     "ONE-SHOT-SUPERNET": UniformRandomSupernet}

nasnet_super_nets = {"GDAS": NASNetworkGDAS,
                     "DARTS": NASNetworkDARTS}
