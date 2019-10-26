from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

DARTS_V1 = Genotype(
    normal=[
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('skip_connect', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('skip_connect', 2)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('skip_connect', 2),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 0),
        ('skip_connect', 2),
        ('skip_connect', 2),
        ('avg_pool_3x3', 0)
    ],
    reduce_concat=[2, 3, 4, 5]
)

DARTS_V2 = Genotype(
    normal=[
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('skip_connect', 0),
        ('dil_conv_3x3', 2)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('skip_connect', 2),
        ('max_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('skip_connect', 2),
        ('skip_connect', 2),
        ('max_pool_3x3', 1)
    ],
    reduce_concat=[2, 3, 4, 5]
)

DARTS = DARTS_V2

XNAS = Genotype(
    normal=[
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 2),
        ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 1),
        ('dil_conv_5x5', 4),
        ('sep_conv_3x3', 0)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('max_pool_3x3', 0),
        ('avg_pool_3x3', 1),
        ('avg_pool_3x3', 0),
        ('skip_connect', 2),
        ('dil_conv_3x3', 3),
        ('max_pool_3x3', 0),
        ('dil_conv_5x5', 4),
        ('avg_pool_3x3', 0)
    ],
    reduce_concat=range(2, 6)
)
