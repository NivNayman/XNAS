import torch
import torch.nn as nn
import utils as utils
import operations as ops
from genotypes import Genotype
from utils import SELayer


class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()

        if reduction_prev:
            self.preprocess0 = ops.FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ops.ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ops.ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = ops.OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0_initial, s1_initial, drop_prob):
        s0 = self.preprocess0(s0_initial)
        s1 = self.preprocess1(s1_initial)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, ops.Identity):
                    drop_path_inplace(h1, drop_prob)
                if not isinstance(op2, ops.Identity):
                    drop_path_inplace(h2, drop_prob)
            s = h1 + h2
            states += [s]

        return torch.cat([states[i] for i in self._concat], dim=1)


class Network(nn.Module):
    def __init__(self, C: int, num_classes, layers, num_reductions,
                 reduction_location_mode, genotype: Genotype, stem_multiplier, do_SE=False):

        super(Network, self).__init__()
        self._layers = layers
        self.do_SE = do_SE
        self.drop_path_prob = 0

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(stem_multiplier, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        self.cells_SE = nn.ModuleList()
        reduction_prev = False

        self.reduction_indices = utils.place_reduction_cells(num_layers=layers,
                                                             num_reductions=num_reductions,
                                                             mode=reduction_location_mode)
        for i in range(layers):
            if i in self.reduction_indices:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction,
                        reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

            if self.do_SE and i <= layers * 2 / 3:
                if C_curr == C:
                    reduction_factor_SE = 4
                else:
                    reduction_factor_SE = 8
                self.cells_SE += [SELayer(C_curr * 4, reduction=reduction_factor_SE)]

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            cell_output = cell(s0, s1, self.drop_path_prob)

            # SE
            if self.do_SE and i <= len(self.cells) * 2 / 3:
                cell_output = self.cells_SE[i](cell_output)

            s0, s1 = s1, cell_output

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
