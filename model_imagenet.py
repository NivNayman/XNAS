import torch
import torch.nn as nn
import operations as ops
from genotypes import Genotype
from model import Cell
from utils import SELayer

class NetworkImageNet(nn.Module):
    def __init__(self, C, num_classes, layers, genotype, do_SE=True, C_stem=56):
        stem_activation = nn.ReLU
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self.drop_path_prob = 0
        self.do_SE = do_SE

        self.C_stem = C_stem
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C_stem // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C_stem // 2),
            stem_activation(inplace=True),
            nn.Conv2d(C_stem // 2, C_stem, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C_stem),
        )

        self.stem1 = nn.Sequential(
            stem_activation(inplace=True),
            nn.Conv2d(C_stem, C_stem, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C_stem),
        )

        C_prev_prev, C_prev, C_curr = C_stem, C_stem, C
        self.cells = nn.ModuleList()
        self.cells_SE = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

            if self.do_SE and i <= layers * 2 / 3:
                if C_curr == C:
                    reduction_factor_SE = 4
                else:
                    reduction_factor_SE = 8
                self.cells_SE += [SELayer(C_curr * 4, reduction=reduction_factor_SE)]

        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            cell_output = cell(s0, s1, self.drop_path_prob)

            if self.do_SE and i <= len(self.cells) * 2 / 3:
                cell_output = self.cells_SE[i](cell_output)

            s0, s1 = s1, cell_output

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

