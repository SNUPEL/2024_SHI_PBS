import torch
import numpy as np
import scipy.stats as stats

from benchmark.heuristics import *


class PanelBlockShop:
    def __init__(self, num_of_process=6, num_of_blocks=50, distribution="lognormal"):
        self.num_of_process = num_of_process
        self.num_of_blocks = num_of_blocks
        self.distribution = distribution

        if distribution == "lognormal":
            self.shape = [0.543, 0.525, 0.196, 0.451, 0.581, 0.432]
            self.scale = [2.18, 2.18, 0.518, 2.06, 1.79, 2.10]
        elif distribution == "uniform":
            self.loc = [0 for _ in range(num_of_process)]
            self.scale = [100 for _ in range(num_of_process)]

    def generate_data(self, batch_size=1, use_label=False):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        process_time = np.zeros((batch_size * self.num_of_blocks, self.num_of_process))

        for i in range(self.num_of_process):
            if self.distribution == "lognormal":
                r = np.round(stats.lognorm.rvs(self.shape[i], loc=0, scale=self.scale[i],
                                               size=batch_size * self.num_of_blocks), 1)
            elif self.distribution == "uniform":
                r = np.round(stats.uniform.rvs(loc=self.loc[i], scale=self.scale[i],
                                               size=batch_size * self.num_of_blocks), 1)
            process_time[:, i] = r
        process_time = process_time.reshape((batch_size, self.num_of_blocks, self.num_of_process))

        if use_label:
            label = np.zeros((batch_size, self.num_of_blocks))
            for i, pt in enumerate(process_time):
                sequence_neh, makespan_neh = NEH_sequence(self, pt)
                label[i] = sequence_neh
            return torch.FloatTensor(process_time).to(device), torch.FloatTensor(label).to(device)
        else:
            return torch.FloatTensor(process_time).to(device)

    def stack_makespan(self, blocks, sequences):
        list = [self.calculate_makespan(blocks[i], sequences[i]) for i in range(blocks.shape[0])]
        makespan_batch = torch.stack(list, dim=0)
        return makespan_batch

    def calculate_makespan(self, blocks, sequence):
        if isinstance(blocks, torch.Tensor):
            blocks_numpy = blocks.cpu().numpy()
        else:
            blocks_numpy = np.array(blocks)

        if isinstance(sequence, torch.Tensor):
            sequence_numpy = sequence.cpu().numpy()
        else:
            sequence_numpy = np.array(sequence)

        num_of_blocks = blocks.shape[0]
        num_of_process = blocks.shape[1]
        num_p1 = 5
        num_p2 = num_of_process - num_p1
        temp = np.zeros((num_of_blocks + 1, num_of_process + 1))

        for i in range(1, num_of_blocks + 1):
            for j in range(1, num_of_process + 1):
                line = None
                if j < num_p1 + 1:  # 1, 2, 3, ... , num_p1
                    ESD = max(temp[i - 1, j], temp[i, j - 1])  # Earliest Start Date
                    temp[i, j] = ESD + blocks_numpy[sequence_numpy[i - 1], j - 1]
                else:  # j = num_p1+1, num_p1+2, ... , num_of_process
                    # minimum. earliest start date of line1 and line2
                    if temp[i - 1, j] > temp[i - 1, j + num_p2]:
                        print('This block goes to line B.')
                        line = 'B'
                        ESD = max(temp[i, j - 1], temp[i - 1, j + num_p2])
                    elif temp[i - 1, j] == temp[i - 1, j + num_p2]:
                        print('Line 1 and 2 have the same ESD. ') # random하게 뽑을 수도 있지만, 일단은 Line A로 보냄
                        print('This block goes to line A.')
                        line = 'A'
                        ESD = max(temp[i, j - 1], [i - 1, j])
                    else:
                        print('This block goes to line A.')
                        line = 'A'
                        ESD = max(temp[i, j - 1], [i - 1, j])
                        # 이번에 선택되지 않은 line에 대해서, 바로 위 값을 복사해 줌



        C_max = temp[num_of_blocks, num_of_process]

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        return torch.FloatTensor([C_max]).to(device)
