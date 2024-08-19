import torch
import numpy as np
import scipy.stats as stats

# from benchmark.heuristics import *


class PanelBlockShop:
    def __init__(self, data_src, num_process=6, num_p1=3, num_of_blocks=50, distribution="lognormal"):
        self.num_p = num_process
        self.num_p_list = [num_p1, num_process - num_p1]
        self.num_of_blocks = num_of_blocks
        self.distribution = distribution

        if distribution == "lognormal":
            self.shape = [0.543, 0.525, 0.196, 0.451, 0.581, 0.432]
            self.scale = [2.18, 2.18, 0.518, 2.06, 1.79, 2.10]
        elif distribution == "uniform":
            self.loc = [0 for _ in range(num_process)]
            self.scale = [100 for _ in range(num_process)]

    def generate_data(self, batch_size=1, use_label=False):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        process_time = np.zeros((batch_size * self.num_of_blocks, self.num_p))

        for i in range(self.num_p):
            if self.distribution == "lognormal":
                r = np.round(stats.lognorm.rvs(self.shape[i], loc=0, scale=self.scale[i],
                                               size=batch_size * self.num_of_blocks), 1)
            elif self.distribution == "uniform":
                r = np.round(stats.uniform.rvs(loc=self.loc[i], scale=self.scale[i],
                                               size=batch_size * self.num_of_blocks), 1)
            process_time[:, i] = r
        process_time = process_time.reshape((batch_size, self.num_of_blocks, self.num_p))

        # if use_label:
        #     label = np.zeros((batch_size, self.num_of_blocks))
        #     for i, pt in enumerate(process_time):
        #         sequence_neh, makespan_neh = NEH_sequence(self, pt)
        #         label[i] = sequence_neh
        #     return torch.FloatTensor(process_time).to(device), torch.FloatTensor(label).to(device)
        # else:
        #     return torch.FloatTensor(process_time).to(device)

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

        CT_table = dict()
        CT_table['1'] = np.zeros((num_of_blocks + 1, self.num_p_list[0] + 1))
        CT_table['2A'] = np.zeros((num_of_blocks + 1, self.num_p_list[1]))
        CT_table['2B'] = np.zeros((num_of_blocks + 1, self.num_p_list[1]))
        line = None

        for i in range(num_of_blocks):
            for j in range(self.num_p):
                pt = blocks_numpy[sequence_numpy[i], j]

                # Line 1
                if j in range(self.num_p_list[0]):
                    # process의 관점에서 선행 job의 끝나는 시간
                    esd_1 = CT_table['1'][i, j + 1]

                    # job의 관점에서 선행 process의 끝나는 시간
                    esd_2 = CT_table['1'][i + 1, j]
                    ESD = max(esd_1, esd_2)

                    # completion time
                    ct = ESD + pt
                    CT_table['1'][i + 1, j + 1] = ct

                # Starting Line 2 ...
                # Determine in which line to put
                elif j == self.num_p_list[0]:

                    esd_1 = CT_table['1'][i + 1, j]
                    # job의 관점에서 선행 process가 끝나는 시간

                    esd_A = CT_table['2A'][i, 0]
                    esd_B = CT_table['2B'][i, 0]

                    if esd_A > esd_B:
                        line = '2B'
                        unused = '2A'
                        print('Block {0} goes to line B.'.format(i))
                        esd_2 = esd_B
                    elif esd_A == esd_B:
                        line = '2A'
                        unused = '2B'
                        print('Block {0} goes to line A.'.format(i))
                        esd_2 = esd_A
                    else:
                        line = '2A'
                        unused = '2B'
                        print('Block {0} goes to line A.'.format(i))
                        esd_2 = esd_A

                    ESD = max(esd_1, esd_2)
                    ct = ESD + pt
                    CT_table[line][i+1,0] = ct
                    # fill the unused line information
                    CT_table[unused][i+1,:] = CT_table[unused][i,:]

                # Line 2
                elif j in range(self.num_p_list[0] + 1, self.num_p):
                    esd_1 = CT_table[line][i,j-self.num_p_list[0]]
                    esd_2 = CT_table[line][i+1,j-self.num_p_list[0]-1]
                    ESD = max(esd_1, esd_2)
                    ct = ESD + pt
                    CT_table[line][i + 1, j - self.num_p_list[0]] = ct
                else:
                    print('Invalid Input Size!')


        C_max = max(np.max(CT_table['2A']),np.max(CT_table['2B']))

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        return torch.FloatTensor([C_max]).to(device)
