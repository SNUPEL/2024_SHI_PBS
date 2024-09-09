import torch
import numpy as np
import scipy.stats as stats
from panelblock_PBS import DataGenerator

# from benchmark.heuristics import *

# print 구문 주석처리

class PanelBlockShop:
    def __init__(self, num_process=10, num_p1=3, num_of_blocks=50):
        self.num_p = num_process
        self.num_p_list = [num_p1, num_process - num_p1]
        self.num_of_blocks = num_of_blocks
        self.data_generator = DataGenerator(data_file=None, num_of_blocks=num_of_blocks)
        self.selected_types = None
        self.all_data = None

    def generate_data(self, batch_size=32):
        total_blocks = batch_size * self.num_of_blocks
        
        # print(f"Generating {total_blocks} samples...")
        self.data_generator.num_of_blocks = total_blocks  # DataGenerator의 num_of_blocks를 업데이트
        self.all_data, self.selected_types = self.data_generator.generate_all_types(self.data_generator.type_counts)

        process_time = np.zeros((total_blocks, self.num_p))
        
        for j in range(total_blocks):
            type_name = self.selected_types[j]
            
            if type_name not in self.all_data or '0' not in self.all_data[type_name]:
                # print(f"Warning: Data not found for type {type_name}.")
                continue
            
            data_for_block = self.all_data[type_name]['0'][j]
            
            for i in range(self.num_p):
                if i < len(data_for_block):
                    value = data_for_block[i].item() if isinstance(data_for_block[i], torch.Tensor) else data_for_block[i]
                    process_time[j, i] = value
                # else:
                #     print(f"Warning: Process index {i} out of range for block {j} of type {type_name}.")
        
        return process_time.reshape(batch_size, self.num_of_blocks, self.num_p)

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
                        # print('Block {0} goes to line B.'.format(i))
                        esd_2 = esd_B
                    elif esd_A == esd_B:
                        line = '2A'
                        unused = '2B'
                        # print('Block {0} goes to line A.'.format(i))
                        esd_2 = esd_A
                    else:
                        line = '2A'
                        unused = '2B'
                        # print('Block {0} goes to line A.'.format(i))
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