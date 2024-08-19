from env_separatedPBS import PanelBlockShop
import numpy as np

pbs = PanelBlockShop(num_process=4, num_p1=2, num_of_blocks=5, distribution="lognormal")
block_data = np.random.randint(low=1, high=5, size=(5,4))
sequence = [0, 1, 2, 3, 4]
pbs.calculate_makespan(block_data, sequence)
print()
