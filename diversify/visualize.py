from alg import alg
import torch
import matplotlib.pyplot as plt

data = open("./data/emg/01/1_raw_data_13-12_22.03.16.txt")
algorithm = alg.get_algorithm_class("diversify")
ckpt = torch.load("./data/ckpt_test0.pth.rar", map_location='cpu')
algorithm.load_state_dict(ckpt)
algorithm.eval()

plt.plot(data)
plt.show()










