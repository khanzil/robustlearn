from alg import alg
from utils.util import get_args
from torchvision.transforms import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# python visualize.py --data_dir ./data/ --task cross_people --test_envs 0 --dataset emg --algorithm diversify --latent_domain_num 10 --alpha1 1.0 --alpha 1.0 --lam 0.0 --local_epoch 3 --max_epoch 50 --lr 0.01 --output ./data/train_output/act/cross_people-emg-Diversify-0-10-1-1-0-3-50-0.01

# args = get_args()
# algorithm_class = alg.get_algorithm_class("diversify")
# algorithm = algorithm_class(args)
# ckpt = torch.load("./data/ckpt_test0.pth.rar", map_location='cpu')
# algorithm.load_state_dict(ckpt)

# data = np.load("./data/emg/emg_x.npy", ) # data size (smt, 8, 200)
# data = torch.from_numpy(data).float()
# fea_map = algorithm.dbottleneck(algorithm.featurizer(data[1,:,:]))

# plt.subplot(112)
# plt.plot(data[1,1,:])
# plt.subplot(212)
# plt.plot(fea_map)
# plt.show()

data = np.float32(np.loadtxt("./data/emg/01/1_raw_data_13-12_22.03.16.txt", dtype=str)[1:,:]) # data size (smt, 10)
idx = np.argwhere(data[:,9] == 3)
x = data[idx,:9]
print(x.shape)
# plt.plot(data[:,1])
# plt.show()

# y = np.load("./data/emg/emg_y.npy", ) # data size (smt, 3), 3 is (class, people, 0)
# df = pd.DataFrame(y)
# df.to_csv("test.csv", header=['one', 'two', 'three'])

# y = np.asanyarray(y)
# print(np.max(y,axis=0))
# plt.plot(y[:,0])
# plt.plot(y[:,1])
# plt.plot(y[:,2])
# plt.show()

