import torch
import numpy as np
import os
from torch.utils.data import  Dataset,DataLoader


def generate_scp_dataset(dataset_dir):
    with open('Train_Scp.txt','a',encoding='utf-8' ) as txtf :
        for dirname,subdirs,files in os.walk(dataset_dir):
            for f in files:
                if f.split('.')[-1] == 'npy':
                    txtf.write(os.path.join(dirname,f) + "\n")
    print("写入表单")

## 下面两个函数 ，实现，将一个二维矩阵 补零到 指定长度。 （补一列一列的零）. 如果超过 指定的seglen，则切掉多余的。
def pad(x, seglen, mode='wrap'):
    pad_len = seglen - x.shape[1]
    y = np.pad(x, ((0,0), (0,pad_len)), mode=mode)
    return y

def segment(x, seglen=128):
    '''
    :param x: npy形式的mel [80,L]
    :param seglen: padding长度
    :return: padding mel
    '''
    ## 该函数将melspec [80,len] ，padding到固定长度 seglen
    if x.shape[1] < seglen:
        y = pad(x, seglen)
    elif x.shape[1] == seglen:
        y = x
    else:
        r = np.random.randint(x.shape[1] - seglen) ## r : [0-  (L-128 )]
        y = x[:,r:r+seglen]
    return y

class MeldataSet_1(Dataset):
    def __init__(self,scp_dir,seglen):
        self.scripts = []
        self.seglen = seglen
        with open(scp_dir,encoding='utf-8') as f :
            for l in f.readlines():
                self.scripts.append(l.strip('\n'))
        self.L = len((self.scripts))
        pass
    def __getitem__(self,index):

        src_path = self.scripts[index]
        src_mel = np.load(src_path)## 从硬盘将数据 读入内存的一个io过程。.npy
        #src_mel = segment(np.load(src_path), seglen=self.seglen)
        return torch.FloatTensor(src_mel) ## 【80,256】

    def __len__(self):
        return self.L
        pass
    
def my_collection_way(batch): ## batch : tuple
    print("Dataloder 中的 collection func 调用：")
    print([  x.shape for x in batch])
    output = torch.stack([ torch.FloatTensor(segment(x,seglen=256)) for x in batch  ],dim=0)
    return output



    pass
if __name__ == '__main__':

    #generate_scp_dataset("meldata_22k_trimed")

    # ############################################################
    # ####   padding与不padding的 dataloader  演示。
    # Mdata = MeldataSet_1("Train_Scp.txt",seglen=256)
    # print(Mdata[0].shape) ##  进行索引操作的时候，就是在调用 getitem (index)
    # print(Mdata[1].shape)
    # print("-----------")
    #
    # ## 包装这个dataset，成为datalodaer
    # Mdataloader = DataLoader(Mdata, batch_size=3)
    # for batch in Mdataloader:
    #     print(batch.shape)  ## [3,80,256]
    #     print("-----")
    #     ## 为什么要读取一批数据呢 ？显卡 处理一批数据 ，速度比较cpu快的~
    # exit()
    ############################################################

    ############################################################
    ###  演示collection——fn
    Mdata = MeldataSet_1("Train_Scp.txt",seglen=256)

    Mdataloader = DataLoader(Mdata, batch_size=3,collate_fn=my_collection_way)
    for batch in Mdataloader:
        print(batch.shape)
        exit()
    ############################################################