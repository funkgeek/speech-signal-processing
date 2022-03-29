import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import  Dataset,DataLoader
import torch.nn.functional as F
import pickle
def emo_dir_transpose(src_emo, tar_emo, src_path: Path):
    '''

    :param src_emo: 'Neutral'
    :param tar_emo: 'Angry'
    :param path: Path(/disk1/ywh/s2sevc/norm_meldata/0001/Neutral/train/0001_000051.npz)
    :return:    Path(/disk1/ywh/s2sevc/norm_meldata/0001/Angry/train/0001_000401.npz)
    '''
    em2ind = {"Neutral": 0, "Angry": 1,   "Happy": 2,  "Sad": 3, "Surprise": 4}

    srcpath_list = [] + list(src_path.parts)
    ## 先替换情感
    srcpath_list[-3] = tar_emo

    ## 再换 basename的数字
    src_emo_index = em2ind[src_emo]  ## if neutral  = 0
    tar_emo_index = em2ind[tar_emo]  ## if angry = 1

    filename_part2 = int(src_path.stem.split('_')[-1])  ## 51

    index_dis = (tar_emo_index - src_emo_index) * 350  # 这里的例子是350
    tar_filename_part2 = filename_part2 + index_dis  # 51 + 350 = 401
    L1 = len(f"{tar_filename_part2}")
    tar_file_index_str = "0" * (6 - L1) + f"{tar_filename_part2}"  # 000401
    ##新的文件名：
    tar_basename = srcpath_list[-4] + "_" + tar_file_index_str + src_path.suffix  ## srcpath_list[-4]  表示0001，说话人编号
    # 替换到srcpathlist
    srcpath_list[-1] = tar_basename
    ## 新的路径：
    return Path().joinpath(*srcpath_list)

    pass

#### 补零。
def pad(x, seglen):
    pad_len = seglen - x.shape[1]
    y = np.pad(x, ((0, 0), (0, pad_len)), 'constant', constant_values=(0, 0))
    return y

def segment(x, seglen=128, r=None, return_r=False):
    '''
    :param x: npy形式的mel [80,L]
    :param seglen: padding长度
    :param r:  不需要传入
    :param return_r: 是否返回 取的 mel seglen 区间 的开头索引（长度为128）
    :return: padding mel
    '''
    ## 该函数将melspec [80,len] ，padding到固定长度 seglen
    if x.shape[1] < seglen:
        y = pad(x, seglen)
    elif x.shape[1] == seglen:
        y = x
    else:
        if r is None:
            r = np.random.randint(x.shape[1] - seglen) ## r : [0-  (L-128 )]
        y = x[:,r:r+seglen]
    if return_r:
        return y, r
    else:
        return y

class MeldataSet(Dataset):

    def __init__(self,scp_dir,seglen,mean_std_path=None):
        self.scripts = []
        self.seglen = seglen
        self.mean_std_path = mean_std_path
        if self.mean_std_path != None:
            ## 说明使用均值方差
            with open(self.mean_std_path,'rb') as hf:
                self.mead_std_dict = pickle.load(hf)

        ##  读取训练集路径
        with open(scp_dir,encoding='utf-8') as f :
            for l in f.readlines():
                self.scripts.append(Path(l.strip('\n')))
        self.L = len((self.scripts))

        pass

    def __getitem__(self, index):
        src_p = self.scripts[index] ## 路径

        src_emo = src_p.parts[-3]  ## 读取 说话人 和 情感

        src_mel_pad = segment(np.load(str(src_p)),seglen=self.seglen) ## 切割成 128 帧
        ## 【80,128】
        if self.mean_std_path != None:
            spkname = src_p.parts[-4]
            m = self.mead_std_dict[spkname][src_emo][0]
            t = self.mead_std_dict[spkname][src_emo][1]
            src_mel_pad = (src_mel_pad - m ) / (t + 1e-6)

        ## 这里考虑mask版本。
        return torch.FloatTensor(src_mel_pad)




        pass
    def __len__(self):
        return self.L
        pass
################################################
################################################
################################################
## 使用pathlib库重构这个函数的 路径操作 （os版本的这个函数在s2sevc project里面）
def generate_scripts(meldatadir_name,emotion_domains,speaker_domains,save_log_dir):
    # 该函数生成目标情感的训练数据 表单 scripts
    # 表单每一条为 源路径
    # 如 /disk1/ywh/s2sevc/norm_meldata/0001/Angry/train/0001_000001.npz

    if save_log_dir.exists() == False:
        print("表单的保存目录不存在！")
        exit()

    data_dir_p = Path(meldatadir_name)
    # 在stargan 的训练中，src可以为任意情感的 语音。只需要规定好 src 的domain域。
    src_dataemo = emotion_domains # ['Neutral']
    src_dataspk = speaker_domains # ['0001',]
    '''
    0001_000001	打远一看，它们的确很是美丽，	中立
    0001_000351	打远一看，它们的确很是美丽，	生气
    0001_000701	打远一看，它们的确很是美丽，	快乐
    0001_001051	打远一看，它们的确很是美丽，	伤心
    0001_001401	打远一看，它们的确很是美丽，	惊喜
    '''
    scp = ["train","test","evaluation"]
    for k in range(len(scp)): ### 写入三种表单。
        # 首先获取source 数据路径。利用字典存储
        src_data_paths_list = []
        for meldataf in data_dir_p.rglob('*.npy'):
                if (meldataf.parts[-3] in src_dataemo) and meldataf.parts[-4] in src_dataspk: ## 提取 全部情感、说话人 domains中语音 的 路径
                    if meldataf.parts[-2] == scp[k]:
                        src_data_paths_list.append(meldataf) #注意添加Path()对象

        ################################################
        with open(  save_log_dir / (scp[k] + ".txt") , 'a', encoding='utf-8') as f:
            ## './Experiments/vX/train.txt'
            for P in src_data_paths_list:
                f.write(str(P) + '\n')
        print("写入:CLVC-AGAIN 的 {}表单".format(scp[k]))
        ################################################
    pass


if __name__ == "__main__":

    # ## 测试生成 指定说话人和情感的表单
    # meld = '/data/private/user43/workspace/ywh/datasets/meldata_22k_trimed'
    # emod = ['Neutral']
    # spkd = ['0001','0005']
    # sa = './'
    # generate_scripts(meld,emod,spkd,Path(sa))
    # exit()

    # 测试数据集~~~
    scpn = 'Experiments/train.txt'
    meldataset = MeldataSet(scp_dir=scpn,
                            seglen=128,
                            mean_std_path='mean_std_mels_ESD.pickle')
    melloaer = DataLoader(meldataset,batch_size=2,shuffle=True,drop_last=True)
    for b in melloaer:
        src_mel= b
        print("src mel:",src_mel.shape) # [2,80,256]
        exit()
    print("测试dataloader完毕") # OK
    pass