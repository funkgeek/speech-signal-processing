import pickle
import torch
import numpy as np
import random
from Create_Hparams import boot_a_new_experiment
from MyDataSet import generate_scripts
from Trainer import Trainer

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__=="__main__":
    ##################################################################

    same_seeds(2021)
    ## 开启一组实验文件
    CNENspeakers = []
    for i in range(20):
        s_i = str(i+1)
        if i+1 < 10:
            CNENspeakers.append("000" +  s_i)
        else:
            CNENspeakers.append("00" + s_i)
    print(CNENspeakers)
    CNspeakers = []
    for i in range(10):
        s_i = str(i+1)
        if i + 1 <10:
            CNspeakers.append("000" + s_i)
        else:
            CNspeakers.append("00" + s_i)
    # print(CNspeakers)
    # exit()



    hypo_params_dict = {
        "ep_version":"v1_CN",
        "mel_seglen":128,
        "batchsize_train":32,
        "total_iters":200000,
        "num_iters_decay":100000,
        "lr_update_every":100,
        "is_lr_decay":False,
        "start_lr":0.0005,
        "mean_std_path": None,       ##  "nN" 意思是 不做归一化 not norm
        "emotion_domains":['Neutral'],
        "train_speakers":CNspeakers ## FM MM ME FE => 0001 0005 0010 0015
        ### CNENspeakers[:18]
        ###
        ###  第一个字母 F M  代表 女、男
        ## 第二个字母 M E 代表 Mandarin \ English
     }
    print("训练用说话人：",hypo_params_dict["train_speakers"])
    vhp = boot_a_new_experiment(hypo_params_dict)
    # #### ['Neutral','Angry','Happy','Sad','Surprise']

    meld = '/data/private/user43/workspace/ywh/datasets/ESD_meldata_22k_trimed'
    generate_scripts(meld,
                           vhp.emotion_domains, # 指定 训练 用的情感数据 ,这里仅使用neutral
                           vhp.train_speakers,  ## 指定训练说话人。
                           vhp.ep_version_dir,    ## 指定 scp存放的文件夹
                           )

    # 训练 ,从刚才生成的pickle读取 hp
    hp_file_path = str(vhp.hp_filepath)
    loaded_hp = None
    with open(hp_file_path,'rb') as f2:
        loaded_hp=pickle.load(f2)
    print("seg mel len:",loaded_hp.mel_seglen)
    t = Trainer(loaded_hp)
    t.train_by_epoch()