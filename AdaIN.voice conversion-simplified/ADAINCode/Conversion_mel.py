from matplotlib import pyplot as plt
import numpy as np
import requests

from pathlib import Path
from Models import AE
from Create_Hparams import Create_Train_Hparams
import pickle
import torch

### #meldata_22k_trimed/0004/Neutral/evaluation/0004_000001.npy

def Conversion_CrossLang_ESDmeldata(meldatadir:Path, ## 推理的时候，用的数据集位置
                                    hp:Create_Train_Hparams,
                              src_speakers:list,## 中文 源语音 说话人列表
                              tar_speakers:list, ## 英文源语音 说话人列表
                              emotion_list:list,## 训练时候用的 情感。
                              use_split:str): ## test or evaluation
    '''
    ## conversion 仅仅在 不同语言的说话人之间 进行

    '''

    ## result 文件创建  (已经存在:Experiments/v1/conversion_result,则删除)
    Direct_1 = hp.conversion_dir / "src_mels_figures"
    Direct_2 = hp.conversion_dir / "dec_mels_figures"
    Direct_3 = hp.conversion_dir / "dec_mels_npys"
    Direct_4 = hp.conversion_dir / "tar_mels_figures"

    Direct_1.mkdir(parents=True, exist_ok=True)
    Direct_2.mkdir(parents=True, exist_ok=True)
    Direct_3.mkdir(parents=True, exist_ok=True)
    Direct_4.mkdir(parents=True, exist_ok=True)
    print("inference emos:",hp.emotion_domains )
    print("inference src speaker:",src_speakers)
    print("inference tar speaker:",tar_speakers)
    print("*" * 100)

    ## 加载模型参数 ####### 默认加载最大步数的模型 ######## 用cpu 推理即可 ################
    again_model = AE(hp.model_params_config)
    model_state_paths = [f for f in hp.model_savedir.rglob('*') if f.is_file()]
    model_state_path = sorted(model_state_paths, key=lambda x: int(x.stem), reverse=True)[0]  ## 模型路径降序排列

    modelstate = torch.load(str(model_state_path), map_location='cpu')
    again_model.load_state_dict(modelstate["model"])
    print("使用模型：{}".format(model_state_path))

    pass


def Inference_a_mel(model,hp:Create_Train_Hparams,mean_std_dict:dict,src_mel_path:Path,tar_mel_path:Path,is_use_vocoder=False):
    '''
    :param hp:    训练参数文件夹
    :param src_mel_path:    源mel 的npy路径
    :param tar_mel_path:    tar mel 的npy 路径
    :param model_state_path:  模型字典文件
    :param seglen: 长度padding
    :return:
    '''
    ### #meldata_22k_trimed/0004/Neutral/evaluation/0004_000001.npy
    ### 由 src tar mel推理
    ### 若自己不指定seglen，则默认按 前 hp.mel,0eg_len 帧进行推理 。
    tar_emo = tar_mel_path.parts[-3]
    tar_spkname = tar_mel_path.parts[-4]

    src_mel = np.load(str(src_mel_path)) #[80,L]
    tar_mel = np.load(str(tar_mel_path))


    ## result 文件创建  (已经存在:Experiments/v1/conversion_result,则删除)
    new_basename = "{}_{}.npy".format(src_mel_path.stem,tar_mel_path.stem) ##0001_000351_Neutral_Angry.npy
    this_conv_file = new_basename.replace('.npy', '')
    Direct_1 = hp.conversion_dir /  this_conv_file / "src_mels_figures"
    Direct_2 = hp.conversion_dir /  this_conv_file /"dec_mels_figures"
    Direct_3 = hp.conversion_dir /  this_conv_file /"dec_mels_npys"
    Direct_4 = hp.conversion_dir /  this_conv_file /"tar_mels_figures"

    Direct_1.mkdir(parents=True, exist_ok=True)
    Direct_2.mkdir(parents=True, exist_ok=True)
    Direct_3.mkdir(parents=True, exist_ok=True)
    Direct_4.mkdir(parents=True, exist_ok=True)

    srcfig_save_path = hp.conversion_dir / this_conv_file /"src_mels_figures" / src_mel_path.stem  ## 0001_000351
    tarfig_save_path = hp.conversion_dir / this_conv_file /"tar_mels_figures" / tar_mel_path.stem  ## 0001_000351
    decdata_save_path = hp.conversion_dir /this_conv_file / "dec_mels_npys" / new_basename
    decfig_save_path = hp.conversion_dir / this_conv_file /"dec_mels_figures" / new_basename.replace('.npy', '')


    src_mel = torch.FloatTensor(src_mel).unsqueeze(0)
    tar_mel = torch.FloatTensor(tar_mel).unsqueeze(0)
    dec_mel = model.inference(src_mel,tar_mel)
    dec_mel = dec_mel.squeeze().detach().cpu().numpy() # [80, L]

    if  mean_std_dict != None:
        m = mean_std_dict[tar_spkname][tar_emo][0]
        t = mean_std_dict[tar_spkname][tar_emo][1]
        dec_mel = dec_mel * t + m  ## 反归一化

    np.save(str(decdata_save_path),dec_mel)
        ####
    ######### 作srcmel图 ########
    plt.figure()
    plt.title(src_mel_path.stem)
    plt.imshow(np.flip(src_mel.squeeze().numpy(),axis=0), cmap='Greens')
    plt.savefig(srcfig_save_path)
    plt.close() ## 减少缓存占用
    #plt.show()

    ######### 作tarmel图 ########
    plt.figure()
    plt.title(tar_mel_path.stem)
    plt.imshow(np.flip(tar_mel.squeeze().numpy(),axis=0), cmap='Greens')
    plt.savefig(tarfig_save_path)
    plt.close() ## 减少缓存占用
    #plt.show()

    ######### 作decmel图 ########
    plt.figure()
    plt.title(new_basename)
    plt.imshow(np.flip(dec_mel,axis=0),cmap='Greens')
    plt.savefig(decfig_save_path)
    plt.close()
    #plt.show()
    # ################################################################################################
    if is_use_vocoder:
        # ####  dec_mel 转 wav use melGan
        decwav_save_path = hp.conversion_dir / this_conv_file/"dec_mels_npys" / new_basename.replace('npy','wav')
        tmp_io = open(str(decdata_save_path), 'rb')
        res = requests.post('http://39.106.187.242:20080/upload_mel/', files={'file': tmp_io})
        tmp_io.close()
        wavfilepath = decwav_save_path
        with open(wavfilepath, 'wb') as w:
            w.write(res.content)
        print("use vocoder and save wav:",decwav_save_path)
    print("-" * 50)

if __name__ == '__main__':
    # v = "v1" ##
    # use_split = 'evaluation'
    # src_spks = ['0001','0005']  ##  Madarin_female , Madarin_male
    # tar_spks = ['0011','0015'] ## English_female , English_male
    #
    #
    # meld = Path('/data/private/user43/workspace/ywh/datasets/ESD_meldata_22k_trimed')
    #
    # hp_file_path = "Experiments/{}/hparams_{}.pickle".format(v,v)
    # ## 加载实验参数 ################################################
    # hp = None
    # with open(hp_file_path,'rb') as f :
    #     hp = pickle.load(f)
    #     f.close()
    # if hp == None:
    #     print("Load hparams error")
    # ##############################################################
    #
    # emotions = hp.emotion_domains
    #
    # Conversion_CrossLang_ESDmeldata(meldatadir=meld,
    #                                 hp=hp,
    #                                 src_speakers=src_spks,
    #                                 tar_speakers=tar_spks,
    #                                 emotion_list=emotions,
    #                                 use_split=use_split)

    # ######### test infe one 2 one



    vers = "v1_CN"
    modelname = "200001.pth"
    infe_src_mel_path = '/data/private/user43/workspace/ywh/datasets/ESD_meldata_22k_trimed/0001/Neutral/evaluation/0001_000002.npy'
    infe_tar_mel_path = '/data/private/user43/workspace/ywh/datasets/ESD_meldata_22k_trimed/0005/Neutral/evaluation/0005_000002.npy'

    with open('Experiments/{}/hparams_{}.pickle'.format(vers,vers),'rb') as f :
        hp = pickle.load(f)
        f.close()

    model = AE(hp.model_params_config)


    modelstate = torch.load(f'Experiments/{vers}/checkpoints_{vers}/{modelname}', map_location='cpu')["model"]
    model.load_state_dict(modelstate)

    ## 归一化
    #mean_std_path = 'mean_std_mels_ESD.pickle'
    mean_std_path = None
    if mean_std_path != None:
        ## 说明使用均值方差
        with open(mean_std_path, 'rb') as hf:
            mean_std_dict = pickle.load(hf)
    else:
        mean_std_dict = None
    # print(mean_std_dict)
    # exit()
    Inference_a_mel(model,hp,mean_std_dict,Path(infe_src_mel_path), Path(infe_tar_mel_path),is_use_vocoder=True)

    pass
