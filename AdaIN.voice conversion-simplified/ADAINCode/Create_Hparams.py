from pathlib import Path  ###  某种程度上来说这个库确实比 os好用一些。
import torch
import pickle
class Create_Train_Hparams():
    def __init__(self):
        ################ ESD data set   ###################################

        ###

        self.emo2ind_dict = {'Neutral': 0, 'Angry': 1, 'Happy': 2, 'Sad': 3, 'Surprise': 4}
        self.ind2emo_dict = {0: 'Neutral', 1: 'Angry', 2: 'Happy', 3: 'Sad', 4: 'Surprise'}

        self.emotion_domains = ['Neutral','Angry','Happy','Sad','Surprise' ] ##所训练的情感域
        self.train_speakers = ['0001']
        self.use_meldatadir_name = './meldata_22k_trimed'


        self.mean_std_path = 'mean_std_mels_ESD.pickle' ##是否归一化训练

        ################################################################
        ################ Trainer  ###################################
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.total_iters = 200000
        self.num_iters_decay = 100000### 10 0000 步之后，学习率 按线性下降。

         ##  every
        self.model_save_every = 5000
        self.eval_every = 5000
        self.loss_save_every = 5000
        self.is_lr_decay = False  # 学习率下降
        self.lr_update_every = 100


        ################ dataset / loader  ###################################
        self.mel_seglen = 256  ### 训练mel一律segment到 seglen
        #self.max_train_mellen = 300 ### 生成表单中，所含有的 pairs的最大melspec长度。
        self.min_train_mellen = 120
        self.batchsize_train  = 8
        self.batchsize_eval = 2

        ################ model params ################################
        self.model_params_config = {"SpeakerEncoder":
                      {"c_in": 80,
                        "c_h": 128,
                        "c_out": 128,
                        "kernel_size": 5,
                        "bank_size": 8,
                        "bank_scale": 1,
                        "c_bank": 128,
                        "n_conv_blocks": 6,
                        "n_dense_blocks": 6,
                        "subsample": [1, 2, 1, 2, 1, 2],## 下采样的主要功能：缩小时间帧
                        "act": 'relu',
                        "dropout_rate":0},
                    "ContentEncoder":{
                        "c_in": 80,
                        "c_h": 128,
                        "c_out": 128,
                        "kernel_size": 5,
                        "bank_size": 8,
                       "bank_scale": 1,
                        "c_bank": 128,
                        "n_conv_blocks": 6,
                        "subsample": [1, 2, 1, 2, 1, 2],
                        "act": 'relu',
                        "dropout_rate": 0}
                        ,
                        "Decoder":{
                        "c_in": 128,
                        "c_cond": 128,
                        "c_h": 128,
                        "c_out":80,
                        "kernel_size": 5,
                        "n_conv_blocks": 6,
                        "upsample": [2, 1, 2, 1, 2, 1],
                        "act": 'relu',
                        "sn": False,
                        "dropout_rate": 0}
                        }

        ################################################################
        #################  optimizer  ##############################
        self.start_lr = 0.0005

        self.beta1 = 0.99
        self.beta2 = 0.999
        self.amsgrad = True
        self.weight_decay  = 0.0001
        self.grad_norm  = 5.0
        self.lambda_rec = 10.0
        self.lambda_kl = 1.0
        self.annealing_iters = 20000
        ################################################################
        ## Experiment File dir
        self.epdir = Path('./Experiments')
        ## dirs create
        self.ep_version = None
        self.ep_version_dir = None
        self.model_savedir = None
        ## files create
        self.hp_filepath = None
        self.ep_logfilepath = None
        self.ep_logfilepath_eval = None
        ################################################################

        pass
    def set_emo_domains_nums(self):
        self.num_emotion = len(self.emotion_domains) ## 注意这里要一起改变

    def set_experiment(self,version='v1'):
        ## dirs create
        self.ep_version = version
        self.ep_version_dir = self.epdir / self.ep_version
        self.model_savedir = self.ep_version_dir / 'checkpoints_{}'.format(version)
        self.conversion_dir = self.ep_version_dir / 'conversion_result_{}'.format(version)
        ## files create
        self.hp_filepath = self.ep_version_dir.joinpath('hparams_{}.pickle'.format(version))
        self.ep_logfilepath = self.ep_version_dir / 'logs_{}.txt'.format(version)
        self.ep_logfilepath_eval = self.ep_version_dir / 'logs_eval_{}.txt'.format(version)
        self.losspkl_savedir  = self.ep_version_dir / 'run_losspkl_{}'.format(version)


class Create_Prepro_Hparams():
    def __init__(self):
        ################ preprocess  ###################################
        self.wav_datadir_name = ''
        self.feature_dir_name = ''
        self.trim_db = 20
        self.n_fft = 1024
        self.win_length = 1024
        self.hop_length = 256
        self.sample_rate =22050
        self.f_min = 0
        self.f_max = 11025
        self.n_mels = 80

    def set_preprocess_dir(self, wav_datadir, feature_dir):
        self.wav_datadir_name = wav_datadir
        self.feature_dir_name = feature_dir
        pass

# 该函数创建一个 实验参数类对象
def boot_a_new_experiment_hp(p_dict):
    '''
    实验的超参数可以自己设置，但是一定要和 类中属性名 相同
    '''
    hp = Create_Train_Hparams()
    for hyp_k, hpy_v in p_dict.items():
        for k, v in hp.__dict__.items():
            if k == hyp_k:
                hp.__setattr__(hyp_k, hpy_v)
    hp.set_experiment(hp.ep_version)  ## 根据自己指定的实验号 设定 实验版本
    hp.set_emo_domains_nums()
    return hp
    ###


def boot_a_new_experiment(p_dict):

    hp = boot_a_new_experiment_hp(p_dict)
    ## 创建实验文件夹,
    hp.ep_version_dir.mkdir(parents=True, exist_ok=True) ## Experiment/v1/
    hp.model_savedir.mkdir(parents=True, exist_ok=True)  ## Experiment/v1/checkpoints
    hp.conversion_dir.mkdir(parents=True, exist_ok=True) ## Experiment/v1/conversion_result
    hp.losspkl_savedir.mkdir(parents=True, exist_ok=True)
    ## 存储参数文件本身
    with open(hp.hp_filepath.resolve(),'wb') as hpf:
        pickle.dump(hp,hpf)
    return hp

    pass



if __name__ == "__main__":



    pass









