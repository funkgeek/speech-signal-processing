
import pickle
from matplotlib import pyplot as plt
import os

def plot_loss(dict_path):
    if os.path.exists(dict_path) == False:
        print("路径不存在")

    '''

    '''
    with open(dict_path ,'rb') as f :
        loss_d = pickle.load(f)
    ###########
    steps = []
    for k,v in loss_d.items():
        if "step" in k:
            steps = loss_d[k]
            break


    for k,v in loss_d.items():
        if "loss" in k:
            losses = loss_d[k]
            plt.figure()
            plt.title(k)
            plt.xlabel(os.path.basename(dict_path))
            plt.plot(steps,losses)
            plt.show()

    for k ,v in loss_d.items():
        if "LR" in k:
            losses = loss_d[k]
            plt.figure()
            plt.title(k)
            plt.xlabel(os.path.basename(dict_path))
            plt.plot(steps,losses)
            plt.show()

def plot_loss_bylogtxt(txt_path,ver):

    loss_d = {}
    lines = []
    with open(txt_path ,encoding='utf-8') as f :
        for line in f.readlines():
            if line[:5] == "epoch":
                lines.append(line.strip('\n'))
    ###########
    ## 定义图片存储位置
    fig_file = os.path.join(os.path.dirname(txt_path),"Loss_Curves_Figures_{}".format(ver))
    os.makedirs(fig_file,exist_ok=True)

    ini_0row = [  kvs for kvs in   lines[0].split(',')[:-1] ]
    for kvs in ini_0row:
        k,v = kvs.split(':')
        loss_d[k] = []
    ## 创建loss_d
    for l in lines:
        one_row = [kvs for kvs in l.split(',')[:-1]]
        for kvs in one_row:
            k, v = kvs.split(':')
            loss_d[k].append(float(v))

    steps = []
    for k,v in loss_d.items():
        if "step" in k:
            steps = loss_d[k]
            break
    for k,v in loss_d.items():
        if "loss" in k:
            losses = loss_d[k]
            plt.figure()
            plt.title(k)
            plt.xlabel(os.path.basename(txt_path))
            plt.plot(range(len(losses)),losses)
            plt.savefig(os.path.join(fig_file,k.replace("/","_").strip(" ")))
            plt.show()
            print("max of loss {} : {}".format(k,max(losses)))
            print("min of loss {} : {}".format(k, min(losses)))
    for k ,v in loss_d.items():
        if "LR" in k:
            losses = loss_d[k]
            plt.figure()
            plt.title(k)
            plt.xlabel(os.path.basename(txt_path))
            plt.plot(range(len(losses)),losses)
            plt.savefig(os.path.join(fig_file,k.replace("/","_").strip(" ")))
            plt.show()

if __name__ == '__main__':

    ver = 'v1_CN'
    pa = f'Experiments/{ver}/logs_{ver}.txt'
    plot_loss_bylogtxt(pa,ver)
    pass
