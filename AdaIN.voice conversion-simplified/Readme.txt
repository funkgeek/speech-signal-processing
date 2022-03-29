1、ESD数据集下载链接：

链接：https://pan.baidu.com/s/1n5ox3MPrR5L0q8naRvXIYw 
提取码：d4e4 
--来自百度网盘超级会员V6的分享
里面包含 ESD 数据集本身
和ESD数据集的 melspec图片。

2、首先请使用Preprocess.py提取一下melspec到本地。
在 if __name__=="__main__":中

 wav_datadir_name = 填入你的 数据集存放地址（绝对文件夹路径）
如 '/user/ywh/datasets/ESD'
 feature_dir_name = 填入你的melspec提取到的地址（绝对文件夹路径）
如 '/user/ywh/datasets/ESD_melspec'

3、提取完melspec后， 打开项目里的 main.py
把 meld = 填入你的melspec提取到的地址（绝对文件夹路径）

4、运行Main.py开启训练。

5、可以用 Conversion_mel.py 进行音色转换。
注意实验版本等变量。





