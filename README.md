# OpticalFlow-ICP
## 算法过程
光流法跟踪特征点，用PnP做位姿估计，当估计结果不佳时，引入图片深度信息，改为用ICP估计位姿

## 数据集准备
1. 下载TUM数据集：http://vision.in.tum.de/data/datasets/rgbd-dataset/download 
2. 使用associate.py将数据集对齐:
```python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associate.txt```

3. 将得到的associate.txt、rgb和depth数据集文件夹放至代码目录

## 编译运行
1. 打开终端
2. 键入如下代码片
```
cd PATH_TO_SEQUENCE&CODE
mkdir build && cd build
cmake ..
make
./lkicp
```

## 说明
- 本程序是基于引文的算法原理写出，不需要用Kinect Fusion，跑数据集即可
- **SLAM学习中，本算法实际应用意义有限，仅作学习和熟悉各种基础算法使用**


> 张岩,易柳.基于ICP与光流法结合的Kinect配准算法[J].湖北第二师范学院学报,2015,32(08):11-18.
