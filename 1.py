import os
from osgeo import gdal
import numpy as np
from PIL import Image
import cv2

#  读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset

def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    print(im_data.shape)
    print(im_data)
    im_data=im_data.transpose(1, 2, 0)
    scaled_data = ((im_data - im_data.min()) / (im_data.max() - im_data.min()) * 255).astype(np.uint8)

    im = Image.fromarray( scaled_data )
    im.save(path, "PNG")


'''
滑动窗口裁剪函数
TifPath 影像路径
SavePath 裁剪后保存目录
CropSize 裁剪尺寸
RepetitionRate 重复率
'''


def TifCrop(TifPath, SavePath, CropSize, RepetitionRate):
    dataset_img = readTif(TifPath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据

    #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
    new_name = len(os.listdir(SavePath))
    #  裁剪图片,重复率为RepetitionRate

    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            #  如果图像是单波段
            if (len(img.shape) == 2):
                cropped = img[
                          int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            #  如果图像是多波段
            else:
                cropped = img[:,
                          int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            #  写图像
            writeTiff(cropped, geotrans, proj, SavePath + "/%d.jpg" % new_name)
            #  文件名 + 1
            new_name = new_name + 1
    #  向前裁剪最后一列
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        if (len(img.shape) == 2):
            cropped = img[int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize): width]
        else:
            cropped = img[:,
                      int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize): width]
        #  写图像
        writeTiff(cropped, geotrans, proj, SavePath + "/%d.jpg" % new_name)
        new_name = new_name + 1
    #  向前裁剪最后一行
    for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        if (len(img.shape) == 2):
            cropped = img[(height - CropSize): height,
                      int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        else:
            cropped = img[:,
                      (height - CropSize): height,
                      int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        writeTiff(cropped, geotrans, proj, SavePath + "/%d.jpg" % new_name)
        #  文件名 + 1
        new_name = new_name + 1
    #  裁剪右下角
    if (len(img.shape) == 2):
        cropped = img[(height - CropSize): height,
                  (width - CropSize): width]
    else:
        cropped = img[:,
                  (height - CropSize): height,
                  (width - CropSize): width]
    writeTiff(cropped, geotrans, proj, SavePath + "/%d.jpg" % new_name)
    new_name = new_name + 1


# 训练集和验证集都要裁剪
# 裁剪出 图像   特征。拿到  影像数据增强中进行数据增强
#  将影像1裁剪为重复率为0.1的256×256的数据集
# 图像
TifCrop(r"3band/1.tif",
        r"D:/zhuomian/GF_test/output_tiles/", 800, 0.1)

# 自左向右依次为影像路径、输出路径、图像大小及重复率

