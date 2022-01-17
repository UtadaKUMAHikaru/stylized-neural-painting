import numpy as np
from icecream import ic
data = np.load('/Users/chendeen/同步空间/绘画作业/不是人画的-2/4._strokes.npz')
# data = np.load('/Users/chendeen/Desktop/apple_tar/output/apple_strokes.npz')
ic(data.files)
ic(data['x_ctt'])
ic(data['x_ctt'].shape)
ic(data['x_color'])
ic(data['x_color'].shape)

from PIL import Image


img = Image.new("RGB",(15,15))###创建一个5*5的图片
stroke_num = 1
# 为较淡，不一定
pixTuple = (int(255*data['x_color'][0,stroke_num,0]),int(255*data['x_color'][0,stroke_num,1]),int(255*data['x_color'][0,stroke_num,2]),0)###三个参数依次为R,G,B,A   R：红 G:绿 B:蓝 A:透明度
# 为较浓
# pixTuple = (int(255*data['x_color'][0,stroke_num,3]),int(255*data['x_color'][0,stroke_num,4]),int(255*data['x_color'][0,stroke_num,5]),0)###三个参数依次为R,G,B,A   R：红 G:绿 B:蓝 A:透明度
# 还是就用 RGB0 吧
for i in range(15):
    for j in range(15):
        img.putpixel((i,j),pixTuple)
# img.save("bb.png")
img.show()

