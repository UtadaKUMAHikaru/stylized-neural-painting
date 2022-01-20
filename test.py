from cgi import test
import numpy as np
from icecream import ic
import pandas as pd
# data = np.load('/Users/chendeen/同步空间/绘画作业/不是人画的-2/4._strokes.npz')
# # data = np.load('/Users/chendeen/Desktop/apple_tar/output/apple_strokes.npz')
# ic(data.files)
# ic(data['x_ctt'])
# ic(data['x_ctt'].shape)
# ic(data['x_color'])
# ic(data['x_color'].shape)

# from PIL import Image


# img = Image.new("RGB",(15,15))###创建一个5*5的图片
# stroke_num = 1
# # 为较淡，不一定
# pixTuple = (int(255*data['x_color'][0,stroke_num,0]),int(255*data['x_color'][0,stroke_num,1]),int(255*data['x_color'][0,stroke_num,2]),0)###三个参数依次为R,G,B,A   R：红 G:绿 B:蓝 A:透明度
# # 为较浓
# # pixTuple = (int(255*data['x_color'][0,stroke_num,3]),int(255*data['x_color'][0,stroke_num,4]),int(255*data['x_color'][0,stroke_num,5]),0)###三个参数依次为R,G,B,A   R：红 G:绿 B:蓝 A:透明度
# # 还是就用 RGB0 吧
# for i in range(15):
#     for j in range(15):
#         img.putpixel((i,j),pixTuple)
# # img.save("bb.png")
# img.show()

# count = 1
# with open('stroke_width_test.txt', 'r') as f:
#     stroke_width = f.readlines()
#     for i in stroke_width:
#         # ic(i)
#         i = float(i.strip())
#         if i < 5 and count >= 304:
#             ic(count)
#             break
#         count += 1

test_df = pd.read_csv('happy_data_transfer_df.csv')
# 从 一 开始
df_1 = test_df[test_df['num_brush'] == 1]
df_1.to_csv('happy_data_transfer_df_brush_1.csv')

df_2 = test_df[test_df['num_brush'] == 2]
df_2.to_csv('happy_data_transfer_df_brush_2.csv')

df_3 = test_df[test_df['num_brush'] == 3]
df_3.to_csv('happy_data_transfer_df_brush_3.csv')

df_4 = test_df[test_df['num_brush'] == 4]
df_4.to_csv('happy_data_transfer_df_brush_4.csv')

df_5 = test_df[test_df['num_brush'] == 5]
df_5.to_csv('happy_data_transfer_df_brush_5.csv')

