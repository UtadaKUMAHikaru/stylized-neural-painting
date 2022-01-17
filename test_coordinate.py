import matplotlib.pyplot as plt
import cv2
import numpy as np
from icecream import ic

# brush = np.load('brush_test.npy')
# rect_points = np.array([[76, 352], [76, 33], [336, 33], [336, 352], [76, 352]])
# # rect_points = np.array([[41, 340], [41, 33], [164, 33], [164, 340], [41, 340]])
# # plt.plot()
# rect_img = np.zeros_like(brush)
# for rect_point in rect_points:
#     i, j = rect_point
#     for k, _ in enumerate(rect_img[i][j]):
#         rect_img[i][j][k] = 255
# plt.plot(rect_points[:,0], rect_points[:,1])
# plt.imshow(brush) 
# ic(rect_points[:,0])

# 得到旋转后的矩阵，从中得到四个点

CANVAS_WIDTH = 373

data = np.load('/Users/chendeen/同步空间/绘画作业/不是人画的-2/4._strokes.npz')
# data = np.load('/Users/chendeen/Desktop/apple_tar/output/apple_strokes.npz')
ic(data.files)
ic(data['x_ctt'])
ic(data['x_ctt'].shape)
ic(data['x_color'])
ic(data['x_color'].shape)

brush_small_vertical = cv2.imread(
    r'./brushes/brush_fromweb2_small_vertical.png', cv2.IMREAD_GRAYSCALE)
brush_small_horizontal = cv2.imread(
    r'./brushes/brush_fromweb2_small_horizontal.png', cv2.IMREAD_GRAYSCALE)
brush_large_vertical = cv2.imread(
    r'./brushes/brush_fromweb2_large_vertical.png', cv2.IMREAD_GRAYSCALE)
brush_large_horizontal = cv2.imread(
    r'./brushes/brush_fromweb2_large_horizontal.png', cv2.IMREAD_GRAYSCALE)

def check_stroke(stroke_params):
    r_ = 1.0
    
    # stroke的长或宽要 > 0.025
    r_ = max(stroke_params[2], stroke_params[3])
    
    if r_ > 0.025:
        return True
    else:
        return False

def _normalize(x, width):
    return (int)(x * (width - 1) + 0.5)

# 真正的绘画函数。
def _draw_oilpaintbrush(self):
    # 这些参数都已经得到了
    # xc, yc, w, h, theta, R0, G0, B0, R2, G2, B2, A
    x0, y0, w, h, theta = self.stroke_params[0:5]
    R0, G0, B0, R2, G2, B2, ALPHA = self.stroke_params[5:]

    # 参数的处理
    # 可理解为x是比例
    # 从比例到物理大小。绘画的时候按照类似的过程进行即可。
    x0 = _normalize(x0, CANVAS_WIDTH)
    y0 = _normalize(y0, CANVAS_WIDTH)
    # w h 应也是比例。
    w = (int)(1 + w * CANVAS_WIDTH)
    h = (int)(1 + h * CANVAS_WIDTH)
    theta = np.pi*theta

    # w, h 可能分别是笔触作为一个矩形的宽和高，在未作旋转之前，和之后。
    if w * h / (CANVAS_WIDTH**2) > 0.1:
        if h > w: # 都是读进来的图片形成的矩阵，灰度
            # ic(self.brush_large_vertical)
            brush = self.brush_large_vertical
        else:
            # ic(self.brush_large_horizontal)
            brush = self.brush_large_horizontal
    else:
        if h > w:
            # ic(self.brush_small_vertical)
            brush = self.brush_small_vertical
        else:
            # ic(self.brush_small_horizontal)
            brush = self.brush_small_horizontal

    # 把所有参数都输入进去做笔触变换。关键是做了什么变换。这句是关键
    self.foreground, self.stroke_alpha_map = utils.create_transformed_brush(
        brush, self.CANVAS_WIDTH, self.CANVAS_WIDTH,
        x0, y0, w, h, theta, R0, G0, B0, R2, G2, B2)

def _render(v):
        # 不知为何只取了其第1个维度第0个位置的数据。
        ic(v)
        # 画的时候只需原样读出来，传进来就行
        v = v[0,:,:]

        # 开始渲染。
        print('rendering canvas...')
        for i in range(v.shape[0]):  # for each stroke
           
            # 给rderr设置第i个笔触的参数。
            # 核心是这几行
            stroke_params = v[i, :]
            if check_stroke(stroke_params):
                # 画笔触。
                # 看看画出的一个笔触的情况吧
                _draw_oilpaintbrush(stroke_params)

            

        

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



plt.show()