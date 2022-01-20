import matplotlib.pyplot as plt
import cv2
import numpy as np
from icecream import ic
import utils
import matplotlib.image as mpimg
import math
import pandas as pd

CANVAS_WIDTH = 600
CANVAS_HEIGHT = 500

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

# 工具方法

data = np.load('/Users/chendeen/同步空间/绘画作业/不是人画的-2/4._strokes.npz')
# data = np.load('/Users/chendeen/Desktop/apple_tar/output/apple_strokes.npz')
ic(data.files)
ic(data['x_ctt'])
ic(data['x_ctt'].shape)
# ic(data['x_color'])
ic(data['x_color'].shape)
# ic(data['x_alpha'])
ic(data['x_alpha'].shape)

# params = np.vstack([data['x_ctt'], data['x_color'], data['x_alpha']])
# params = np.stack([data['x_ctt'], data['x_color'], data['x_alpha']], axis=-1)
params = np.concatenate([data['x_ctt'], data['x_color'], data['x_alpha']], axis=-1)
ic(params.shape)

brush_width_seq = [20, 15, 12, 10, 7]
# stroke_width_12 = 20
# stroke_width_10 = 15
# stroke_width_08 = 12
# stroke_width_06 = 10
# # stroke_width_04 = None
# stroke_width_02 = 7

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
def _draw_oilpaintbrush(stroke_params):
    
    # 这些参数都已经得到了
    # xc, yc, w, h, theta, R0, G0, B0, R2, G2, B2, A
    x0, y0, w, h, theta = stroke_params[0:5]
    R0, G0, B0, R2, G2, B2, ALPHA = stroke_params[5:]

    # 参数的处理
    # 可理解为x是比例
    # 从比例到物理大小。绘画的时候按照类似的过程进行即可。
    x0 = _normalize(x0, CANVAS_WIDTH)
    y0 = _normalize(y0, CANVAS_HEIGHT)
    # w h 应也是比例。
    w = (int)(1 + w * CANVAS_WIDTH)
    h = (int)(1 + h * CANVAS_HEIGHT)
    theta = np.pi*theta

    # w, h 可能分别是笔触作为一个矩形的宽和高，在未作旋转之前，和之后。
    if w * h / (CANVAS_WIDTH*CANVAS_HEIGHT) > 0.1:
        if h > w: # 都是读进来的图片形成的矩阵，灰度
            # ic(self.brush_large_vertical)
            brush = brush_large_vertical # brush_type 1
            brush_type = 1
        else:
            # ic(self.brush_large_horizontal)
            brush = brush_large_horizontal # brush_type 2
            brush_type = 2
    else:
        if h > w:
            # ic(self.brush_small_vertical)
            brush = brush_small_vertical # brush_type 3
            brush_type = 3
        else:
            # ic(self.brush_small_horizontal)
            brush = brush_small_horizontal # brush_type 4
            brush_type = 4

    # 把所有参数都输入进去做笔触变换。关键是做了什么变换。这句是关键
    foreground, _ = create_transformed_brush(
        brush, brush_type, CANVAS_WIDTH, CANVAS_HEIGHT,
        x0, y0, w, h, theta, R0, G0, B0, R2, G2, B2)

#***** 求两点间距离*****
def getDist_P2P(Point0,PointA):
    distance=math.pow((Point0[0]-PointA[0]),2) + math.pow((Point0[1]-PointA[1]),2)
    distance=math.sqrt(distance)
    return distance            

        
# 根据参数变换笔触。
# 我需要知道代码里的坐标位置和摆放。
# 线段的头和尾的笛卡尔坐标
num_physical_stroke = 1
num_stroke = 1
num_brush = 1
my_brush_width = brush_width_seq[0]
# 为了保存数据而设置的
data_transfer_df = pd.DataFrame(columns=['num_physical_stroke',	'num_stroke',	'num_brush',	'start_point_x',	'start_point_y',	'end_point_x',	'end_point_y',	'theta_deg',	'theta_rad', 'r', 'g', 'b'])

def create_transformed_brush(brush, brush_type, canvas_w, canvas_h,
                      x0, y0, w, h, theta, R0, G0, B0, R2, G2, B2):
    global num_stroke
    global my_brush_width
    global num_brush
    global num_physical_stroke
    global data_transfer_df

    # brush_alpha = np.stack([brush, brush, brush], axis=-1)
    # brush_alpha = (brush_alpha > 0).astype(np.float32)
    # brush_alpha = (brush_alpha*255).astype(np.uint8)

    # ic(brush)

    # 注意这是brush的shape。
    colormap = np.zeros([brush.shape[0], brush.shape[1], 3], np.float32)
    # ii表示渐变
    for ii in range(brush.shape[0]):
        t = ii / brush.shape[0] # t是刚好在0和1之间
        # 这个RGB0, RGB2难不成是模拟笔触首段和尾端的颜色渐变。
        this_color = [(1 - t) * R0 + t * R2,
                      (1 - t) * G0 + t * G2,
                      (1 - t) * B0 + t * B2]
        colormap[ii, :, :] = np.expand_dims(this_color, axis=0)

    brush = np.expand_dims(brush, axis=-1).astype(np.float32) / 255.
    # imshow()显示图像时对double型是认为在0~1范围内，即大于1时都是显示为白色，
    # 而imshow显示uint8型时是0~255范围。
    brush = (brush * colormap * 255).astype(np.uint8) # 渐变成功
    # plt.imshow(brush) 
    # plt.show()

    # 位移和旋转。
    M1 = utils.build_transformation_matrix([-brush.shape[1]/2, -brush.shape[0]/2, 0])
    M2 = utils.build_scale_matrix(sx=w/brush.shape[1], sy=h/brush.shape[0])
    M3 = utils.build_transformation_matrix([0,0,theta])
    M4 = utils.build_transformation_matrix([x0, y0, 0])

    M = utils.update_transformation_matrix(M1, M2)
    M = utils.update_transformation_matrix(M, M3)
    M = utils.update_transformation_matrix(M, M4)

    # rect_points = np.array([[41, 340], [41, 33], [164, 33], [164, 340], [41, 340]])
    # rect_points = [[41, 340], [41, 33], [164, 33], [164, 340], [41, 340]]
    # rect_img = np.zeros_like(brush)
    # for rect_point in rect_points:
    #     i, j = rect_point
    #     for k, _ in enumerate(rect_img[i][j]):
    #         rect_img[i][j][k] = 255
    # np.save('brush_test.npy', brush)
    # # plt.plot(rect_img)
    
    if brush_type == 1:
        rect_points = np.array([[76, 352], [76, 33], [336, 33], [336, 352], [76, 352]])
        # print('大 竖')
    elif brush_type == 2:
        # print('大 横')
        rect_points = np.array([[76, 352], [76, 33], [336, 33], [336, 352], [76, 352]])
    elif brush_type == 3:
        # print('小 竖')
        rect_points = np.array([[22, 175], [22, 14], [85, 14], [85, 175], [22, 175]])
    elif brush_type == 4:
        # print('小 横')
        rect_points = np.array([[22, 175], [22, 14], [85, 14], [85, 175], [22, 175]])

    # plt.plot(rect_points[:, 0], rect_points[:, 1])
    # plt.imshow(brush) 
    # plt.show()

    brush = cv2.warpAffine(
        brush, M, (canvas_w, canvas_h),
        borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_AREA)
    # brush_alpha = cv2.warpAffine(
    #     brush_alpha, M, (canvas_w, canvas_h),
    #     borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_AREA)

    # rect_img = cv2.warpAffine(
    #     rect_img, M, (canvas_w, canvas_h),
    #     borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_AREA)
    # nonzero_index = np.nonzero(rect_img)
    # ic(len(nonzero_index[0]))
    # assert len(nonzero_index[0]) == 4
    # rect_points_after = list([[nonzero_index[0][i], nonzero_index[1][i]] for i in range(nonzero_index[0])])

    # add ones
    ones = np.ones(shape=(len(rect_points), 1))

    points_ones = np.hstack([rect_points, ones])

    # transform points
    transformed_points = M.dot(points_ones.T).T


    point_dl = transformed_points[0]
    point_ul = transformed_points[1]
    point_ur = transformed_points[2]
    point_dr = transformed_points[3]
    point_mid = (point_ul + point_ur)//2
    # point_mid = np.array([(point_ul[] + point_ur)//2, ])
    point_center = (point_dl + point_ur)//2
    point_center_aux = np.array([point_center[0], 0])
    
    plt.plot(transformed_points[:, 0], transformed_points[:, 1])
    # plt.plot([point_center[0],point_center_aux[0]], [point_center[1],point_center_aux[1]])
    # plt.plot([point_center[0],point_mid[0]], [point_center[1],point_mid[1]])
    new_df = test_df[test_df['num_stroke'] == num_stroke]
    for index, row in new_df.iterrows():
        plt.plot([row['start_point_x'],row['end_point_x']], 
        [row['start_point_y'],row['end_point_y']], color = 
        [row['r'] / 255,row['g'] / 255,row['b']/ 255])

    # for name, group in test_df.groupby('num_stroke'):
    #     print name


    # print()
    # # plt.imshow(rect_img)
    img = mpimg.imread('/Users/chendeen/同步空间/绘画作业/不是人画的-2/4._rendered_stroke_{:04d}.png'.format(num_stroke))
    # num_stroke += 1
    # plt.imshow(brush)
    img = cv2.resize(img, (CANVAS_WIDTH, CANVAS_HEIGHT), cv2.INTER_AREA)
    plt.imshow(img)
    plt.title(f'theta °: {math.degrees(theta)}\nw:{w}, h:{h}')
    plt.show()

    # 要开始得到 dataframe 中的数据了
    my_w = int(getDist_P2P(point_ul, point_ur))

    # num_stroke_12 = 
    # num_stroke_10 = 
    # num_stroke_08 = 
    # num_stroke_06 = 
    # num_stroke_04 = 
    # num_stroke_02 = 

    
    brush_switch_width_seq = [18, 13, 10, 8]

    # 最严苛在前
    # 当小于8时，切换成02号
    # num_brush是5时就不用换笔了
    if num_brush < 5 and my_w < brush_switch_width_seq[num_brush - 1]:
        # ic(f'{my_w} < {brush_switch_width_seq[num_brush - 1]}')
        my_brush_width = brush_width_seq[num_brush]
        # ic(num_stroke)
        # ic(f'my_brush_width = brush_width_seq[{num_brush}]')
        num_brush += 1
    # 当小于18时，切换成10号    
    # 当小于13时，切换成08号    
    # 当小于10时，切换成06号
    
    

    brushes_start = []
    # 把一个stroke分解成多个physical_stroke
    num_micro_strokes =  my_w // my_brush_width
    res = my_w - num_micro_strokes * my_brush_width
    p0 = point_dl
    p2 = point_dr
    for i in range(num_micro_strokes):
        brushes_start.append(p0 + (0.5 + i) * my_brush_width/my_w * (p2 - p0))
    if res:
        brushes_start.append(p2 - (0.5) * my_brush_width/my_w * (p2 - p0))

    brushes_end = []
    p0 = point_ul
    p2 = point_ur
    for i in range(num_micro_strokes):
        brushes_end.append(p0 + (0.5 + i) * my_brush_width/my_w * (p2 - p0))
    if res:
        brushes_end.append(p2 - (0.5) * my_brush_width/my_w * (p2 - p0))
    
    for i in range(len(brushes_start)):
        data_transfer_df = data_transfer_df.append({'num_physical_stroke': int(num_physical_stroke),
        'num_stroke': int(num_stroke), 'num_brush': int(num_brush), 
        'start_point_x': brushes_start[i][0], 'start_point_y': brushes_start[i][1],
        'end_point_x': brushes_end[i][0], 'end_point_y': brushes_end[i][1],
        'theta_deg': math.degrees(theta), 'theta_rad': theta, 'r': int(R0*255), 'g': int(G0*255), 
        'b': int(B0*255)
        }, ignore_index=True)
        num_physical_stroke += 1
    
    num_stroke += 1
    
    # img = mpimg.imread('/Users/chendeen/同步空间/绘画作业/不是人画的-4/0.jpeg'.format(stroke_count))
    # img = cv2.resize(img, (CANVAS_WIDTH, CANVAS_HEIGHT), cv2.INTER_AREA)
    # plt.imshow(img)
    # plt.show()

    return transformed_points, theta


def check_transfer_data(start_point_x, start_point_y, end_point_x, end_point_y, 
theta_deg, theta_rad, r, g, b):

    # 注意这是brush的shape。
    colormap = np.zeros([brush.shape[0], brush.shape[1], 3], np.float32)
    # ii表示渐变
    for ii in range(brush.shape[0]):
        t = ii / brush.shape[0] # t是刚好在0和1之间
        # 这个RGB0, RGB2难不成是模拟笔触首段和尾端的颜色渐变。
        this_color = [(1 - t) * R0 + t * R2,
                      (1 - t) * G0 + t * G2,
                      (1 - t) * B0 + t * B2]
        colormap[ii, :, :] = np.expand_dims(this_color, axis=0)

    brush = np.expand_dims(brush, axis=-1).astype(np.float32) / 255.
    # imshow()显示图像时对double型是认为在0~1范围内，即大于1时都是显示为白色，
    # 而imshow显示uint8型时是0~255范围。
    brush = (brush * colormap * 255).astype(np.uint8) # 渐变成功
    # plt.imshow(brush) 
    # plt.show()

    # 位移和旋转。
    M1 = utils.build_transformation_matrix([-brush.shape[1]/2, -brush.shape[0]/2, 0])
    M2 = utils.build_scale_matrix(sx=w/brush.shape[1], sy=h/brush.shape[0])
    M3 = utils.build_transformation_matrix([0,0,theta])
    M4 = utils.build_transformation_matrix([x0, y0, 0])

    M = utils.update_transformation_matrix(M1, M2)
    M = utils.update_transformation_matrix(M, M3)
    M = utils.update_transformation_matrix(M, M4)

    # rect_points = np.array([[41, 340], [41, 33], [164, 33], [164, 340], [41, 340]])
    # rect_points = [[41, 340], [41, 33], [164, 33], [164, 340], [41, 340]]
    # rect_img = np.zeros_like(brush)
    # for rect_point in rect_points:
    #     i, j = rect_point
    #     for k, _ in enumerate(rect_img[i][j]):
    #         rect_img[i][j][k] = 255
    # np.save('brush_test.npy', brush)
    # # plt.plot(rect_img)
    
    if brush_type == 1:
        rect_points = np.array([[76, 352], [76, 33], [336, 33], [336, 352], [76, 352]])
        # print('大 竖')
    elif brush_type == 2:
        # print('大 横')
        rect_points = np.array([[76, 352], [76, 33], [336, 33], [336, 352], [76, 352]])
    elif brush_type == 3:
        # print('小 竖')
        rect_points = np.array([[22, 175], [22, 14], [85, 14], [85, 175], [22, 175]])
    elif brush_type == 4:
        # print('小 横')
        rect_points = np.array([[22, 175], [22, 14], [85, 14], [85, 175], [22, 175]])

    # plt.plot(rect_points[:, 0], rect_points[:, 1])
    # plt.imshow(brush) 
    # plt.show()

    brush = cv2.warpAffine(
        brush, M, (canvas_w, canvas_h),
        borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_AREA)
    # brush_alpha = cv2.warpAffine(
    #     brush_alpha, M, (canvas_w, canvas_h),
    #     borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_AREA)

    # rect_img = cv2.warpAffine(
    #     rect_img, M, (canvas_w, canvas_h),
    #     borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_AREA)
    # nonzero_index = np.nonzero(rect_img)
    # ic(len(nonzero_index[0]))
    # assert len(nonzero_index[0]) == 4
    # rect_points_after = list([[nonzero_index[0][i], nonzero_index[1][i]] for i in range(nonzero_index[0])])

    # add ones
    ones = np.ones(shape=(len(rect_points), 1))

    points_ones = np.hstack([rect_points, ones])

    # transform points
    transformed_points = M.dot(points_ones.T).T


    # plt.plot(transformed_points[:, 0], transformed_points[:, 1])
    point_dl = transformed_points[0]
    point_ul = transformed_points[1]
    point_ur = transformed_points[2]
    point_dr = transformed_points[3]
    point_mid = (point_ul + point_ur)//2
    # point_mid = np.array([(point_ul[] + point_ur)//2, ])
    point_center = (point_dl + point_ur)//2
    point_center_aux = np.array([point_center[0], 0])
    # plt.plot([point_center[0],point_center_aux[0]], [point_center[1],point_center_aux[1]])
    # plt.plot([point_center[0],point_mid[0]], [point_center[1],point_mid[1]])
    # # print()
    # # # plt.imshow(rect_img)
    # img = mpimg.imread('/Users/chendeen/同步空间/绘画作业/不是人画的-2/4._rendered_stroke_{:04d}.png'.format(stroke_count))
    # stroke_count += 1
    # # plt.imshow(brush)
    # img = cv2.resize(img, (CANVAS_WIDTH, CANVAS_HEIGHT), cv2.INTER_AREA)
    # plt.imshow(img)
    # plt.title(f'theta °: {math.degrees(theta)}\nw:{w}, h:{h}')
    # plt.show()

    # 要开始得到 dataframe 中的数据了
    my_w = int(getDist_P2P(point_ul, point_ur))

    # num_stroke_12 = 
    # num_stroke_10 = 
    # num_stroke_08 = 
    # num_stroke_06 = 
    # num_stroke_04 = 
    # num_stroke_02 = 

    
    brush_switch_width_seq = [18, 13, 10, 8]

    # 最严苛在前
    # 当小于8时，切换成02号
    # num_brush是5时就不用换笔了
    if num_brush < 5 and my_w < brush_switch_width_seq[num_brush - 1]:
        # ic(f'{my_w} < {brush_switch_width_seq[num_brush - 1]}')
        my_brush_width = brush_width_seq[num_brush]
        # ic(num_stroke)
        # ic(f'my_brush_width = brush_width_seq[{num_brush}]')
        num_brush += 1
    # 当小于18时，切换成10号    
    # 当小于13时，切换成08号    
    # 当小于10时，切换成06号
    
    

    brushes_start = []
    # 把一个stroke分解成多个physical_stroke
    num_micro_strokes =  my_w // my_brush_width
    res = my_w - num_micro_strokes * my_brush_width
    p0 = point_dl
    p2 = point_dr
    for i in range(num_micro_strokes):
        brushes_start.append(p0 + (0.5 + i) * my_brush_width/my_w * (p2 - p0))
    if res:
        brushes_start.append(p2 - (0.5) * my_brush_width/my_w * (p2 - p0))

    brushes_end = []
    p0 = point_ul
    p2 = point_ur
    for i in range(num_micro_strokes):
        brushes_end.append(p0 + (0.5 + i) * my_brush_width/my_w * (p2 - p0))
    if res:
        brushes_end.append(p2 - (0.5) * my_brush_width/my_w * (p2 - p0))
    
    # for i in range(len(brushes_start)):
    #     data_transfer_df = data_transfer_df.append({'num_physical_stroke': num_physical_stroke,
    #     'num_stroke': num_stroke, 'num_brush': num_brush, 
    #     'start_point_x': brushes_start[i][0], 'start_point_y': brushes_start[i][1],
    #     'end_point_x': brushes_end[i][0], 'end_point_y': brushes_end[i][1],
    #     'theta_deg': math.degrees(theta), 'theta_rad': theta
    #     }, ignore_index=True)
    #     num_physical_stroke += 1
    
    # num_stroke += 1
    
    # brushes_to_save
    # stroke_width_to_save.append(getDist_P2P(point_ul, point_ur))

    # img = mpimg.imread('/Users/chendeen/同步空间/绘画作业/不是人画的-4/0.jpeg'.format(stroke_count))
    # img = cv2.resize(img, (CANVAS_WIDTH, CANVAS_HEIGHT), cv2.INTER_AREA)
    # plt.imshow(img)
    # plt.show()




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
            # 在这里保存所有的需要的结果 -- 实际笔画
            _draw_oilpaintbrush(stroke_params)
    
    # with open('stroke_width_test.txt', 'w') as f:
    #     for item in stroke_width_to_save:
    #         f.write("%s\n" % item)

    data_transfer_df.to_csv('happy_data_transfer_df.csv')


# def _check(v):
#     # # 不知为何只取了其第1个维度第0个位置的数据。
#     # ic(v)
#     # # 画的时候只需原样读出来，传进来就行
#     # v = v[0,:,:]

#     # for index, row in df.iterrows():  # for each stroke
        
#         # num_physical_stroke, num_stroke, num_brush, start_point_x, \
#             # start_point_y, end_point_x, end_point_y, theta_deg, theta_rad = row
#         # if check_stroke(stroke_params):
#             # 画笔触。
#             # 看看画出的一个笔触的情况吧
#             # 在这里保存所有的需要的结果 -- 实际笔画
#         check_transfer_data(stroke_params)
    
#     # with open('stroke_width_test.txt', 'w') as f:
#     #     for item in stroke_width_to_save:
#     #         f.write("%s\n" % item)


#     my_brush_width
#     data_transfer_df

# _check()
test_df = pd.read_csv('happy_data_transfer_df.csv')


_render(params)


# for name, group in df.groupby('num_stroke'):
#     # print name
#     # print group




plt.show()