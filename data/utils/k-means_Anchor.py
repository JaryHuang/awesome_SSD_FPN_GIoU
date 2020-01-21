
# coding=utf-8
# k-means for anchors
# 通过k-means 需要的anchors的尺寸
import numpy as np
import os
import xml.etree.ElementTree as ET
# 定义Box类，描述bounding box的坐标
class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


# 计算两个box在某个轴上的重叠部分
# x1是box1的中心在该轴上的坐标
# len1是box1在该轴上的长度
# x2是box2的中心在该轴上的坐标
# len2是box2在该轴上的长度
# 返回值是该轴上重叠的长度
def overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left


# 计算box a 和box b 的交集面积
# a和b都是Box类型实例
# 返回值area是box a 和box b 的交集面积
def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area


# 计算 box a 和 box b 的并集面积
# a和b都是Box类型实例
# 返回值u是box a 和box b 的并集面积
def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


# 计算 box a 和 box b 的 iou
# a和b都是Box类型实例
# 返回值是box a 和box b 的iou
def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)


# 使用k-means ++ 初始化 centroids，减少随机初始化的centroids对最终结果的影响
# boxes是所有bounding boxes的Box对象列表
# n_anchors是k-means的k值
# 返回值centroids 是初始化的n_anchors个centroid
def init_centroids(boxes,n_anchors):
    centroids = []          #save the final n_anchors boxes
    boxes_num = len(boxes)  #all the number of boxes

    centroid_index = np.random.choice(boxes_num, 1)  #random to selected a anchor
    centroids.append(boxes[centroid_index])          

    print(centroids[0].w,centroids[0].h)

    for centroid_index in range(0,n_anchors-1):    #circle selected the n_anchor

        sum_distance = 0 
        distance_thresh = 0
        distance_list = []
        cur_sum = 0

        for box in boxes:                    #circle all the boxes
            min_distance = 1
            for centroid_i, centroid in enumerate(centroids):
                distance = (1 - box_iou(box, centroid))
                if distance < min_distance:
                    min_distance = distance
            sum_distance += min_distance
            distance_list.append(min_distance)  

        distance_thresh = sum_distance*np.random.random()

        for i in range(0,boxes_num):
            cur_sum += distance_list[i]
            if cur_sum > distance_thresh:
                centroids.append(boxes[i])
                print(boxes[i].w, boxes[i].h)
                break

    return centroids


# 进行 k-means 计算新的centroids
# boxes是所有bounding boxes的Box对象列表
# n_anchors是k-means的k值
# centroids是所有簇的中心
# 返回值new_centroids 是计算出的新簇中心
# 返回值groups是n_anchors个簇包含的boxes的列表
# 返回值loss是所有box距离所属的最近的centroid的距离的和
def do_kmeans(n_anchors, boxes, centroids):
    loss = 0
    groups = []
    new_centroids = []
    for i in range(n_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))

    for box in boxes:
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = (1 - box_iou(box, centroid))
            if distance < min_distance:            #the smallest distance between the  boxs and center_box
                min_distance = distance
                group_index = centroid_index
        groups[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(n_anchors):
        new_centroids[i].w /= len(groups[i])
        new_centroids[i].h /= len(groups[i])

    return new_centroids, groups, loss


# 计算给定bounding boxes的n_anchors数量的centroids
# label_path是训练集列表文件地址
# n_anchors 是anchors的数量,
# loss_convergence是允许的loss的最小变化值
# grid_size * grid_size 是栅格数量
# iterations_num是最大迭代次数
# plus = 1时启用k means ++ 初始化centroids
def compute_centroids(label_path,n_anchors,loss_convergence,base_size,scale,iterations_num,plus):

    boxes = []
    label_files = []
    id_list_file = os.path.join(
            label_path, 'ImageSets/Main/train.txt')

    label_files = [id_.strip() for id_ in open(id_list_file)]
    use_difficult=False
    #print(label_files)
    for id_ in label_files:
        anno = ET.parse(
            os.path.join(label_path, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()

        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            box = [int(bndbox_anno.find(tag).text) - 1
                for tag in ('xmin', 'ymin', 'xmax', 'ymax')]
                #for tag in ('ymin', 'xmin', 'ymax', 'xmax')]
            #print(float(box[2]-box[0]), float(box[3]-box[1]))
            boxes.append(Box(0, 0,scale*float(box[2]-box[0]), scale*float(box[3]-box[1])))
    print(max([a.w for a in boxes]),min([a.w for a in boxes]),max([a.h for a in boxes]),min([a.h for a in boxes]))
    
    if plus:
        centroids = init_centroids(boxes, n_anchors)
    else:
        centroid_indices = np.random.choice(len(boxes), n_anchors)
        centroids = []
        for centroid_index in centroid_indices:
            centroids.append(boxes[centroid_index])

    # iterate k-means
    centroids, groups, old_loss = do_kmeans(n_anchors, boxes, centroids)
    iterations = 1
    while (True):
        centroids, groups, loss = do_kmeans(n_anchors, boxes, centroids)
        iterations = iterations + 1
        print("loss = %f" % loss)
        if abs(old_loss - loss) < loss_convergence or iterations > iterations_num:
            break
        old_loss = loss

        for centroid in centroids:
            print(centroid.w, centroid.h)
    print("#######################################")
    print("#######################################\n")
    # print result
    for centroid in centroids:
        
        ratio = float(centroid.w/centroid.h)
        print(centroid.w,centroid.h,np.sqrt(ratio))
        print("k-means result：base={:.2f},ratio={:.2f}".format(\
            float(centroid.w/np.sqrt(ratio)),ratio))

label_path = "../CrackData/crack"
n_anchors = 9
loss_convergence = 1e-5
imgsize_scale = 300/1400
iterations_num = 5000
plus = 0
base_size = 10
compute_centroids(label_path,n_anchors,loss_convergence,base_size,imgsize_scale,iterations_num,plus)
#anchor scale = [3]
#ratio = [0.08,0.15,0.45]