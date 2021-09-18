import xml.etree.ElementTree as ET
import numpy as np
from pycocotools.coco import COCO
from torch._C import dtype
from tqdm import tqdm
import glob
import sys


"""
Be note that:

this anchor only useful when your YOLO input is not force resized.
Which means, your input image is padding at bottom and right without any
distortion. Other wise this anchor is WRONG!

because we don't using forced resize as input such as 608 or 512, we just using original image size!

anchor in [w, h] order

"""


def iou(box, clusters):
    """
    计算一个ground truth边界盒和k个先验框(Anchor)的交并比(IOU)值。
    参数box: 元组或者数据，代表ground truth的长宽。
    参数clusters: 形如(k,2)的numpy数组，其中k是聚类Anchor框的个数
    返回：ground truth和每个Anchor框的交并比。
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        print("Box has no area")
        return 0
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_


def avg_iou(boxes, clusters):
    """
    计算一个ground truth和k个Anchor的交并比的均值。
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def Iou_Kmeans(boxes, k, dist=np.median):
    """
    利用IOU值进行K-means聚类
    参数boxes: 形状为(r, 2)的ground truth框，其中r是ground truth的个数
    参数k: Anchor的个数
    参数dist: 距离函数
    返回值：形状为(k, 2)的k个Anchor框
    """
    # 即是上面提到的r
    rows = boxes.shape[0]
    # 距离数组，计算每个ground truth和k个Anchor的距离
    distances = np.empty((rows, k))
    # 上一次每个ground truth"距离"最近的Anchor索引
    last_clusters = np.zeros((rows,))
    # 设置随机数种子
    np.random.seed()

    # 初始化聚类中心，k个簇，从r个ground truth随机选k个
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    # 开始聚类
    while True:
        # 计算每个ground truth和k个Anchor的距离，用1-IOU(box,anchor)来计算
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        # 对每个ground truth，选取距离最小的那个Anchor，并存下索引
        nearest_clusters = np.argmin(distances, axis=1)
        # 如果当前每个ground truth"距离"最近的Anchor索引和上一次一样，聚类结束
        if (last_clusters == nearest_clusters).all():
            break
        # 更新簇中心为簇里面所有的ground truth框的均值
        for cluster in range(k):
            clusters[cluster] = dist(
                boxes[nearest_clusters == cluster], axis=0)
        # 更新每个ground truth"距离"最近的Anchor索引
        last_clusters = nearest_clusters

    return clusters


def id2name(coco):
    classes = dict()
    classes_id = []
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']

    for key in classes.keys():
        classes_id.append(key)
    return classes, classes_id


def load_dataset(path, types='voc'):
    dataset = []
    if types == 'voc':
        for xml_file in glob.glob("{} /*xml".format(path)):
            tree = ET.parse(xml_file)
            # 图片高度
            height = int(tree.findtext("./size/height"))
            # 图片宽度
            width = int(tree.findtext("./size/width"))

            for obj in tree.iter("object"):
                # 偏移量
                xmin = int(obj.findtext("bndbox/xmin")) / width
                ymin = int(obj.findtext("bdbox/ymin")) / height
                xmax = int(obj.findtext("bndbox/xmax")) / width
                ymax = int(obj.findtext("bndbox/ymax")) / height
                xmin = np.float64(xmin)
                ymin = np.float64(ymin)
                xmax = np.float64(xmax)
                ymax = np.float64(ymax)
                if xmax == xmin or ymax == ymin:
                    print(xml_file)
                # 将Anchor的长宽放入dateset，运行kmeans获得Anchor
                dataset.append([xmax - xmin, ymax - ymin])

    if types == 'coco':
        coco = COCO(path)
        classes, classes_id = id2name(coco)
        print(classes)
        print('class_ids:', classes_id)

        img_ids = coco.getImgIds()
        print(len(img_ids))

        for imgId in img_ids:
            i = 0
            img = coco.loadImgs(imgId)[i]
            height = img['height']
            width = img['width']
            i = i + 1
            if imgId % 1000 == 0:
                print('processing imgId {}...'.format(imgId))
            annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = coco.loadAnns(annIds)
            # time.sleep(0.2)
            for ann in anns:
                if 'bbox' in ann:
                    bbox = ann['bbox']
                    ann_width = bbox[2]
                    ann_height = bbox[3]

                    # 偏移量
                    # ann_width = np.float64(ann_width / width)
                    # ann_height = np.float64(ann_height / height)
                    dataset.append([ann_width, ann_height])
                else:
                    raise ValueError("coco no bbox -- wrong!!!")
    return np.array(dataset)


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Please send coco json annotation file as first param')
        exit(0)
    else:
        annFile = sys.argv[1]
        clusters = 9
        data = load_dataset(path=annFile, types='coco')
        out = Iou_Kmeans(data, k=clusters)
        print("Boxes: {} ".format(out))
        print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))

        final_anchors = np.around(out[:, 0] * out[:, 1], decimals=2).tolist()
        print("Before Sort area:\n {}".format(final_anchors))
        idx = np.argsort(final_anchors)[::-1]
        print("After Sort area:\n {}".format(sorted(final_anchors)))
        out = out[idx]
        print('Final anchor: {}'.format(
            list([list(np.array(a, dtype=np.int)) for a in out])))
