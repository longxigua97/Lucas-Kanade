import cv2
import numpy as np
import sys
image_path = 'flg/image.png'
image_output = "flg/transformed_pic.png"

# 使用cv2.imread()读取图像
image = cv2.imread(image_path)

import numpy as np
from PIL import Image

def affine_transform(image, transform_matrix):
    """
    Apply an affine transform to an image using a given transformation matrix.
    
    Parameters:
    - image: Input image as a NumPy array.
    - transform_matrix: 2x3 affine transform matrix.
    
    Returns:
    - Transformed image as a NumPy array.
    """
    # 获取输入图像的尺寸和通道数
    img_height, img_width = image.shape[:2]
    transformed_image = np.zeros_like(image)
    
    # 遍历输出图像的每个像素
    for y in range(img_height):
        for x in range(img_width):
            # 应用反向变换来找到输入图像中对应的点
            src_x, src_y = np.dot(transform_matrix, [x, y, 1])
            
            # 取最近的整数来避免插值
            src_x, src_y = int(round(src_x)), int(round(src_y))
            
            # 如果计算出的坐标在输入图像的范围内，则复制像素值
            if 0 <= src_x < img_width and 0 <= src_y < img_height:
                transformed_image[y, x] = image[src_y, src_x]
    
    return transformed_image

def resize_image(image, scale_x, scale_y):
    # 创建缩放矩阵
    scale_matrix = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])

    # 计算缩放后的图像尺寸
    new_width = int(image.shape[1])
    new_height = int(image.shape[0])

    # 使用cv2.warpAffine进行缩放
    resized_image = cv2.warpAffine(image, scale_matrix, (new_width, new_height))

    # 返回缩放后的图像
    return resized_image

# 加载图像

image = np.array(Image.open(image_path))
print(image.shape)
# 定义一个仿射变换矩阵（例子：向右平移30像素，向下平移50像素）
transform_matrix = np.array([[1.05, 0.1, 30],
                             [0.1, 1.1, 20]])
#print(transform_matrix.shape)

# 应用仿射变换
transformed_image = affine_transform(image, transform_matrix)
cv2.imshow('transformed_image', transformed_image)

# 显示变换后的图像
#Image.fromarray(transformed_image).show()


# 设定仿射变换前后三点的坐标
# 假设我们要将图像的左上、左下和右上三个点
# 分别移动到新的位置
src_points = np.float32([[0, 0], [0, image.shape[0]], [image.shape[1], 0]])
dst_points = np.float32([[50, 50], [50, image.shape[0]-50], [image.shape[1]-50, 50]])

# 根据这六个点计算仿射变换矩阵
affine_matrix = cv2.getAffineTransform(src_points, dst_points)

#transformed_image = cv2.warpAffine(image, affine_matrix, (image.shape[1]-50, image.shape[0]-50))

transformed_image = resize_image(image,20,20)

# 显示原始和变换后的图像
#cv2.imshow('Original Image', image)
cv2.imshow('Transformed Image', transformed_image)
cv2.waitKey(0)

cv2.imwrite(image_output, transformed_image)
cv2.destroyAllWindows()

