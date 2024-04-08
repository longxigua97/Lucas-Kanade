import cv2
import numpy as np
import pandas as pd
import sys

def resize_image(image, scale_matrix):
    # 创建缩放矩阵
    #scale_matrix = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])

    # 计算缩放后的图像尺寸
    new_width = int(image.shape[1])
    new_height = int(image.shape[0])

    # 使用cv2.warpAffine进行缩放
    resized_image = cv2.warpAffine(image, scale_matrix, (new_width, new_height))

    # 返回缩放后的图像
    return resized_image

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

def steepest_descent(grad_x, grad_y, x, y):
    
    
    grad_fusion = np.zeros([1,2])
    Jacobian = np.zeros([2,6])
    
    #数据格式准备
    grad_fusion[0,0] = grad_x[x,y]
    grad_fusion[0,1] = grad_y[x,y]
    
    """Jacobian 
        x,  0,  y,  0,  1,  0
        0,  x,  0,  y,  0,  1
    """
    Jacobian[0,0] = y
    Jacobian[0,1] = 0
    Jacobian[0,2] = x
    Jacobian[0,3] = 0
    Jacobian[0,4] = 1
    Jacobian[0,5] = 0
    
    Jacobian[1,0] = 0
    Jacobian[1,1] = y
    Jacobian[1,2] = 0
    Jacobian[1,3] = x
    Jacobian[1,4] = 0
    Jacobian[1,5] = 1
    
    return np.matmul(grad_fusion, Jacobian)
    

    
    

if (1):
    image_path = 'flg/image.png'
    image_output = "flg/transformed_pic.jpeg"

    # 使用cv2.imread()读取图像
    image_Original = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)


    #image_Original = cv2.resize(image_Original,(100,100))

    """ affine 
        p1+1,     p3,   p5,
          p2,   p4+1,   p6,
    """
    affine_p1 = -0.1
    affine_p2 = 0.1
    affine_p3 = 0
    affine_p4 = -0.1
    affine_p5 = 0
    affine_p6 = 0


    transform_matrix = np.float32([[affine_p1+1,    affine_p3,   affine_p5],
                                 [affine_p2  ,  affine_p4+1,   affine_p6]])
    
    affine_matrix = np.float32([[1, 0, 0],
                                [0, 1, 0]])
    

    #image template 
    image_template_centerx = 75
    image_template_centery = 150
    side_length = 100

    

    image_w = image_Original[60:160,30:130]

    image_template = resize_image(image_w, transform_matrix)
      
    # 计算x方向上的梯度
    grad_x = cv2.Sobel(image_w, cv2.CV_64F, 1, 0, ksize=1)
    print(grad_x.shape)

    # 计算y方向上的梯度
    grad_y = cv2.Sobel(image_w, cv2.CV_64F, 0, 1, ksize=1)
    print(grad_y.shape) 
    
    # Compute the Hessian matrix using Equation
    # 获取输入图像的尺寸和通道数
    img_height, img_width = grad_x.shape[:2]
    i = 0
    while(i<10):
        if(i==0):
            affined_image = resize_image(image_w, affine_matrix)
        else:
            affined_image = resize_image(image_w, affine_matrix)
        #计算模板图和affine图之间的差异
        if affined_image.shape != image_w.shape:
            print("Error: 图像尺寸不匹配。")
        else:
        # 计算两个图像的差异
            difference_image = cv2.absdiff(image_template, affined_image)
        
        cv2.imshow('Original',image_w)
        cv2.imshow('template',image_template)
        cv2.imshow('affine',affined_image)
        cv2.imshow('difference',difference_image)  
        cv2.waitKey(0)  
        
        Hessian_matrix = np.zeros([6,6])
        Descent = np.zeros([6,1])
        # 遍历输出图像的每个像素
        for y in range(img_height):
            for x in range(img_width):
                # gradient * Jacobian
                    Steepest_Descent = steepest_descent(grad_x, grad_y, x, y)
                    #print(Steepest_Descent.shape) 
                    # H
                    Hessian = np.matmul(np.transpose(Steepest_Descent), Steepest_Descent)
                    #print(Hessian.shape)
                    
                    Hessian_matrix = Hessian_matrix + Hessian
                    #print(Hessian_matrix.shape)
                    
                    Descent = Descent + Steepest_Descent.T * difference_image[y,x]
                    #print(Descent.shape)    
        #print(Hessian_matrix)  
        #print(Descent)
        
        Delta_P = np.matmul(np.linalg.inv(Hessian_matrix),Descent)  
        print(Delta_P)
        
        affine_matrix[0,0] = affine_matrix[0,0] + Delta_P[0,0]  #p1
        affine_matrix[1,0] = affine_matrix[1,0] + Delta_P[1,0]  #p2
        affine_matrix[0,1] = affine_matrix[0,1] + Delta_P[2,0]  #p3
        affine_matrix[1,1] = affine_matrix[1,1] + Delta_P[3,0]  #p4
        affine_matrix[0,2] = affine_matrix[0,2] + Delta_P[4,0]  #p5
        affine_matrix[1,2] = affine_matrix[1,2] + Delta_P[5,0]  #p6
        
        print(affine_matrix)
        
        
          
            
        i = i+1    
        #print(np.transpose(Steepest_Descent))



    
    
    cv2.destroyAllWindows()
