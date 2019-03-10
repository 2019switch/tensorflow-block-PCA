import numpy as np
from PIL import Image
np.seterr(divide='ignore', invalid='ignore')
'''
imgs 是三维的图像矩阵，第一维是图像的个数
'''
def TwoDPCA(imgs, dim):
    a,b,c = imgs.shape                      #a=1;b=28;c=28
    average1,average2,average3,average4 = np.zeros((b//2,c//2))
    for i in range(a):
        average1 += imgs[i,:b//2,:c//2]/(a*1.0)   #lefttop
        average2 += imgs[i,:b//2,c//2:]/(a*1.0)   #righttop
        average3 += imgs[i,b//2:,:c//2]/(a*1.0)   #letfdown
        average4 += imgs[i,b//2:,c//2:]/(a*1.0)   #rightdown
    G_t = np.zeros((c//2,c//2))            #14*14
    for j in range(a):
        img1 = imgs[j,:b//2,:c//2]
        temp1 = img1-average1
        img2 = imgs[j,:b//2,c//2:]
        temp2 = img2 - average2
        img3 = imgs[j,b//2:,:c//2]
        temp3 = img3 - average3
        img4 = imgs[j,b//2:,c//2:]
        temp4 = img4 - average4

        G_t = G_t + (np.dot(temp1.T,temp1)+np.dot(temp2.T,temp2)
                     + np.dot(temp3.T, temp3)+np.dot(temp4.T,temp4))/(a*1.0)
    w,v = np.linalg.eigh(G_t)           #w:特征值;v:特征向量
    # print('v_shape:{}'.format(v.shape))
    w = w[::-1]  #特征值倒序
    v = v[::-1]  #特征向量：14*14
    '''
    for k in range(c):
        # alpha = sum(w[:k])*1.0/sum(w)
        alpha = 0
        if alpha >= p:
            u = v[:,:k]
            break
    '''
    print('alpha={}'.format(sum(w[:dim]*1.0/sum(w))))
    u = v[:,:dim]   #取前k个最大的特征值对应的特征向量：14*10
    print('u_shape:{}'.format(u.shape))  # u_shape:(14, 10)
    return u  # u是投影矩阵


def TTwoDPCA(imgs, dim):
    u = TwoDPCA(imgs, dim)
    a1,b1,c1 = imgs.shape
    img = []
    for i in range(a1):
        temp11 = imgs[i,:,:]    # temp1 = np.dot(imgs[i,:,:],u) imgs[i,:b2//2,:c2//2]
        img.append(temp11.T)
    img = np.array(img)
    uu = TwoDPCA(img, dim)
    print('uu_shape:{}'.format(uu.shape))  # uu_shape:(28, 10)
    return u,uu  # uu是投影矩阵


def PCA2D_2D(samples, row_top, col_top):
    '''samples are 2d matrices'''
    a2,b2,c2 = samples.shape
    size = samples[0,:b2//2,:c2//2].shape  #28*28
    # m*n matrix
    mean = np.zeros(size)
    for s in samples:
        mean1 = mean + s[:b2//2,:c2//2]
        mean2 = mean + s[:b2//2,c2//2:]
        mean3 = mean + s[b2//2:,:c2//2]
        mean4 = mean + s[b2//2:,c2//2:]
    # get the mean of all samples
    mean1 /= float(len(samples))
    mean2 /= float(len(samples))
    mean3 /= float(len(samples))
    mean4 /= float(len(samples))

    # n*n matrix
    cov_row = np.zeros((c2//2,c2//2))
    for s in samples:
        diff1 = s[:b2//2,:c2//2] - mean1
        diff2 = s[:b2//2,c2//2:] - mean2
        diff3 = s[b2//2:,:c2//2] - mean3
        diff4 = s[b2//2:,c2//2:] - mean4
        # (A-EA).T*(A-EA)
        cov_row = cov_row + (np.dot(diff1.T, diff1)+np.dot(diff2.T, diff2)
                             +np.dot(diff3.T, diff3)+np.dot(diff4.T, diff4))
    cov_row /= float(len(samples))
    # 分解计算协方差矩阵的（特征值 特征向量）
    row_eval, row_evec = np.linalg.eig(cov_row)
    # select the top t evals
    sorted_index = np.argsort(row_eval)
    # using slice operation to reverse
    X = row_evec[:,sorted_index[:-row_top-1 : -1]]

    # m*m matrix
    cov_col = np.zeros((b2//2, b2//2))
    for s in samples:
        diff1 = s[:b2//2,:c2//2] - mean1
        diff2 = s[:b2//2,c2//2:] - mean2
        diff3 = s[b2//2:,:c2//2] - mean3
        diff4 = s[b2//2:,c2//2:] - mean4
        cov_col = cov_col + (np.dot(diff1,diff1.T)+np.dot(diff2, diff2.T)
                           +np.dot(diff3, diff3.T)+np.dot(diff4, diff4.T))
    cov_col /= float(len(samples))
    col_eval, col_evec = np.linalg.eig(cov_col)
    sorted_index = np.argsort(col_eval)
    Z = col_evec[:,sorted_index[:-col_top-1 : -1]]

    return X, Z


def image_2D2DPCA(images, u, uu):  #投影过去的图像
    a, b, c = images.shape         #a=1;b=28;c=28
    new_images = np.ones((a, 2*uu.shape[1], 2*u.shape[1]))  #(1,10,10),dtype = complex
    e, f, g = new_images.shape
    for i in range(a):
        Y = np.dot(uu.T, images[i,:b//2,:c//2])  # UU'Y
        Y = np.dot(Y, u)  # 5*5   UU'YU
        new_images[i,:f//2,:g//2] = Y
        Z = np.dot(uu.T, images[i,:b//2,c//2:])  # UU'Z
        Z = np.dot(Z, u)  # 5*5   UU'ZU
        new_images[i,:f//2,g//2:] = Z
        M = np.dot(uu.T, images[i,b//2:,:c//2])  # UU'M
        M = np.dot(M, u)  # 5*5   UU'MU
        new_images[i,f//2:,:g//2] = M
        N = np.dot(uu.T, images[i,b//2:,c//2:])  # UU'N
        N = np.dot(N, u)  # 5*5   UU'NU
        new_images[i,f//2:,g//2:] = N

    return new_images
# M = np.dot(images[i,:,:].T, uu)
# N = (np.dot(images[i,:,:], u))
# Y = M + N   #28*10

# if __name__ == '__main__':
#     im = Image.open('./bloodborne2.jpg')
#     im_grey = im.convert('L')   #转化为灰度图
#     # im_grey.save('a.png')
#     a, b = np.shape(im_grey)
#     data = im_grey.getdata()
#     data = np.array(data)
#     data2 = data.reshape(1, a, b)
#     # 裁剪的代码
#     original = im_grey
#     # original.show()
#     width, height = original.size   # Get dimensions
#     # left = width/4
#     left = 0
#     top = 0
#     # top = height/4
#     right =  width/2
#     bottom =  height/2
#     cropped_example = original.crop((left, top, right, bottom))
#     cropped_example.show()
#     # img1= tf.image.crop_and_resize(
#     #     im_grey,
#     #     boxes,
#     #     box_ind,
#     #     crop_size,
#     #     method='bilinear',
#     #     extrapolation_value=0,
#     #     name=None
#     # )
#     print('data2_shape:{}'.format(data2.shape))
#     u, uu = TTwoDPCA(data2, 10)
#     print('data2_2DPCA_u:{}'.format(u.shape))
#     print('data2_2D2DPCA_uu:{}'.format(uu.shape))
#     new_images = image_2D2DPCA(data2, u, uu)
#     print(new_images)
#     print('new_images:{}'.format(new_images.shape))
