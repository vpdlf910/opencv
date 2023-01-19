import numpy as np ,cv2 , time

##행렬 원소 접근 방법1
# def mat_access1(mat):
#     for i in range(mat.shape[0]):
#         for j in range(mat.shape[1]):
#             k = mat[i,j]
#             mat[i,j]= k*2
#
# def mat_access2(mat):
#     for i in range(mat.shape[0]):
#         for j in range(mat.shape[1]):
#             k = mat.item(i,j)
#             mat.itemset((i,j),k*2)
#
# mat1 = np.arange(10).reshape(2,5)
# mat2 = np.arange(10).reshape(2,5)
#
# print('원소 처리 전: \n%s\n'%mat1)
# mat_access1(mat1)
# print('원소 처리 후: \n%s\n'%mat1)
#
# print('원소 처리 전: \n%s\n'%mat2)
# mat_access2(mat2)
# print('원소 처리 후: \n%s\n'%mat2)

##행렬 원소 접근 방법2
#
# def pixel_access1(image):
#     image1 = np.zeros(image.shape[:2], image.dtype)
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             pixel = image[i,j]
#             image1[i,j] = 255 - pixel
#
#     return image1
#
# def pixel_access2(image):
#     image2 = np.zeros(image.shape[:2], image.dtype)
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             pixel = image.item(i,j)
#             image2.itemset((i,j), 255-pixel)
#         return image2
#
# def pixel_access3(image):
#     lut = [255 - i for i in range(256)]
#     lut = np.array(lut, np.uint8)
#     image3 = lut[image]
#     return image3
#
# def pixel_access4(image):
#     image4 = cv2.subtract(255, image)
#     return image4
#
# def pixel_access5(image):
#     image5 = 255 - image
#     return image5
#
# image = cv2.imread('images/write_test1.jpg',cv2.IMREAD_GRAYSCALE)
# if image is None: raise Exception("영상파일 일기 오류")
# def time_check(func, msg):
#     start_time = time.perf_counter()
#     ret_img = func(image)
#     elapsed = (time.perf_counter() - start_time) * 1000
#     print(msg, "수행시간 : %0.2f ms" %elapsed)
#     return ret_img
#
# image1 = time_check(pixel_access1, "[방법1] 직접 접근 방식")
# image2 = time_check(pixel_access2, "[방법2] item() 함수 방식")
# image3 = time_check(pixel_access3, "[방법3] 룩업테이블 방식")
# image4 = time_check(pixel_access4, "[방법4] OpenCV 함수 방식")
# image5 = time_check(pixel_access5, "[방법5] ndarray 연산 방식")

##명암도 영상 생성

# image1 = np.zeros((50,512), np.uint8)
# image2 = np.zeros((50,512), np.uint8)
# rows, cols = image1.shape[:2]
# for i in range(rows):
#     for j in range(cols):
#         image1.itemset((i,j), j//2)
#         image2.itemset((i,j), j// 20*10)
# cv2.imshow('image1', image1)
# cv2.imshow('image2', image2)
# cv2.waitKey(0)
#
##영상 화소값 확인

# image = cv2.imread('images/write_test1.jpg', cv2.IMREAD_GRAYSCALE)
# if image is None: raise Exception("영상파일 읽기 오류")
#
# (x,y),(w,h) = (180,37) ,(15,10)
# roi_img = image[y:y+h, x:x+w]
# print("roi_img",roi_img)
# print('[roi_img] =')
# for row in roi_img:
#     for p in row:
#         print('%4d'%p,end=" ")
#
# print()
# cv2.rectangle(image, (x,y,w,h), 255,1)
# cv2.imshow('image', image)
# cv2.waitKey(0)

##행렬 가감 연산 통한 영상 밝기 변경
#
# image = cv2.imread("images/sample.png", cv2.IMREAD_GRAYSCALE)
# if image is None: raise Exception('영상파일 읽기 오류')
#
# dst1 =cv2.add(image, 100)
# dst2 =cv2.subtract(image, 100)
# dst3 = image +100
# dst4 = image -100
#
# cv2.imshow('original image', image)
# cv2.imshow('dst1- bright:OpenCV', dst1)
# cv2.imshow('dst2- dark:OpenCV', dst2)
# cv2.imshow('dst3- bright:numpy', dst3)
# cv2.imshow('dst4- dark:numpy', dst4)
# cv2.waitKey(0)

##행렬 합과 곱 연산을 통한 영상 합성
#
# image1 = cv2.imread('images/opencv_sample.jpg',cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread('images/opencv_sample1.jpg',cv2.IMREAD_GRAYSCALE)
# if image1 is None or image2 is None: raise Exception('영상파일 읽기 오류')
# image1 = cv2.resize(image1, (762,568))
# image2 = cv2.resize(image2, (762,568))
#
# alpha , beta = 0.6, 0.7
# add_img1 = cv2.add(image1, image2)
# add_img2 = cv2.add(image1 * alpha, image2 *beta)
# add_img2 = np.clip(add_img2, 0,255).astype('uint8')
# add_img3 = cv2.addWeighted(image1,alpha,image2,beta,0)
#
# titles = ['image1','image2','add_img1','add_img2','add_img3']
# for t in titles: cv2.imshow(t, eval(t))
# cv2.waitKey(0)

##영상 대비 변경

# image = cv2.imread('images/opencv_sample1.jpg',cv2.IMREAD_GRAYSCALE)
# if image is None: raise Exception('영상파일 읽기 오류')
#
# image = cv2.resize(image,(762,568))
# noimage = np.zeros(image.shape[:2], image.dtype)
# avg = cv2.mean(image)[0]/2.0
#
# dst1 = cv2.scaleAdd(image, 0.5, noimage)
# dst2 = cv2.scaleAdd(image, 2.0, noimage)
# dst3 = cv2.addWeighted(image, 0.5, noimage, 0,avg)
# dst4 = cv2.addWeighted(image, 2.0, noimage, 0,-avg)
# cv2.imshow('image', image)
# cv2.imshow('dst1 - decrease contrast',dst1)
# cv2.imshow('dst2 - increase contrast',dst2)
# cv2.imshow('dst3 - decrease contrast using average',dst3)
# cv2.imshow('dst4 - increase contrast using average',dst4)
# cv2.waitKey(0)

##영상 히스토그램 계산
##히스토그램은 해당 숫자가 얼마나 많은지에대한 계급도를 나타낸다
# def calc_histo(image, histSize, ranges= [0,256]):
#     hist = np.zeros((histSize[0],1), np.float32)
#     gap = ranges[1] / histSize[0]
#
#     for row in image:
#         for pix in row:
#             idx = int(pix/gap)
#             hist[idx] +=1
#     return hist
#
# image = cv2.imread("images/sample.png",cv2.IMREAD_GRAYSCALE)
# if image is None: raise Exception('영상파일 읽기 오류')
#
# histSize, ranges = [32],[0,256]
# gap = ranges[1]/histSize[0]
# ranges_gap = np.arange(0, ranges[1]+1,gap)
# hist1 = calc_histo(image, histSize, ranges)
# hist2 = cv2.calcHist([image], [0],None, histSize,ranges)
# hist3, bins = np.histogram(image, ranges_gap)
#
# print('User 함수: \n',hist1.flatten())
# print('OpenCV 함수: \n', hist2.flatten())
# print('numpy 함수: \n', hist3.flatten())

##히스토그램 그래프 그리기
#
# def draw_histo(hist, shape=(200,256)):
#     hist_img = np.full(shape, 255, np.uint8)
#     cv2.normalize(hist, hist, 0,shape[0],cv2.NORM_MINMAX)
#     gap = hist_img.shape[1]/hist.shape[0]
#     for i , h in enumerate(hist):
#         x = int(round(i *gap))
#         w = int(round(gap))
#         cv2.rectangle(hist_img,(x,0,w,int(h)),0,cv2.FILLED)
#     return cv2.flip(hist_img,0)
# image = cv2.imread("images/write_test1.jpg",cv2.IMREAD_GRAYSCALE)
# if image is None : raise Exception('영상파일 읽기 오류')
#
# hist = cv2.calcHist([image], [0], None, [32],[0,256])
# hist_img = draw_histo(hist)
# cv2.imshow('image', image)
# cv2.imshow('hist_img',hist_img)
# cv2.waitKey(0)

##색상 히스토그램 그리기
#
# def make_palette(rows):
#     hue = [round(i *180 /rows) for i in range(rows)]
#     hsv = [[[h,255,255]]for h in hue]
#     hsv = np.array(hsv, np.uint8)
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# def draw_hist_hue(hist, shape = [200,256,3]):
#     hsv_palette = make_palette(hist.shape[0])
#     hist_img = np.full(shape, 255, np.uint8)
#     cv2.normalize(hist, hist, 0,shape[0],cv2.NORM_MINMAX)
#
#     gap = hist_img.shape[1] /hist.shape[0]
#     for i, h in enumerate(hist):
#         x, w= int(round(i *gap)), int(round(gap))
#         color = tuple(map(int,hsv_palette[i][0]))
#         cv2.rectangle(hist_img,(x,0,w,int(0)),color,cv2.FILLED)
#     return cv2.flip(hist_img, 0)
#
# image = cv2.imread('images/write_test1.jpg',cv2.IMREAD_COLOR)
# if image is None: raise Exception('영상파일 읽기 오류')
# image = cv2.resize(image, (762,586))
# hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# hue_hist = cv2.calcHist([hsv_img], [0], None, [18],[0,180])
# hue_hist_img = draw_hist_hue(hue_hist,(200,360,3))
#
# cv2.imshow('image', image)
# cv2.imshow('hue_hist',hue_hist)
# cv2.imshow('hue_hist_img',hue_hist_img)
# cv2.waitKey(0)

##히스토그램 스트레칭
#
# def draw_histo(hist, shape=(200,256)):
#     hist_img = np.full(shape, 255, np.uint8)
#     cv2.normalize(hist, hist, 0,shape[0],cv2.NORM_MINMAX)
#     gap = hist_img.shape[1]/hist.shape[0]
#     for i , h in enumerate(hist):
#         x = int(round(i *gap))
#         w = int(round(gap))
#         cv2.rectangle(hist_img,(x,0,w,int(h)),0,cv2.FILLED)
#     return cv2.flip(hist_img,0)
#
# def search_value_idx(hist, bias = 0):
#     for i in range(hist.shape[0]):
#         idx = np.abs(bias -1)
#         if hist[idx] > 0: return idx
#     return -1
#
# image = cv2.imread('images/write_test1.jpg', cv2.IMREAD_GRAYSCALE)
# if image is None: raise Exception('영상파일 읽기 오류')
#
# bsize , ranges = [64], [0,256]
# hist = cv2.calcHist([image],[0],None, bsize,ranges)
#
# bin_width = ranges[1]/bsize[0]
# low = search_value_idx(hist,0) * bin_width
# high = search_value_idx(hist, bsize[0] -1) *bin_width
#
# idx = np.arange(0,256)
# idx = (idx - low)/(high -low) *255
# idx[0:int(low)] =0
# idx[int(high+1):] =255
#
# dst = cv2.LUT(image, idx.astype('uint8'))
# hist_dst = cv2.calcHist([dst], [0],None, bsize, ranges)
# hist_img = draw_histo(hist, (200,360))
# hist_dst_img = draw_histo(hist_dst, (200,360))
#
# print('high_vlue',high)
# print('low_vlue', low)
# cv2.imshow('image', image); cv2.imshow('hist_img',hist_img)
# cv2.imshow('dst',dst); cv2.imshow('hist_dst_img', hist_dst_img)
# cv2.waitKey(0)

