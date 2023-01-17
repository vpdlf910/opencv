import cv2
import numpy as np

## 행렬 처리 함수
# image = cv2.imread("images/opencv_sample1.jpg",cv2.IMREAD_COLOR)
# if image is None : raise Exception("영상파일 읽기 오류 발생")
# resized_image = cv2.resize(image,(780,576))
# cv2.imshow("resized_image",resized_image)
# x_axis = cv2.flip(resized_image, 0)
# y_axis = cv2.flip(resized_image, 1)
# xy_axis = cv2.flip(resized_image, -1)
# rep_image = cv2.repeat(resized_image, 1,2)
# trans_image = cv2.transpose(resized_image)
#
#
# titles= ['image','x_axis','y_axis','xy_axis','rep_image','trans_image']
# for title in titles:
#     cv2.imshow(title, eval(title))
#
# cv2.waitKey(0)

##채널 분리 및 합성
#
# ch0 = np.zeros((2,4), np.uint8 )+10
# ch1 = np.ones((2,4),np.uint8) *20
# ch2 = np.full((2,4),30, np.uint8)
#
# list_bgr = [ch0,ch1,ch2]
# merge_bgr = cv2.merge(list_bgr)
# split_bgr = cv2.split(merge_bgr)
#
# print("split_bgr 행렬 형태",np.array(split_bgr).shape)
# print("merge_bgr 행렬 형태",merge_bgr.shape)
# print("ch[0] = \n%s" %ch0)
# print("ch[1] = \n%s" %ch1)
# print("ch[2] = \n%s\n" %ch2)
# print("[merge_bgr] = \n %s \n"%merge_bgr)
#
# print("[split_bgr[0]] =\n%s" %split_bgr[0])
# print("[split_bgr[1]] =\n%s" %split_bgr[1])
# print('[split_bgr[2]] = \n%s ' %split_bgr[2])

##컬러 채널 분리
# image = cv2.imread("images/opencv_sample.jpg", cv2.IMREAD_COLOR)
# if image is None: raise Exception("영상파일 읽기 오류")
# if image.ndim != 3: raise Exception("컬러 영상 아님")
# resized_image = cv2.resize(image,(768,576))
# bgr =cv2.split(resized_image)
# print("bgr 자료형:", type(bgr), type(bgr[0]),type(bgr[0][0][0]))
# print("bgr 원소개수", len(bgr))
#
# cv2.imshow("image", resized_image)
# cv2.imshow('Blue channel', bgr[0])
# cv2.imshow('Green channel', bgr[1])
# cv2.imshow('Red channel', bgr[2])
# #또다른 방법
# # cv2.imshow('Blue channel', image[:,:,0)
# # cv2.imshow('Green channel', bgr[:,:,1])
# # cv2.imshow('Red channel', bgr[:,:,2])
# cv2.waitKey(0)
##행렬 연산 산술
# m1 = np.full((3,6),10,np.uint8)
# m2 = np.full((3,6),50, np.uint8)
# m_mask = np.zeros(m1.shape, np.uint8)
# m_mask[:,3:] =1
# m_add1 = cv2.add(m1,m2)
# m_add2 = cv2.add(m1, m2, mask=m_mask)
#
# m_div1 = cv2.divide(m1,m2)
# m1 = m1.astype(np.float32)
# m2 = np.float32(m2)
# m_div2 = cv2.divide(m1,m2)
#
# titles = ['m1','m2','m_mask','m_add1','m_add2','m_div1','m_div2']
# for title in titles:
#     print("[%s] = \n%s \n" %(title, eval(title)))
##행렬 지수 및 로그 연산

# v1 = np.array([1,2,3], np.float32)
# v2 = np.array([[1],[2],[3]], np.float32)
# v3 = np.array([[1,2,3]], np.float32)
#
# v_exp = cv2.exp(v1)
# m_exp = cv2.exp(v2)
# m1_exp = cv2.exp(v3)
# v_log = cv2.log(v1)
# m_sqrt = cv2.sqrt(v2)
# m1_pow = cv2.pow(v3,3)
#
#
# print("[v1] 형태: %s 원소: %s"%(v1.shape, v1))
# print('[v2] 형태: %s 원소: %s'%(v2.shape, v2))
# print('[v3] 형태: %s 원소: %s'%(v3.shape, v3))
# print()
#
# print('[v1_exp] 자료형: %s 형태: %s'%(type(v_exp), v_exp.shape) )
# print('[v2_exp] 자료형: %s 형태: %s'%(type(m_exp), m_exp.shape) )
# print('[v3_exp] 자료형: %s 형태: %s'%(type(m1_exp), m1_exp.shape) )
#
# print("[log] =", v_log.T)
# print("[sqrt] =",np.ravel(m_sqrt))
# print('[pow] =',m1_pow.flatten())

##행렬 크기 및 위상 연산

# x = np.array([1,2,3,4,10],np.float32)
# y = np.array([2,5,7,2,9]).astype("float32")
# mag = cv2.magnitude(x,y) #크기 계산
# ang = cv2.phase(x,y) # 각도 계산
# p_mag ,p_ang = cv2.cartToPolar(x,y) #극 좌표로 변환
# x2, y2 = cv2.polarToCart(p_mag,p_ang) #직교좌표로 변환
#
# print("[x] 형태: %s 원소: %s"%(x.shape,x))
# print('[mag] 형태: %s 원소: %s'%(mag.shape, mag))
# print('>>>열백터를 1행에 출력하는 방법')
#
# print('[m_mag] = %s'%mag.T)
# print('[p_mag] = %s'%np.ravel(p_mag) )
# print('[p_amg] = %s'%np.ravel(p_ang))
# print('[x_mat2] = %s' %x2.flatten())
# print('[y_mat2] =%s' %y2.flatten())

##행렬 비트 연산

# image1 = np.zeros((300,300), np.uint8)
# image2 = image1.copy()
#
# h, w = image1.shape[:2]
# cx, cy = w//2, h//2
#
# cv2.circle(image1, (cx,cy), 100,255,-1)
# cv2.rectangle(image2,(0,0,cx,h),255,-1)
#
# image3 = cv2.bitwise_or(image1, image2)
# image4 = cv2.bitwise_and(image1, image2)
# image5 = cv2.bitwise_xor(image1, image2)
# image6 = cv2.bitwise_not(image1)
#
# cv2.imshow("image1",image1); cv2.imshow('image2',image2)
# cv2.imshow('bitwise_or',image3); cv2.imshow('bitwise_and',image4)
# cv2.imshow('bitwise_xor',image5); cv2.imshow('bitwise_not',image6)
# cv2.waitKey(0)

##행렬 비트 연산2
#
# image = cv2.imread('images/opencv_sample.jpg',cv2.IMREAD_COLOR)
# logo = cv2.imread('images/opencv_sample1.jpg', cv2.IMREAD_COLOR)
# if image is None or logo is None: raise Exception('영상파일 읽기 오류')
#
# masks = cv2.threshold(logo, 220,255,cv2.THRESH_BINARY)[1] #로고 영상 이진화
# masks = cv2.split(masks)
# fg_pass_mask = cv2.bitwise_or(masks[0],masks[1])
# fg_pass_mask = cv2.bitwise_or(masks[2], fg_pass_mask) #전경 통과 마스크
# bg_pass_mask = cv2.bitwise_not(fg_pass_mask) #배경 통과 마스크
#
# (H,W), (h,w) = image.shape[:2], logo.shape[:2]
# x, y = (W-w)//2 , (H-h)//2
# roi = image[y:y+h, x:x+w]
#
# foreground = cv2.bitwise_and(logo, logo, mask = fg_pass_mask)
# background = cv2.bitwise_and(roi,roi, mask =bg_pass_mask)
# dst = cv2.add(background,foreground)
# image[y:y+h, x:x+w] = dst
# cv2.imshow('backgoround', background)
# cv2.imshow('foregorund', foreground)
# cv2.imshow('dst',dst)
# cv2.imshow('image',image)
# cv2.waitKey(0)
#
## 행렬 절대값 및 차분 연산
# image1 = cv2.imread('images/opencv_sample.jpg', cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread('images/opencv_sample1.jpg', cv2.IMREAD_GRAYSCALE)
# if image1 is None or image2 is None: raise Exception("영상 파일 읽기 오류")
#
# res_image1 = cv2.resize(image1,(762,568))
# res_image2 = cv2.resize(image2,(762,568))
# dif_img1 = cv2.subtract(res_image1, res_image2)
# dif_img2 = cv2.subtract(np.int16(res_image1),np.int16(res_image2))
# abs_dif1 = np.absolute(dif_img2).astype('uint8')
# abs_dif2 = cv2.absdiff(res_image1,res_image2)
#
# x,y,w,h = 100, 150,7,3
# print('[dif_img1(roi) uint8] = \n%s\n'% dif_img1[y:y+h, x:x+w])
# print('[dif_img2(roi) uint16] = \n%s\n'% dif_img2[y:y+h, x:x+w])
# print('[abs_dif1(roi)] = \n%s\n'% abs_dif1[y:y+h, x:x+w])
# print('[abs_dif2(roi)] = \n%s\n'% abs_dif2[y:y+h, x:x+w])
#
# titles = ['res_image1','res_image2','dif_img1','dif_img2','abs_dif1','abs_dif2']
# for title in titles:
#     cv2.imshow(title, eval(title))
# cv2.waitKey(0)

##행렬 최소값 및 최대값 연산

# data =[ 10,200,5,7,9,15,35,60,80,170,100,2,55,37,70]
# m1 = np.reshape(data, (3,5))
# m2 = np.full((3,5), 50)
# m_min = cv2.min(m1, 30)
# m_max = cv2.max(m1,m2)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(m1)
# print('[m1] = \n%s\n'% m1)
# print('[m_min] = \n%s\n'% m_min)
# print('[m_max] = \n%s\n'% m_max)
# print('m1 행렬 최소값 좌료%s, 최소값: %d' %(min_loc,min_val))
# print('m1 행렬 최대값 좌료%s, 최대값: %d' %(max_loc,max_val))

##영상 최소값 최대값 연산

# image =cv2.imread('images/opencv_sample.jpg',cv2.IMREAD_GRAYSCALE)
# if image is None: raise Exception("영상파일 읽기 오류 발생")
# min_val, max_val,_,_ =cv2.minMaxLoc(image)
#
# ratio = 255/ (max_val- min_val)
# dst = np.round((image - min_val) *ratio).astype('uint8')
# min_dst, max_dst, _,_ = cv2.minMaxLoc(dst)
#
# print('원본 영상 최솟값= %d, 최댓값 =%d' %(min_val, max_val))
# print('수정 영상 최솟값 =%d, 최댓값 =%d' %(min_dst, max_dst))
# cv2.imshow("image", image)
# cv2.imshow('dst', dst)
# cv2.waitKey(0) ##위 과정을 거치면 어두운 영상이 평균치만큼 밝아진다.

##행렬 합/ 평균 연산

# image = cv2.imread("images/opencv_sample.jpg", cv2.IMREAD_COLOR)
# if image is None: raise Exception("영상파일 읽기 오류 발생")
# mask = np.zeros(image.shape[:2],np.uint8)
# mask[60:160,20:120] = 255
#
# sum_value = cv2.sumElems(image) #채널별 합 튜플로 반환
# mean_value1 = cv2.mean(image) # 채널별 평균 튜플로 반환
# mean_value2 = cv2.mean(image, mask) #마스크 영역 만 평균을 구함
#
# print("sum_value 자료형:",type(sum_value),type(sum_value[0]))
# print('[sum_value] =', sum_value)
# print('[mean_value1] =',mean_value1)
# print('[mean_value2] =', mean_value2)
# print()
#
# mean, stddev = cv2.meanStdDev(image)
# mean2, stddev2 = cv2.meanStdDev(image, mask = mask)
# print('mean 자료형:', type(mean), type(mean[0][0]))
# print('[mean] =', mean.flatten())
# print('[stddev] =', stddev.flatten())
# print()
#
# print('[mean2] =', mean2.flatten())
# print('[stddev2] =', stddev2.flatten())
# cv2.imshow('image', image)
# cv2.imshow('mask', mask)
# cv2.waitKey(0)

##행렬 원소 정렬

# m = np.random.randint(0,100,15).reshape(3,5)
#
# sort1 = cv2.sort(m,cv2.SORT_EVERY_ROW)
# sort2 = cv2.sort(m,cv2.SORT_EVERY_COLUMN)
# sort3 = cv2.sort(m,cv2.SORT_EVERY_ROW + cv2.SORT_DESCENDING)
# sort4 = np.sort(m, axis =1)
# sort5 = np.sort(m, axis =0)
# sort6 = np.sort(m, axis =1)[:,::-1]
#
# titles = ['m','sort1','sort2','sort3','sort4','sort5','sort6']
# for title in titles:
#     print('[%s] = \n%s\n' %(title, eval(title)))

## 정렬 인덱스 반환

# m = np.random.randint(0,100,15).reshape(3,5)
# m_sort1 = cv2.sortIdx(m,cv2.SORT_EVERY_ROW)
# m_sort2 = cv2.sortIdx(m,cv2.SORT_EVERY_COLUMN)
# m_sort3 = np.argsort(m,axis =0)
# ##정렬된 후의 정렬되기 전 인덱스 위치를 반환
# print('[m1] = \n%s\n'% m)
# print( '[m_sort1] = \n%s\n' % m_sort1)
# print( '[m_sort2] = \n%s\n' % m_sort3)
# print( '[m_sort3] = \n%s\n' % m_sort2)

##cv2.sortIdx()를 활용한 사각형 크기 정렬

# def print_rects(rects):
#     print('-' * 46)
#     print('사각형 원소\t\t랜덤 사각형 정보\t 크기')
#     print('-' * 46)
#     for i ,(x,y,w,h,a) in enumerate(rects):
#         print('rects[%i] = [(%3d,%3d) from (%3d,%3d)] %5d' %(i,x,y,w,h,a))
#
# rands = np.zeros((5,5), np.uint16)
# starts = cv2.randn(rands[:,:2], 100,50)
# ends = cv2.randn(rands[:,2:-1], 300,50)
# sizes = cv2.absdiff(starts, ends)
# areas = sizes[:,0] *sizes[:,1]
# rects = rands.copy()
# rects[:,2:-1] = sizes
# rects[:,-1] = areas
# idx = cv2.sortIdx(areas, cv2.SORT_EVERY_COLUMN).flatten()
# print_rects(rects)
# print_rects(rects[idx.astype('int')])

##행렬 축소 연산

# m = np.random.rand(3,5) * 1000//10
#
# reduce_sum = cv2.reduce(m, dim = 0, rtype =cv2.REDUCE_SUM)
# reduce_avg = cv2.reduce(m, dim = 1, rtype =cv2.REDUCE_AVG)
# reduce_max = cv2.reduce(m, dim = 0, rtype =cv2.REDUCE_MAX)
# reduce_min = cv2.reduce(m, dim = 1, rtype =cv2.REDUCE_MIN)
#
# print("[m] = \n%s\n" % m)
# print('[m_reduce_sum] =', reduce_sum.flatten())
# print('[m_reduce_avg] =', reduce_avg.flatten())
# print('[m_reduce_max] =', reduce_max.flatten())
# print("[m_reduce_min] =", reduce_min.flatten())

##행렬 곱 연산
# src1 = np.array([1,2,3,1,2,3], np.float32).reshape(2,3)
# src2 = np.array([1,2,3,4,5,6], np.float32).reshape(2,3)
# src3 = np.array([1,2,1,2,1,2], np.float32).reshape(3,2)
# alpha, beta = 1.0,1.0
#
# dst1 = cv2.gemm(src1,src2,alpha,None, beta, flags =cv2.GEMM_1_T)
# dst2 = cv2.gemm(src1,src2,alpha,None, beta, flags =cv2.GEMM_2_T)
# dst3 = cv2.gemm(src1,src3,alpha,None, beta)
# titles = ['src1','src2','src3','dst1','dst2','dst3']
# for title in titles:
#     print('[%s] = \n%s\n'%(title, eval(title)))

##cv2.gemm()을 이용한 회전 변환

# theta = 20 * np.pi /180
# rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],np.float32)
# pts1 =np.array([(250,30),(400,70),(350,250),(150,200)],np.float32)
# pts2 = cv2.gemm(pts1,rot_mat, 1, None, 1,flags=cv2.GEMM_2_T)
#
# for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
#     print('pts1[%d] = %s, pst2[%d] = %s'%(i,pt1,i,pt2))
#
# image = np.full((400,500,3), 255 ,np.uint8)
# cv2.polylines(image,[np.int32(pts1)], True,(0,255,0),2)
# cv2.polylines(image,[np.int32(pts2)], True,(255,0,0),3)
# cv2.imshow('image', image)
# cv2.waitKey(0)

## cv2.invert()와 cv2.solve()로 연립방정식 풀이

# data = [3,0,6,-3,4,2,-5,-1,9]
# m1 = np.array(data, np.float32).reshape(3,3)
# m2 = np.array([36,10,28], np.float32)
#
# ret, inv = cv2.invert(m1, cv2.DECOMP_LU)
# if ret:
#     dst1 = inv.dot(m2)
#     dst2 = cv2.gemm(inv, m2, 1,None, 1)
#     _ , dst3 = cv2.solve(m1,m2, cv2.DECOMP_LU)
#     print('[inv] = \n%s\n' % inv)
#     print('[dst1] =', dst1.flatten())
#     print('[dst2] =', dst2.flatten())
#     print('[dst3] =', dst3.flatten())
# else:
#     print('역행렬이 존재하지 않습니다.')

