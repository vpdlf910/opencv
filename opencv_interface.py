import numpy as np
import cv2

##윈도우 이동
# image = np.zeros((200, 400), np.uint8) # 200행 400열을 생성
# image[:] = 200 # 색상을 회색으로
#
# title1 , title2 = 'Position1' , 'Position2'
# cv2.namedWindow(title1, cv2.WINDOW_AUTOSIZE) #크리 조정 옵션
# cv2.namedWindow(title2)
# cv2.moveWindow(title1, 150,150) #윈도우 기준 좌측 상단으로 좌표이동
# cv2.moveWindow(title2,400,50)
#
# cv2.imshow(title1, image)
# cv2.imshow(title2, image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##윈도우의 크기 변경
# image = np.zeros((200, 300), np.uint8)
# image.fill(255)
# title1, title2 = "AUTOSIZE" , "NORMAL"
# cv2.namedWindow(title1, cv2.WINDOW_AUTOSIZE)
# cv2.namedWindow(title2, cv2.WINDOW_NORMAL)
#
# cv2.imshow(title1, image)
# cv2.imshow(title2, image)
# cv2.resizeWindow(title1, 400, 300)
# cv2.resizeWindow(title2, 400,300)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

