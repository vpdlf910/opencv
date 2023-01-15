import numpy as np
import pandas as pd

#
# variable1 = 100 #정수 변수 선언
# variable2 = 3.14 #실수 변수 선언
# variable3 = -200 #정수 변수 선언
# variable4 = 1.2 + 3.4j #복소수 변수 선언
# variable5 = 'This is Python' #문자열 변수 선언
#
# variable6 = True #bool 변수 선언
# variable7 = float(variable1) #자료형 변경
# variable8 = int(variable2)
#
# print('varialbe1 =' , variable1, type(variable1))
# print('varialbe2 =' , variable2, type(variable2))
# print('varialbe3 =' , variable3, type(variable3))
# print('varialbe4 =' , variable4, type(variable4))
# print('varialbe5 =' , variable5, type(variable5))
# print('varialbe6 =' , variable6, type(variable6))
# print('varialbe7 =' , variable7, type(variable7))
# print('varialbe8 =' , variable8, type(variable8))
#
# list1 = [1,2,3,4,5]
# list2 = [1,1.5,'a','a', '문자열']
# tuple1 = (1,2)
# tuple2 = (1,1.5,'a', 'a','문자열')
# dict1 = {"name" : "배종욱", "email": "daum.net"}
# set1 ,set2 = set(list1), set(tuple2)
# list1[0] =5
# list2.insert(3,'b')
# #tuple1[0] =5 #tuple은 추가나 새롭게 변경 불가
# dict1["email"] = "naver.com"
# print('list1', list1 , type(list1))
# print('list2', list2 , type(list2))
# print('tuple1', tuple1 , type(tuple1))
# print('tuple2', tuple2 , type(tuple2))
# print('dict1', dict1 , type(dict1))
# print('set1', set1 , type(set1))
# print('set2', set2 , type(set2))
# print('intersection', set1 & set2)
#
# title = "서기 1년 1월 1일부터"\
#     '오늘까지'\
#     '일수 구하기'
#
# months = [31,28,31,30,31,30,31,31,30,31,30,31]
# year , month = 2020, 1
# day = 7; ratio =365.2425
# days = (year) -1 * ratio + \
#     sum(months[:month-1]) + day
#
# print(title), print(' - 년:', year), print(' - 월:', month)
# print(' - 일:', day); print(' * 일수 총합:', int(days))
#
# a = [0,1,2,3,4,5,6,7,8,]
# print('a = ', a)
# print('a[:2] ->', a[:2])
# print('a[2::2]' , a[2::2])
# print('a[::-1]', a[::-1])
# print('a[1::-1]', a[1::-1])
# print('a[7:1:-2]' , a[7:1:-2])
# print('a[:-4:-1]', a[:4:-1])
#
# year = 2020
# if (year % 4==0) and (year % 100 != 0):
#     print(year, "는 윤년입니다.")
# elif year % 400 == 0:
#     print(year, "는 윤년입니다.")
# else:
#     print(year, "는 윤년이 아닙니다.")
#
# n =3
# while n >= 0:
#     m = input("Enter a integer: ")
#     if int(m) ==0: break
#     n = n-1
# else:
#     print('4 inputs.')
#
# kor = [70, 80, 90,40,50]
# eng = [90,80,70,70,60]
# sum1 , sum2, sum3 ,sum4 = 0,0,0,0
#
# for i in range(0,5):
#     sum1 = sum1 + kor[i] +eng[i]
#
# for k in kor:
#     sum2 =sum2 +k
# for e in eng:
#     sum3 = sum3 +e
# for i , k in enumerate(kor):
#     sum3 = sum3 + k + eng[i]
# for k, e in zip(kor,eng):
#     sum4 = sum4 + k + e
#
# for i , k in enumerate(kor):
#     print("enumerate(kor) =" ,i , k)
# for i , k in enumerate(eng):
#     print("enumerate(eng)=" , i, k)
# for k , e in zip(kor, eng):
#     print("zip(kor,eng)=",k,e)
# print('sum1=', sum1), print('sum2=', sum2)
# print('sum3=', sum3), print('sum4=', sum4)
#
# def calc_area(type, a, b, c=None):
#     if type == 1:
#         result = a* b
#         msg = "사각형"
#     elif type ==2:
#         result = a* b/2
#         msg = "삼각형"
#     elif type == 3:
#         result = (a + b) *c/2
#         msg = "평행사변형"
#     return result , msg
# def say():
#     print("넓이를 구해료")
#
# def write(result, msg):
#     print(msg," 넓이는 ", result, "m2입니다")

# say()
# ret = calc_area(type=1, a=5, b=5)
# area, msg = calc_area(2,5,5)
# area2, _ = calc_area(3,10,7,5)
#
# print(type(ret))
# print(type(area), type(msg))
# print(ret[0], ret[1])
# print(area, msg)
# print(area2, "평행사변형")

# import header_area as mod
# from header_area import write
# mod.say()
# area, msg = mod.calc_area(type=1,a =5, b=5)
# write(area, msg)
#
# a = [1.5,2,3,4,5]
# b = map(float, a)
# c = divmod(5,3)
#
# #파이썬 내장 함수
# abs(a) #a의 절댓갑 반환
# chr(a) #a 값을 문자 반환
# divmod(a,b) #나눈 몫과 나머지 반환
# enumerate(a) #객체 a의 인덱스와 원소 반환
# eval(a) #문자열 a를 반환
# input() #키보드로 값을 입력받는 함수
# int(a) #a를 정수형으로 반환
# isinstance(a) #객체 a가 클래스의 인스턴스 인지 검사
# len(a) #객체 a의 원소 개수 반환
# list(a) #객체 a를 리스트로 반환
# map(int,a) #객체 a의 원소들을 함수 int로 수행한 결과 반환
# print() #콘솔창에 결과 출력
# max(a) #객체 a에서 최대 원소값 반환
# min(a) #객체 a에서 최소 원소값 반환
# open() #파일 읽기위한 파일 객체 만들기
# ord(a) #문자 a의 아스키코드 값 반환
# pow(a,b) #a를 b 제곱한 결과 반환
# range(a,b,c) #c 간격 갖는 법위(a~b)의 객체 반환
# round(a)  #a를 반올림하여 반환
# str(a) #a를 문자열로 반환
# sum(a) #객체 a 원소갑들의 합 반환
# tuple(a) #객체 afmf 튜플로 변환
# type(a) #a의 자료형 반환
# zip(a,b,c,d) #여러 객체를 묶어주는 함수
#
# print("최댓값: ", max(a), "최솟갑: ", min(a))
# print("몫과 나머지: ", c)
# print("c의 자료형: ", type(c), type(c[0]), type(c[1]))
# print("2의 4제곱: ", pow(2,4))
# print("절댓갑: ", abs(-4))
#
# list1, list2 = [1,2,3], [4,5.0,6]
# a, b = np.array(list1), np.array(list2)
# c = a+ b
# d = a -b
# e = a * b
# f = a/ b
# g = a * 2
# h = b +2
# for i in [a,b,c,d,e,f,g,h]:
#     print("{0} 자료형".format(i) , i)
# list3 = list1 +list2
# print(list3)
# a = np.zeros((2,5) , np.int8)
# b = np.ones((3,1), np.uint8)
# c = np.empty((1,5), np.float16)
# d = np.full(5,15, np.float32)
#
# for i in [a,b,c]:
#     print(type(i), type(i[0]), type(i[0][0]))
# print(type(d), type(d[0]))
# print("c 형채:" ,c.shape, 'd 형태', d.shape)
# print(a),print(b)
# print(c), print(d)
#
# np.random.seed(10)
# a = np.random.rand(2,3) #균일분포 난수 2행 3열 행렬
# b = np.random.rand(3,2) #평균0, 표준편차 1인 정규분포 난수
# c = np.random.rand(6) #규일분포 난수 -1차원 행렬
# d = np.random.randint(1,100,6) #1~100 사이의 정수 난수 1차원 행렬
# c = np.reshape(c, (2,3)) #형태 변경 방법
# d = d.reshape(2, -1)
# for i in [a,b,c,d]:
#     print("형태:", i.shape , "\n", i)
# print('다차원 객체 1차원 변환 방법')
# print('a = ', a.flatten()) #다차원 ndarray 객체를 1차원 벡터로 변환
# print('b = ', np.ravel(b)) #다차원 모든 개개체를 1차원 벡터로 변환
# print('c = ', np.reshape(c, (-1, ))) #넘파이의 reshape() 함수 사용
# print('d = ', d.reshape(-1, )) #ndarray 객체 내장 reshape() 함수 사용