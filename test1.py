from plotDecBoundaries_ovr import plotDecBoundaries_ovr
from plot import plot
import csv
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


def disteclud(vec1, vec2):  # 定义一个欧式距离的函数
    return np.sqrt(np.sum(np.power(vec1 - vec2, 2)))


training = np.zeros((89, 2))
label_train = np.zeros(89, dtype=np.int)
mean1 = np.zeros((2, 2))
mean2 = np.zeros((2, 2))
mean3 = np.zeros((2, 2))
test = np.zeros((89, 2))
label_test = np.zeros(100, dtype=np.int)
flag = np.zeros((89, 3))
sum_x1 = 0
sum_y1 = 0
sum_x23 = 0
sum_y23 = 0
sum_x2 = 0
sum_y2 = 0
sum_x13 = 0
sum_y13 = 0
sum_x3 = 0
sum_y3 = 0
sum_x12 = 0
sum_y12 = 0

i = 0
class1_cnt = 0
class23_cnt = 0
class2_cnt = 0
class13_cnt = 0
class3_cnt = 0
class12_cnt = 0

with open('wine_train.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        training[i, 0] = float(row[0])
        training[i, 1] = float(row[1])
        label_train[i] = float(row[13])
        i += 1

        if int(row[13]) == 1:
            sum_x1 = sum_x1 + float(row[0])
            sum_y1 = sum_y1 + float(row[1])
            class1_cnt += 1
        else:
            sum_x23 = sum_x23 + float(row[0])
            sum_y23 = sum_y23 + float(row[1])
            class23_cnt += 1

        if int(row[13]) == 2:
            sum_x2 = sum_x2 + float(row[0])
            sum_y2 = sum_y2 + float(row[1])
            class2_cnt += 1
        else:
            sum_x13 = sum_x13 + float(row[0])
            sum_y13 = sum_y13 + float(row[1])
            class13_cnt += 1

        if int(row[13]) == 3:
            sum_x3 = sum_x3 + float(row[0])
            sum_y3 = sum_y3 + float(row[1])
            class3_cnt += 1
        else:
            sum_x12 = sum_x12 + float(row[0])
            sum_y12 = sum_y12 + float(row[1])
            class12_cnt += 1

# first pair
mean1[0, 0] = sum_x1 / class1_cnt
mean1[0, 1] = sum_y1 / class1_cnt
mean1[1, 0] = sum_x23 / class23_cnt
mean1[1, 1] = sum_y23 / class23_cnt
plotDecBoundaries_ovr(training, label_train, mean1)
index = 0
for axis_train in training:
    d1 = disteclud(axis_train, mean1[0])
    d2 = disteclud(axis_train, mean1[1])
    if d1 > d2:
        flag[index, 0] = 0
    if d1 < d2:
        flag[index, 0] = 1
    index += 1

# second pair
mean2[0, 0] = sum_x2 / class2_cnt
mean2[0, 1] = sum_y2 / class2_cnt
mean2[1, 0] = sum_x13 / class13_cnt
mean2[1, 1] = sum_y13 / class13_cnt
plotDecBoundaries_ovr(training, label_train, mean2)
index = 0
for axis_train in training:
    d1 = disteclud(axis_train, mean2[0])
    d2 = disteclud(axis_train, mean2[1])
    if d1 > d2:
        flag[index, 1] = 0
    if d1 < d2:
        flag[index, 1] = 1
    index += 1

# third pair
mean3[0, 0] = sum_x3 / class3_cnt
mean3[0, 1] = sum_y3 / class3_cnt
mean3[1, 0] = sum_x12 / class12_cnt
mean3[1, 1] = sum_y12 / class12_cnt
plotDecBoundaries_ovr(training, label_train, mean3)
plot(training, label_train, mean1, mean2, mean3)

index = 0
for axis_train in training:
    d1 = disteclud(axis_train, mean3[0])
    d2 = disteclud(axis_train, mean3[1])
    if d1 > d2:
        flag[index, 2] = 0
    if d1 < d2:
        flag[index, 2] = 1
    index += 1

# calculate the error for train
right = 0
for i in range(0, 89):
    if label_train[i] == 1:
        if flag[i, 0] == 1 and flag[i, 1] == 0 and flag[i, 2] == 0:
            right += 1
    if label_train[i] == 2:
        if flag[i, 0] == 0 and flag[i, 1] == 1 and flag[i, 2] == 0:
            right += 1
    if label_train[i] == 3:
        if flag[i, 0] == 0 and flag[i, 1] == 0 and flag[i, 2] == 1:
            right += 1
error = i + 1 - right
print("error rate for train:", error / (i + 1))

index = 0
# error rate for test
with open('wine_test.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        test[index, 0] = float(row[0])
        test[index, 1] = float(row[1])
        label_test[index] = float(row[13])
        index += 1

index = 0
for axis_test in test:
    d1 = disteclud(axis_test, mean1[0])
    d2 = disteclud(axis_test, mean1[1])
    if d1 > d2:
        flag[index, 0] = 0
    if d1 < d2:
        flag[index, 0] = 1
    index += 1

index = 0
for axis_test in test:
    d1 = disteclud(axis_test, mean2[0])
    d2 = disteclud(axis_test, mean2[1])
    if d1 > d2:
        flag[index, 1] = 0
    if d1 < d2:
        flag[index, 1] = 1
    index += 1

index = 0
for axis_test in test:
    d1 = disteclud(axis_test, mean3[0])
    d2 = disteclud(axis_test, mean3[1])
    if d1 > d2:
        flag[index, 2] = 0
    if d1 < d2:
        flag[index, 2] = 1
    index += 1

# calculate the error for train
right = 0
for i in range(0, 89):
    if label_test[i] == 1:
        if flag[i, 0] == 1 and flag[i, 1] == 0 and flag[i, 2] == 0:
            right += 1
    if label_test[i] == 2:
        if flag[i, 0] == 0 and flag[i, 1] == 1 and flag[i, 2] == 0:
            right += 1
    if label_test[i] == 3:
        if flag[i, 0] == 0 and flag[i, 1] == 0 and flag[i, 2] == 1:
            right += 1
error = i + 1 - right
print("error rate for test:", error / (i + 1))
