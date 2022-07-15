import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import math
import os

mode = 'val'

file_path = "/home/user/xiongdengrui/opennre_chinese/OpenNRE/benchmark/people-relation/people-relation_{}.txt".format(mode)
image_save_prefix = "/home/user/xiongdengrui/opennre_chinese/OpenNRE/benchmark/people-relation/"

rel2id = json.load(open('/home/user/xiongdengrui/opennre_chinese/OpenNRE/benchmark/people-relation/people-relation_rel2id.json'))
# print("rel2id:", rel2id, type(rel2id))
# # rel2id: {'父母': 0, '夫妻': 1, '师生': 2, '兄弟姐妹': 3, '合作': 4, '情侣': 5, '祖孙': 6, '好友': 7, '亲戚': 8, '同门': 9, '上下级': 10, 'unknown': 11} <class 'dict'>

count_classes = np.zeros(len(rel2id))
# print(count_classes)
# # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

# Load the file
f = open(file_path)
# data_list是一个列表，其中每一个元素都是一个字典，代表数据集txt文件中的一行
data_list = []
for line in f.readlines():
    # rstrip()返回字符串的副本，末尾空格移除
    line = line.rstrip()
    if len(line) > 0:
        # 将字符串str当成有效的表达式来求值并返回计算结果，本情况下即返回dict
        data_list.append(eval(line))
f.close()
# print("data_list[0]:", data_list[0])
# # data_list[0]: {'token': ['马', '师', '曾', '-', '婚', '姻', '与', '红', '线', '女', '的', '婚', '姻', '红', '线', '女', '与', '马', '师', '曾', '马', '师', '曾', '是', '女', '姐', '的', '首', '任', '丈', '夫', '，', '由', '于', '马', '师', '曾', '经', '常'], 'h': {'name': '红线女', 'pos': [7, 9]}, 't': {'name': '马师曾', 'pos': [0, 2]}, 'relation': '夫妻'}

for data in data_list:
    count_classes[rel2id[data['relation']]] += 1

# print(count_classes)
# # [139. 190.  52.  61.  66.  45.  17.  14.   8.  20.  11. 377.]
    
plt.rcParams['figure.figsize'] = (8.0, 5.0) 
plt.rcParams['savefig.dpi'] = 1000 # resolution of the saved image 
plt.rcParams['figure.dpi'] = 100 # resolution of the shown image
plt.figure()

class_names = list(rel2id.values())

plt.bar(class_names, count_classes, facecolor="blue", edgecolor="black", alpha=0.7)

# plt.xticks((0, 0.125, 0.25, 0.5, 1, 2, 4, 8))
# plt.xscale('log', 2)
plt.xlabel("Class ID")
plt.ylabel("Number")
# 显示图标题
plt.title("Class Distribution")

# 把x轴的刻度间隔设置为1，并存在变量里
x_major_locator=MultipleLocator(1)
# ax为两条坐标轴的实例
ax=plt.gca()
# 把x轴的主刻度设置为1的倍数
ax.xaxis.set_major_locator(x_major_locator)

plt.savefig(image_save_prefix + "{}_class_distribution.jpg".format(mode), dpi = 1000)

# *******************************************************

# ratio = []
# num_illegal = 0
# num_round_to_0 = 0

# # for anno in load_json["annotations"]:
# #     ratio.append(round(anno["bbox"][2]/anno["bbox"][3],2))
    
# for anno in load_json["annotations"]:
#     if (anno["bbox"][3] <= 0 or anno["bbox"][2] <= 0):
#         num_illegal = num_illegal + 1
#         # print(anno["id"])
#         continue
#     else:
#         if(round(anno["bbox"][2]/anno["bbox"][3],2) != 0):
#             ratio.append(round(anno["bbox"][2]/anno["bbox"][3],2))
#         else:
#             num_round_to_0 = num_round_to_0 + 1
                
# # print("total illegal instance number: ", num_illegal)
# # print("total round-to-0 instance number: ", num_round_to_0)
    
# # list from minimum to maximum
# ratio.sort()

# # print("min in ratio: ", ratio[0])
# # print("max in ratio: ", ratio[-1])

# count_ratio_section = [0, 0, 0, 0, 0, 0, 0, 0]
# for r in ratio:
#     log2_r = math.log(r, 2)
#     # print(r, math.log(r, 2), math.ceil(math.log(r, 2)))
#     for i in range(1, 7):
#         if log2_r >= (i-4) and log2_r < (i-3):
#             count_ratio_section[i] += 1
#     if log2_r < -3:
#         count_ratio_section[0] += 1
#     if log2_r >= 3:
#         count_ratio_section[7] += 1

# # print(count_ratio_section)

# ratio_range = ['' for i in range(8)]
# for i in range(1, 4):
#     ratio_range[i] = str(2**(i - 1) * 1/8) + '-' + str(2**i * 1/8)
# for i in range(4, 7):
#     ratio_range[i] = str(float(2**(i - 4))) + '-' + str(float(2**(i-3)))
# ratio_range[0] = '0-0.125'
# ratio_range[7] = '8.0 and more'

# # print(ratio_range)

# plt.bar(ratio_range, count_ratio_section, facecolor="blue", edgecolor="black", alpha=0.7)

# # plt.xticks((0, 0.125, 0.25, 0.5, 1, 2, 4, 8))
# # plt.xscale('log', 2)
# plt.xlabel("ratio")
# plt.ylabel("Number")
# # 显示图标题
# plt.title("Ratio Distribution")

# plt.savefig(image_save_prefix + "analysis_ratio.jpg", dpi = 1000)