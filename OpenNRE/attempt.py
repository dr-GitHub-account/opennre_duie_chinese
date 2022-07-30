# s = '{"token": ["这", "样", "李", "世", "民", "最", "为", "心", "腹", "之", "人", "只", "有", "长", "孙", "无", "忌", "仍", "在", "府", "中", "。"], "h": {"name": "长孙无忌", "pos": [13, 16]}, "t": {"name": "李世民", "pos": [2, 4]}, "relation": "兄弟姐妹"}'
# eval_s = eval(s)
# print(eval_s)
# print(type(eval_s))

import torch

# onehot = torch.zeros(1, 38).float()

# pos1 = torch.tensor([[2]])
# pos2 = torch.tensor([[33]])

# onehot_head = onehot.scatter_(1, pos1, 1)
# onehot_tail = onehot.scatter_(1, pos2, 1)

# print("onehot_head:", onehot_head)
# print("onehot_tail:", onehot_tail)
# print(onehot_head == onehot_tail)

before_sum = torch.tensor([[[ 0.0000,  0.0000, -0.0000, -0.0000, -0.0000,  0.0000],
                            [ 0.1371,  0.2093, -0.0006, 0.1384, -0.4327, -0.1794],
                            [ 0.0000, 0.0000,  0.0000, 0.0000,  0.0000, 0.0000],
                            [ 0.1111, 0.1111,  0.1111, 0.1111,  0.1111, 0.1111],
                            [ 0.0000, 0.0000,  0.0000, 0.0000,  0.0000, 0.0000]]])

after_sum = before_sum.sum(1)

print("after_sum: {}".format(after_sum))