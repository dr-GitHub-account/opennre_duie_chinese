import torch
import torch.nn as nn

from .base_encoder import BaseEncoder
from transformers import BertModel, BertTokenizer

import logging
import numpy as np


class BERTEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768
        self.mask_entity = mask_entity
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        
        # logging.info("*****self.mask_entity: {}*****".format(self.mask_entity))
        

    def forward(self, token, att_mask):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        # self.bert包含embeddings、encoder、pooler
        # self.bert的第一个输出是sequence_output(即当前批每一个句子中每一个字符对应的隐变量，维度(batch_size, max_len))，第二个输出是pooled_output(即当前批每一个句子中第一个字符[CLS]对应的隐变量，维度(batch_size, 1))
        # 这里取的是第二个输出
        _, x = self.bert(token, attention_mask=att_mask)
        return x

    # item = self.data[index]
    # # item: {'token': ['最', '新', '电', '影', '《', '夺', '路', '而', '逃', '》', '张', '一', '山', '角', '色', '大', '曝', '光'], 'h': {'name': '夺路而逃', 'pos': [5, 8]}, 't': {'name': '张一山', 'pos': [10, 12]}, 'relation': '主演'}
    # seq = list(self.tokenizer(item, **self.kwargs))
    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        # 默认不进入下面的if
        if 'text' in item:
            sentence = item['text']
            is_token = False
        # 默认进入下面的else
        else:
            sentence = item['token']
            is_token = True
        # 头部实体和尾部实体的位置
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        if not is_token:
            pos_min = pos_head
            pos_max = pos_tail
            if pos_head[0] > pos_tail[0]:
                pos_min = pos_tail
                pos_max = pos_head
                rev = True
            else:
                rev = False
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
            if self.mask_entity:
                ent0 = ['[unused4]']
                ent1 = ['[unused5]']
                if rev:
                    ent0 = ['[unused5]']
                    ent1 = ['[unused4]']
            pos_head = [len(sent0), len(sent0) + len(ent0)]
            pos_tail = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
            if rev:
                pos_tail = [len(sent0), len(sent0) + len(ent0)]
                pos_head = [
                    len(sent0) + len(ent0) + len(sent1),
                    len(sent0) + len(ent0) + len(sent1) + len(ent1)
                ]
            tokens = sent0 + ent0 + sent1 + ent1 + sent2
        # 默认进入下面的else
        else:
            # tokens = sentence = ['最', '新', '电', '影', '《', '夺', '路', '而', '逃', '》', '张', '一', '山', '角', '色', '大', '曝', '光']
            tokens = sentence

        # Token -> index
        re_tokens = ['[CLS]']
        # 初始化当前token在tokens中所处的位置
        cur_pos = 0
        # 遍历tokens中所有的token
        # self.mask_entity默认为False
        for token in tokens:
            token = token.lower()
            # 默认进入下面的if，当前token是头部实体起始的token，'[unused0]'加到当前token的前面
            if cur_pos == pos_head[0] and not self.mask_entity:
                re_tokens.append('[unused0]')
            # 默认进入下面的if，当前token是尾部实体起始的token，'[unused1]'加到当前token的前面
            if cur_pos == pos_tail[0] and not self.mask_entity:
                re_tokens.append('[unused1]')
            # 当前token经过BertTokenizer.tokenize后的结果加入re_tokens
            re_tokens += self.tokenizer.tokenize(token)
            # 默认进入下面的if，当前token是头部实体倒数第二个token，'[unused2]'加到当前token的后面
            if cur_pos == pos_head[1] - 1 and not self.mask_entity:
                re_tokens.append('[unused2]')
            # 默认进入下面的if，当前token是尾部实体倒数第二个token，'[unused3]'加到当前token的后面
            if cur_pos == pos_tail[1] - 1 and not self.mask_entity:
                re_tokens.append('[unused3]')
            # 当前token在tokens中所处的位置加一
            cur_pos += 1
        # 句末加入'[SEP]'
        re_tokens.append('[SEP]')

        # re_tokens转化为id后的结果赋值给indexed_tokens
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        # indexed_tokens的长度，赋值给avai_len
        avai_len = len(indexed_tokens)

        # Padding到最大长度
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        # 返回indexed_tokens, att_mask
        return indexed_tokens, att_mask


class BERTEntityEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768 * 2
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.flag = 0

    def forward(self, token, att_mask, pos1, pos2):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
        Return:
            (B, 2H), representations for sentences
        """
        # self.bert包含embeddings、encoder、pooler
        # self.bert的第一个输出是sequence_output(即当前批每一个句子中每一个字符对应的隐变量，维度(batch_size, max_len, hidden_size))，第二个输出是pooled_output(即当前批每一个句子中第一个字符[CLS]对应的隐变量，维度(batch_size, 1, hidden_size))
        # 这里取的是第一个输出
        hidden, _ = self.bert(token, attention_mask=att_mask)
        # Get entity start hidden state
        onehot = torch.zeros(hidden.size()[:2]).float()  # (B, L)，即(batch_size, max_len)
        if torch.cuda.is_available():
            onehot = onehot.cuda()
        # 得到维度为(batch_size, max_len)的onehot_head和onehot_tail
        # onehot_head和onehot_tail相等，都是只有pos1、pos2处值为1，其他位置值为0的独热向量*****?????*****，维度(batch_size, max_len)
        onehot_head = onehot.scatter_(1, pos1, 1)
        onehot_tail = onehot.scatter_(1, pos2, 1)
        # onehot_head.unsqueeze(2)和onehot_tail.unsqueeze(2)维度都是(batch_size, max_len, 1)，第三维度pos1、pos2处值为1，其他位置为0
        # hidden维度(batch_size, max_len, hidden_size)，与上行两对象相乘后维度不变，但在200个位置的隐变量中，只保留pos1、pos2处的隐变量
        # .sum(1)操作在max_len所在的维度进行相加，得到维度(batch_size, hidden_size)，head_hidden和tail_hidden依旧是一样的
        head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden).sum(1)  # (B, H)
        # 将head_hidden和tail_hidden这两个一样的，维度为(batch_size, hidden_size)的向量进行拼接，得到维度为(batch_size, 2 * hidden_size)的向量
        x = torch.cat([head_hidden, tail_hidden], 1)  # (B, 2H)
        
        if self.flag == 0:
            logging.info("onehot_head: {}\n{}".format(onehot_head, np.shape(onehot_head)))
            logging.info("onehot_tail: {}\n{}".format(onehot_tail, np.shape(onehot_tail)))
            
            logging.info("onehot_head.unsqueeze(2): {}\n{}".format(onehot_head.unsqueeze(2), np.shape(onehot_head.unsqueeze(2))))
            logging.info("onehot_tail.unsqueeze(2): {}\n{}".format(onehot_tail.unsqueeze(2), np.shape(onehot_tail.unsqueeze(2))))

            logging.info("hidden: {}\n{}".format(hidden, np.shape(hidden)))
            
            logging.info("onehot_head.unsqueeze(2) * hidden: {}\n{}".format(onehot_head.unsqueeze(2) * hidden, np.shape(onehot_head.unsqueeze(2) * hidden)))
            logging.info("onehot_tail.unsqueeze(2) * hidden: {}\n{}".format(onehot_tail.unsqueeze(2) * hidden, np.shape(onehot_tail.unsqueeze(2) * hidden)))
            
            logging.info("head_hidden: {}\n{}".format(head_hidden, np.shape(head_hidden)))
            logging.info("tail_hidden: {}\n{}".format(tail_hidden, np.shape(tail_hidden)))
            
            logging.info("x: {}\n{}".format(x, np.shape(x)))
            self.flag = 1
        
        # (batch_size, 2 * hidden_size)的向量经过隐藏层处理，得到的维度仍然为(batch_size, 2 * hidden_size)
        x = self.linear(x)
        return x

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        # 默认进入下面的else
        else:
            sentence = item['token']
            is_token = True
        # 头部实体与尾部实体的位置
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        # 默认不进入下面的if
        if not is_token:
            pos_min = pos_head
            pos_max = pos_tail
            if pos_head[0] > pos_tail[0]:
                pos_min = pos_tail
                pos_max = pos_head
                rev = True
            else:
                rev = False
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
            pos_head = [len(sent0), len(sent0) + len(ent0)]
            pos_tail = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
            if rev:
                pos_tail = [len(sent0), len(sent0) + len(ent0)]
                pos_head = [
                    len(sent0) + len(ent0) + len(sent1),
                    len(sent0) + len(ent0) + len(sent1) + len(ent1)
                ]
            tokens = sent0 + ent0 + sent1 + ent1 + sent2
        # 默认进入下面的else，
        # tokens = sentence = ['《', '奇', '趣', '海', '洋', '动', '物', '》', '是', '2', '0', '1', '1', '年', '少', '年', '儿', '童', '出', '版', '社', '出', '版', '的', '图', '书', '，', '作', '者', '是', '雷', '宗', '友']
        else:
            tokens = sentence

        # Token -> index
        re_tokens = ['[CLS]']
        cur_pos = 0
        # 在加入了特殊字符后，头部实体起始位置pos1，尾部实体起始位置pos2
        # pos1: position of the head entity starter
        # pos2: position of the tail entity starter
        pos1 = 0
        pos2 = 0
        # 遍历tokens中的每个token
        for token in tokens:
            token = token.lower()
            # 默认进入下面的if，当前token是头部实体起始的token，'[unused0]'加到当前token的前面，当前re_tokens长度赋值给pos1
            if cur_pos == pos_head[0]:
                pos1 = len(re_tokens)
                re_tokens.append('[unused0]')
            # 默认进入下面的if，当前token是尾部实体起始的token，'[unused1]'加到当前token的前面，，当前re_tokens长度赋值给pos2
            if cur_pos == pos_tail[0]:
                pos2 = len(re_tokens)
                re_tokens.append('[unused1]')
            # 当前token经过tokenize的结果加到re_tokens中
            re_tokens += self.tokenizer.tokenize(token)
            # 默认进入下面的if，当前token是头部实体倒数第二个token，'[unused2]'加到当前token的后面
            if cur_pos == pos_head[1] - 1:
                re_tokens.append('[unused2]')
            # 默认进入下面的if，当前token是尾部实体倒数第二个token，'[unused3]'加到当前token的后面
            if cur_pos == pos_tail[1] - 1:
                re_tokens.append('[unused3]')
            # 更新当前token在tokens中所处位置
            cur_pos += 1
        # 末尾加上'[SEP]'
        re_tokens.append('[SEP]')
        # 在加入了特殊字符后，头部实体起始位置pos1，尾部实体起始位置pos2
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)
        
        # re_tokens转化为id后的结果赋值给indexed_tokens
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        # indexed_tokens的长度，赋值给avai_len
        avai_len = len(indexed_tokens)

        # 在加入了特殊字符后，头部实体起始位置pos1，尾部实体起始位置pos2
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

        # Padding到最大长度
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(
            0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2
