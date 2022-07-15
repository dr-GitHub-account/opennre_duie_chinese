import torch
from torch import nn, optim
import json
from .data_loader import SentenceRELoader
from .utils import AverageMeter
from tqdm import tqdm
import os
import logging

import numpy as np
from .lr_scheduler import get_linear_schedule_with_warmup

class SentenceRE(nn.Module):

    def __init__(self, 
                 model,
                 train_path, 
                 val_path, 
                 test_path,
                 ckpt, 
                 batch_size=32, 
                 max_epoch=100, 
                 lr=0.1, 
                 weight_decay=1e-5, 
                 opt='sgd'):
    
        # 打印日志中的参数
        logging.info('Hyperparameters Info:')
        logging.info('    batch size: {}'.format(batch_size))
        logging.info('    max epoch: {}'.format(max_epoch))
        logging.info('    lr: {}'.format(lr))
        logging.info('    weight_decay: {}'.format(weight_decay))
        logging.info('    optimizer: {}'.format(opt))
        
        logging.info('*****FRAMEWORK INSTANTIATED*****')
    
        super().__init__()
        self.max_epoch = max_epoch
        # Load data
        if train_path != None:
            self.train_loader = SentenceRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True)

        if val_path != None:
            self.val_loader = SentenceRELoader(
                val_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False)
        
        if test_path != None:
            self.test_loader = SentenceRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False
            )
        # Model
        self.model = model
        self.parallel_model = nn.DataParallel(self.model)
        # Criterion
        self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
        params = self.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw': # Optimizer for BERT
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self, warmup=True, metric='micro_f1'):
        best_metric = 0
        global_step = 0
        
        t_total = len(self.train_loader) * self.max_epoch
        
        # 学习率调度，get_linear_schedule_with_warmup()表示先warmup再线性下降
        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=150,
                                                    num_training_steps=t_total)
        
        # 遍历每一轮
        for epoch in range(self.max_epoch):
            self.train()
            print("=== Epoch %d train ===" % epoch)
            logging.info("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            t = tqdm(self.train_loader)
            # 遍历每个batch
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                # 用id表示的该批量中所有句子的关系标签，torch.Size([batch_size])
                label = data[0]
                # 两个tensor，
                # 第一个是该批量中所有的句子，torch.Size([batch_size, max_length])
                # 第二个是该批量中所有句子对应的有效掩码，torch.Size([batch_size, max_length])
                args = data[1:]
                
                # if iter == 1:
                #     logging.info("*****for iter == 1: *****")
                #     logging.info("label: {}".format(label))
                #     logging.info("np.shape(label): {}".format(np.shape(label)))
                #     logging.info("args: {}".format(args))
                #     logging.info("np.shape(args[0]): {}".format(np.shape(args[0])))
                #     logging.info("np.shape(args[1]): {}".format(np.shape(args[1])))
                
                # self.parallel_model是model并行化以后的结果
                # logits是当前批量经编码(包含embedding、encoder、pooler)、dropout、全连接层得到的结果，torch.Size([batch_size, num_classes])
                logits = self.parallel_model(*args)
                # loss是当前批量的损失，是一个标量
                loss = self.criterion(logits, label)
                # score是当前批量每个预测结果的分数，pred是用id表示的当前批量的预测结果，维度都是torch.Size([batch_size])
                score, pred = logits.max(-1) # (B)
                # 当前批量的accuracy
                acc = float((pred == label).long().sum()) / label.size(0)
                
                # if iter == 1:
                #     logging.info("*****for iter == 1: *****")
                #     logging.info("logits: {}".format(logits))
                #     logging.info("np.shape(logits): {}".format(np.shape(logits)))
                #     logging.info("loss: {}".format(loss))
                #     logging.info("np.shape(loss): {}".format(np.shape(loss)))
                #     logging.info("score: {}".format(score))
                #     logging.info("np.shape(score): {}".format(np.shape(score)))
                #     logging.info("pred: {}".format(pred))
                #     logging.info("np.shape(pred): {}".format(np.shape(pred)))
                
                # Log
                # 将当前批量的loss和accuracy记入总的
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
                # Optimize
                
                # # 默认有warmup，进入下面的if
                # if warmup == True:
                #     warmup_step = 300
                #     if global_step < warmup_step:
                #         warmup_rate = float(global_step) / warmup_step
                #     else:
                #         warmup_rate = 1.0
                #     for param_group in self.optimizer.param_groups:
                #         param_group['lr'] = self.lr * warmup_rate
                
                loss.backward()
                self.optimizer.step()
                
                scheduler.step()
                
                self.optimizer.zero_grad()
                global_step += 1
                
            # Val 
            print("=== Epoch %d val ===" % epoch)
            logging.info("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader) 
            print("********result for epoch {}: {}".format(epoch, result))
            logging.info("********result for epoch {}: {}".format(epoch, result))
            print("********self.optimizer.state_dict()['param_groups'][0]['lr'] for epoch {}: {}".format(epoch, self.optimizer.state_dict()['param_groups'][0]['lr']))
            logging.info("********self.optimizer.state_dict()['param_groups'][0]['lr'] for epoch {}: {}".format(epoch, self.optimizer.state_dict()['param_groups'][0]['lr']))
            if result[metric] > best_metric:
                print("Best ckpt and saved.")
                logging.info("Best ckpt and saved.")
                folder_path = '/'.join(self.ckpt.split('/')[:-1])
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                best_metric = result[metric]
        print("Best %s on val set: %f" % (metric, best_metric))
        logging.info("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader):
        self.eval()
        avg_acc = AverageMeter()
        pred_result = []
        with torch.no_grad():
            t = tqdm(eval_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]        
                logits = self.parallel_model(*args)
                score, pred = logits.max(-1) # (B)
                # Save result
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())
                # Log
                acc = float((pred == label).long().sum()) / label.size(0)
                avg_acc.update(acc, pred.size(0))
                t.set_postfix(acc=avg_acc.avg)
        result = eval_loader.dataset.eval(pred_result)
        return result

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

