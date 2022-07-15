import sys, json
import torch
import os
import numpy as np
import opennre
# from opennre import encoder, model, framework
import argparse
import logging

logger = logging.getLogger()
def init_logger(log_file=None, log_file_level=logging.NOTSET):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')

    # if no name is specified for logging.getLogger(), return the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        # file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger

parser = argparse.ArgumentParser()
parser.add_argument('--mask_entity', action='store_true', help='Mask entity mentions')

# customize
parser.add_argument("--output_time", type=str, default='default')
parser.add_argument("--repeat", type=str, default='default')
parser.add_argument("--test_ckpt", type=str, default='default')

args = parser.parse_args()

# 指定日志文件保存路径并初始化log
logfile = "work_dirs/duie_train_cls_micro_{}_{}.log".format(args.output_time, args.repeat)
init_logger(log_file=logfile)

# Some basic settings
root_path = '.'
sys.path.append(root_path)
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
if not os.path.exists('ckpt/train_{}_{}'.format(args.output_time, args.repeat)):
    os.mkdir('ckpt/train_{}_{}'.format(args.output_time, args.repeat))
ckpt = 'ckpt/train_{}_{}/people_chinese_bert_softmax.pth.tar'.format(args.output_time, args.repeat)
# ckpt = 'ckpt/people_chinese_bert_softmax.pth.tar'

# Check data
rel2id = json.load(open(os.path.join(root_path, 'benchmark/duie/duie_re_1_spo_noWork_rel2id.json')))

# Define the sentence encoder
sentence_encoder = opennre.encoder.BERTEncoder(
    max_length=200, 
    # pretrain_path=os.path.join(root_path, 'pretrain/chinese_wwm_pytorch'),
    pretrain_path=os.path.join(root_path, 'pretrain/bert-base-chinese'),
    mask_entity=args.mask_entity
)

# # Define the sentence encoder
# sentence_encoder = opennre.encoder.BERTEntityEncoder(
#     max_length=200, 
#     # pretrain_path=os.path.join(root_path, 'pretrain/chinese_wwm_pytorch'),
#     pretrain_path=os.path.join(root_path, 'pretrain/chinese-bert-wwm-ext')
# )

# Define the model
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

# Define the whole training framework
framework = opennre.framework.SentenceRE(
    train_path=os.path.join(root_path, 'benchmark/duie/duie_re_train_1_spo_noWork.txt'),
    val_path=os.path.join(root_path, 'benchmark/duie/duie_re_dev_1_spo_noWork.txt'),
    test_path=os.path.join(root_path, 'benchmark/duie/duie_re_dev_1_spo_noWork.txt'),
    model=model,
    ckpt=ckpt,
    batch_size=32, # Modify the batch size w.r.t. your device
    max_epoch=8,
    lr=2e-5,
    opt='adamw'
)

# Train the model
framework.train_model(metric='micro_f1')

# Test the model
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)

# Print the result
print('Accuracy on test set: {}'.format(result['micro_f1']))
