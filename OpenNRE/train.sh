time=$(date "+%Y%m%d%H%M%S")
output_time=${time}

# CUDA_VISIBLE_DEVICES=1 python example/train_supervised_bert.py \
#     --pretrain_path /home/user/xiongdengrui/opennre/downloaded_pretrain_benchmark/pretrain/bert-base-uncased \
#     --dataset wiki80 \
#     --output_time ${output_time} \
#     --do_train

# ************************train************************
# for repeat in 1;
# do
# CUDA_VISIBLE_DEVICES=1 python example/train_people_chinese_bert_softmax.py \
#     --pretrain_path /home/user/xiongdengrui/opennre_chinese/OpenNRE/pretrain/chinese-bert-wwm-ext \
#     --dataset semeval \
#     --output_time ${output_time} \
#     --do_test \
#     --test_ckpt /home/user/xiongdengrui/opennre/OpenNRE/work_dirs/20220519225232_1/best.pth.tar
# done

for repeat in bbc_analyze_getitem;
do
CUDA_VISIBLE_DEVICES=0 python example/train_duie_chinese_bert_softmax.py \
    --output_time ${output_time} \
    --repeat ${repeat}
done
# ************************train************************

# ************************train************************
# CUDA_VISIBLE_DEVICES=1 python example/train_people_chinese_bert_softmax.py \
#     --pretrain_path /home/user/xiongdengrui/opennre_chinese/OpenNRE/pretrain/chinese-bert-wwm-ext \
#     --dataset semeval \
#     --output_time ${output_time} \
#     --do_test \
#     --test_ckpt /home/user/xiongdengrui/opennre/OpenNRE/work_dirs/20220519225232_1/best.pth.tar
# ************************train************************