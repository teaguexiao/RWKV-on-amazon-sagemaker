#!/bin/bash

chmod +x ./s5cmd
./s5cmd sync s3://$MODEL_S3_BUCKET/RWKV/1B5/* /tmp/rwkv/


cd RWKV-LM-LoRA/RWKV-v4neo/

#update-alternatives --display cuda
#update-alternatives --auto cuda
#to resolve issue https://github.com/BlinkDL/RWKV-LM/issues/129
#export CUDA_HOME=/usr/local/cuda
#export PATH=/usr/local/cuda/bin:$PATH
#export CPATH=/usr/local/cuda/include:$CPATH
#export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

RWKV_CUDA_ON=1 python3 train.py  \
    --load_model "/tmp/rwkv/RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth" \
    --proj_dir "./" \
    --data_file "./test_text_document" \
    --data_type "binidx" \
    --vocab_size 50277  \
    --ctx_len 1024 \
    --epoch_save 1000 \
    --epoch_count 10 \
    --n_layer 24 \
    --n_embd 2048 \
    --epoch_steps 10 \
    --epoch_begin 0 \
    --micro_bsz 1  \
    --pre_ffn 0 \
    --head_qk 0 \
    --lr_init 1e-4 \
    --lr_final 1e-5 \
    --warmup_steps 0 \
    --beta1 0.9 \
    --beta2 0.999 \
    --adam_eps 1e-8 \
    --accelerator gpu \
    --devices 1 \
    --precision bf16 \
    --strategy deepspeed_stage_2 \
    --grad_cp 1   \
    --lora_parts=att,ffn,time,ln \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.01


if [ $? -eq 1 ]; then
    echo "Training script error, please check CloudWatch logs"
    exit 1
fi

./s5cmd sync /tmp/llama_out s3://$MODEL_S3_BUCKET/RWKV/output/$(date +%Y-%m-%d-%H-%M-%S)/
