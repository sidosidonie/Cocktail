netif=lo
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export WANDB_NAME=opt-cot

export QUANT_BITS=4
export TOPK_RATIO=0.2
export RANDOMP_RATIO=0.1

export SHOW_DATA=0

ARGS="--model-name models/opt-1.3b-new \
--tokenizer-name models/opt-1.3b-new \
--project-name cocktail-sgd \
--model-type opt \
--optimizer adam \
--seed 42 \
--load-pretrained-model true \
--task-name cot \
--checkpoint-path ./model_ckpts/$WANDB_NAME \
--num-layers 12 --embedding-dim 2048 \
--total-steps 200 --warmup-steps 10 --train-warmup-steps 0 \
--checkpoint-steps 100 \
--lr 1e-4 --seq-length 2048 --batch-size 16 --micro-batch-size 1 --gradient-accumulate-step 1 \
--dist-url tcp://127.0.0.1:7033 \
--world-size 1 --pipeline-group-size 1 --data-group-size 1 \
--job-id 0 --net-interface ${netif} \
--fp16 \
--dp-backend gloo \
--dp-mode cocktail_sgd \
--pp-mode gpipe --profiling no-profiling"

python dist_lm_train.py $(echo ${ARGS}) --cuda-id 0 --rank 0 
