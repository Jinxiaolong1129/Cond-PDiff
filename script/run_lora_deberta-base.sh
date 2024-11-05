#!/bin/bash
declare -A metrics
metrics['cola']=matthews_correlation
metrics['sst2']=accuracy
metrics['mrpc']=combined_score
metrics['qqp']=combined_score
metrics['stsb']=combined_score
metrics['rte']=accuracy
metrics['qnli']=accuracy
metrics['mnli']=accuracy

declare -A t2epoch
t2epoch['cola']=20
t2epoch['sst2']=20
t2epoch['mrpc']=20
t2epoch['qqp']=20
t2epoch['stsb']=20
t2epoch['rte']=20
t2epoch['qnli']=20
t2epoch['mnli']=20


seed=42

export DEBUG=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="src/:$PYTHONPATH"

# dataset: 'cola' 'rte' 'mrpc' 'sst2' 'qnli' 'stsb' 'cola'
# model: 'microsoft/deberta-base'

lora_r=1
lora_alpha=8
per_device_eval_batch_size=16
save_last_n_step=200


for TASK_NAME in 'sst2' 'qnli' 'stsb' 'cola' 'rte' 'mrpc'
do
    for model in 'deberta-base'
    do
        export WANDB_PROJECT=para_diff_lora_${model}
        export WANDB_NAME="${model}.${TASK_NAME}.lora_r_${lora_r}.lora_alpha_${lora_alpha}.perbatch_${per_device_eval_batch_size}.epoch_${t2epoch[${TASK_NAME}]}.seed_${seed}"

        metric=${metrics[${TASK_NAME}]}

        exp_name=${TASK_NAME}.lora_r_${lora_r}.lora_alpha_${lora_alpha}.perbatch_${per_device_eval_batch_size}.epoch_${t2epoch[${TASK_NAME}]}.seed_${seed}
        
        SAVE=dataset/${model}/${TASK_NAME}/${exp_name}/

        echo "TASK_NAME: $TASK_NAME, Model: $model, LoRA r: $lora_r, LoRA alpha: $lora_alpha, per_device_eval_batch_size: $per_device_eval_batch_size"
        echo "Exp Name: $exp_name, Save: $SAVE"
        echo "Save to run_log/train_log_${model}.${TASK_NAME}.lora_r_${lora_r}.lora_alpha_${lora_alpha}_${TASK_NAME}.txt"

        rm -rf ${SAVE}; 
        mkdir -p ${SAVE}
        mkdir -p run_log

        nohup python examples/pytorch/text-classification/run_glue_last_step_save.py \
            --lora_r ${lora_r} \
            --lora_alpha ${lora_alpha}  \
            --apply_lora True \
            --lora_dropout 0.1 \
            \
            --num_train_epochs ${t2epoch[${TASK_NAME}]} \
            --save_last_n_step 200 \
            --model_name_or_path 'microsoft/deberta-base' \
            --task_name $TASK_NAME \
            \
            --do_train \
            --do_eval \
            --do_predict \
            --fp16 \
            \
            --max_seq_length 128 \
            --per_device_train_batch_size ${per_device_eval_batch_size} \
            --per_device_eval_batch_size 256 \
            \
            --learning_rate 0.0001 \
            --warmup_ratio 0.1 \
            --weight_decay 0.1 \
            \
            --overwrite_output_dir \
            --metric_for_best_model ${metric} \
            \
            --evaluation_strategy epoch \
            --save_strategy no \
            \
            --output_dir ${SAVE} \
            --logging_dir "${SAVE}/log" \
            --seed ${seed} > run_log/train_${model}.${TASK_NAME}.lora_r_${lora_r}.lora_alpha_${lora_alpha}_${TASK_NAME}.txt 2>&1 &
    done
done

