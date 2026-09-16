#!/bin/bash
# Bash script for SteelBERT Experiments


num_trial=1000
num_process=1


model_name="SteelBERT"
target_propertys=("Tensile_value") #Yield_value     Elongation_value    Tensile_value 
seeds=(501 401 301 201)
split_ratio=(0.8 0.9)


simple_layer_dict='{"simple_layer_list": [512, 512, 256, 256, 256], "normalization": "null_norm", "activation": "prelu", "dropout_prob": 0.0}'
seq_embed_layer_dict='{"seq_embed_con1d_list": [1, 1], "kernel_size": 2, "stride": 1, "padding": 1, "seq_embed_fc_list": [32, 16, 16], "normalization": "null_norm", "activation": "prelu", "dropout_prob": 0.0}'
common_embed_layer_dict='{"seq_embed_con2d_list": [1, 1], "kernel_size": 2, "stride": 1, "padding": 2, "seq_embed_2d_fc_list": [32, 32, 32, 16, 16, 16, 16], "normalization": "null_norm", "activation": "prelu", "dropout_prob": 0.0}'
concat_layer_dict='{"concat_layer_list": [64, 64, 64, 64, 32, 32, 32, 32, 8, 4, 1], "normalization": "null_norm", "activation": "prelu", "dropout_prob": 0.0}'

norm="null_norm"
mix_layer=13


train_batch_list='[4, 8, 10,16,32]'
epoch_list='[104,150,200]'
lr_list='[1e-4, 2e-4, 5e-5, 3e-5, 1e-5]'
step_list='[15, 20]'
gamma_ratio_list='[0.9,0.95]'

dropout_list='[0.0, 0.1, 0.2]' 


total_runs=$(( ${#target_propertys[@]} * ${#seeds[@]} * num_trial ))
GPUS=(5) 
MAX_JOBS=4

run_experiment() {
    local target_property=$1
    local seed=$2
    local gpu_id=$3
    local local_model=$4 

    echo "========================================"
    echo "Running Experiment:"
    echo "Target_property : $target_property"
    echo "Seed            : $seed"
    echo "GPU ID          : $gpu_id"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=$gpu_id python app_main.py \
        --num_trial "$num_trial" \
        --num_process "$num_process" \
        --model_name "$local_model" \
        --target_property "$target_property" \
        --seed "$seed" \
        --split_ratio "$split_ratio" \
        --simple_layer_dict "$simple_layer_dict" \
        --seq_embed_layer_dict "$seq_embed_layer_dict" \
        --common_embed_layer_dict "$common_embed_layer_dict" \
        --concat_layer_dict "$concat_layer_dict" \
        --norm "$norm" \
        --mix_layer "$mix_layer" \
        --train_batch_list "$train_batch_list" \
        --epoch_list "$epoch_list" \
        --lr_list "$lr_list" \
        --step_list "$step_list" \
        --gamma_ratio_list "$gamma_ratio_list" \
        --dropout_list "$dropout_list"

    echo "Completed: $local_model with $target_property (Seed: $seed)"
    echo "========================================"
    echo ""
}


CSV_FILE="./outputs/reg_model.csv"
mkdir -p ./outputs
echo "model_name,seed,prop,split_ratio,train_batch,epoch,lr,step,gamma_ratio,mix_layer,dropout,train_r2,test_r2,train_mae,test_mae,train_rmse,test_rmse,create_time" > "$CSV_FILE"

echo "📋 CSV : $CSV_FILE"


progress=0

for target_property in "${target_propertys[@]}"; do
    

    for ((t=0; t<num_trial; t++)); do
        
        gpu_id=${GPUS[$((progress % ${#GPUS[@]}))]}
        
        run_experiment "$target_property" "$t" "$gpu_id" "$model_name" &
        
        progress=$((progress + 1))
        
        while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
            sleep 2 
        done
    done
done

wait

echo -e "\n✅ All experiments completed."