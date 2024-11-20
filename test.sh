#!/bin/bash
export MASTER_PORT=6255
export CUDA_VISIBLE_DEVICES=0

model_type="ResNeXt_Attention_Gating_Moe"
logit_method='ResNeXt_Attention_Gating_Moe'
criterion_type="ajs_uskd_mixed"
for fold in {0..9}
    do
    next_fold=$(($fold + 1))
    #/data/wuhuixuan/code/Self_Distill_MoE/out/ResNeXt_Attention_Gating_Moe/num_expert3/ajs_uskd_mixed/1/1.0_1.0_1.0/var/No_Use_Weights/Use_criterion_total/final_output/Focal_weight_1.0/train/ResNeXt_Attention_Gating_Moe/model_best.pth.tar
    resume="/data/wuhuixuan/code/Self_Distill_MoE/out/$model_type/num_expert3/$criterion_type/${next_fold}/1.0_1.0_1.0/var/No_Use_Weights/Use_criterion_total/final_output/Focal_weight_1.0/train/$model_type/model_best.pth.tar"
    python main.py --fold $fold --model_type "$model_type" --batch-size 8 --epoch 150 --logit_method "$logit_method" --criterion_type "$criterion_type" --logit_alpha 0.5 --logit_beta 0.5 --logit_gamma 0.5 --weights 1 1 1 --use_criterion_total True --extra_dim 22 --diversity_metric "var" --extra_input True --evaluate --resume $resume --display_experts True
done
