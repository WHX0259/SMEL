![image](https://github.com/user-attachments/assets/ba7c3693-29f5-4d61-a230-0e05d6b09752)![image](https://github.com/user-attachments/assets/c45609ee-2096-4373-8e08-6451663cd3b7)![image](https://github.com/user-attachments/assets/4923ff74-bc6e-4438-8918-0ae371f5013e)![image](https://github.com/user-attachments/assets/43390f14-9635-4743-baa2-f3af45ccc941)![image](https://github.com/user-attachments/assets/53a60ce3-36cd-41c7-a5fb-78564733be23)![image](https://github.com/user-attachments/assets/c1eee927-8e72-41de-a046-b288a4d57453)# Self-Distillation Multi-Expert Ensemble Learning for Predicting Chemotherapy Efficacy of Gastric Cancer

# Introduction
The Pytorch official implementation of:
Self-Distillation Multi-Expert Ensemble Learning for Predicting Chemotherapy Efficacy of Gastric Cancer

# Requirements
python 3.9
torch 2.0.1
torchaudio 2.0.2 
torchvision 0.15.2    
cuda 11.8
Please refer to the specific details in requirements.txt.

# Running the code
1. Setting up the Python environment
The code has been tested with Python 3.9.19. The necessary libraries can be installed using pip with the following command:
`# from SMEL/
pip install -r requirements.txt
`
2. Dataset setting
A dataset function is required to read your own data. The SMEL model needs both image data and radiomics data. The dimension of the extracted and filtered radiomics data determines the setting of the parameter `extra_dim`.
3. Running the code
You can now run the code on your dataset. You need to write a shell script file (with the extension `.sh`). Below is an example. You can set your parameters in the shell script file.
`
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
`
Note: If you do not need ten-fold cross-validation, you can set the fold to any number or leave it unset. You should remove the `for` loop statement in the `.sh` file.
