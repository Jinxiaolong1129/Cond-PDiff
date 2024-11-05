# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export PYTHONPATH=src/:$PYTHONPATH
# # bert rank=1
# ae_bash=bert_1
# nohup python ae_train_multi_norm.py --ae_config config/multiple/ae_training.yaml \
#                                     --ae_bash config/multiple/ae_bash.yaml \
#                                     --ae_bash_args_num $ae_bash > run_log/cond_diff/ae_bash-${ae_bash}-norm.txt 2>&1 & 



# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export PYTHONPATH=src/:$PYTHONPATH
# # bert rank=2
# ae_bash=bert_2
# nohup python ae_train_multi_norm.py --ae_config config/multiple/ae_training.yaml \
#                                     --ae_bash config/multiple/ae_bash.yaml \
#                                     --ae_bash_args_num $ae_bash > run_log/cond_diff/ae_bash-${ae_bash}-norm.txt 2>&1 & 


# export CUDA_VISIBLE_DEVICES=0,7,4,5
# export PYTHONPATH=src/:$PYTHONPATH
# # bert rank=16
# ae_bash=bert_16
# nohup python ae_train_multi_norm.py --ae_config config/multiple/ae_training.yaml \
#                                     --ae_bash config/multiple/ae_bash.yaml \
#                                     --ae_bash_args_num $ae_bash > run_log/cond_diff/ae_bash-${ae_bash}-norm.txt 2>&1 & 


# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export PYTHONPATH=src/:$PYTHONPATH
# # roberta rank=1
# ae_bash=roberta_1
# nohup python ae_train_multi_norm.py --ae_config config/multiple/ae_training.yaml \
#                                     --ae_bash config/multiple/ae_bash.yaml \
#                                     --ae_bash_args_num $ae_bash > run_log/cond_diff/ae_bash-${ae_bash}-norm.txt 2>&1 & 


# export CUDA_VISIBLE_DEVICES=0,1,2,7
# export PYTHONPATH=src/:$PYTHONPATH
# # roberta rank=1
# ae_bash=roberta_2
# nohup python ae_train_multi_norm.py --ae_config config/multiple/ae_training.yaml \
#                                     --ae_bash config/multiple/ae_bash.yaml \
#                                     --ae_bash_args_num $ae_bash > run_log/cond_diff/ae_bash-${ae_bash}-norm.txt 2>&1 & 



# export CUDA_VISIBLE_DEVICES=6,7,4,5,
# export PYTHONPATH=src/:$PYTHONPATH
# ae_bash=roberta_4
# nohup python ae_train_multi_norm.py --ae_config config/multiple/ae_training.yaml \
#                                     --ae_bash config/multiple/ae_bash.yaml \
#                                     --ae_bash_args_num $ae_bash > run_log/cond_diff/ae_bash-${ae_bash}-norm.txt 2>&1 & 



# export CUDA_VISIBLE_DEVICES=3,2,1,0
# export PYTHONPATH=src/:$PYTHONPATH
# # deberta rank=1
# ae_bash=deberta_r_1
# nohup python ae_train_multi_norm.py --ae_config config/multiple/ae_training.yaml \
#                                     --ae_bash config/multiple/ae_bash.yaml \
#                                     --ae_bash_args_num $ae_bash > run_log/cond_diff/ae_bash-${ae_bash}-norm.txt 2>&1 &                                       