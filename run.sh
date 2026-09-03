bash tools_mmdet/dist_train.sh configs/recondet/recondet_scannet.py 1
bash mmdet_tools/dist_test.sh configs/gdino/grounding_dino_swin-t_fine_tune_scannet.py /root/shared-nvme/data/pretrain/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth 1 --show-dir /root/shared-nvme/code/Recondet_v7/work_dirs/vis
