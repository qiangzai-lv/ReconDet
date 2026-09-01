bash tools/dist_train.sh configs/gdino/grounding_dino_swin-t_fine_tune_scannet.py 1
bash tools/dist_test.sh configs/recondet/recondet_scannet.py /root/shared-nvme/code/ReconDet_v2/work_dirs/recondet_scannet/epoch_24.pth 1
