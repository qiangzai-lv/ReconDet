bash tools/dist_train.sh configs/recondet/recondet_scannet.py 4
bash tools/dist_test.sh configs/recondet/recondet_scannet.py /root/shared-nvme/code/ReconDet_v2/work_dirs/recondet_scannet/epoch_1.pth 4
