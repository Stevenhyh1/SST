DIR=fsdv2
WORK=work_dirs
CONFIG=fsdv2_torc_2x
bash tools/dist_test.sh configs/$DIR/$CONFIG.py ./$WORK/$CONFIG/latest.pth 8 --options "pklfile_prefix=./$WORK/$CONFIG/results" --eval fast
