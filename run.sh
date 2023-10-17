DIR=fsdv2
WORK=work_dirs
CONFIG=fsdv2_torc_2x
bash tools/dist_train.sh configs/$DIR/$CONFIG.py 4 --work-dir ./$WORK/$CONFIG/ --cfg-options evaluation.jsonfile_prefix=./$WORK/$CONFIG/results evaluation.metric=bbox --launcher pytorch --seed 1
