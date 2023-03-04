LOG=$1
TASK=WN18RR
TRIAL=${TASK}
CUDA_VISIBLE_DEVICES=$2

. graph_completion/src/eval/collect_all_facts.sh graph_completion/datasets/$TASK
python graph_completion/src/eval/get_truths.py graph_completion/datasets/$TASK

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PYTHONPATH=. python graph_completion/src/train/main.py \
    --datadir graph_completion/datasets/$TASK \
    --model bridged_lerp --soft_logic naive-prob-matmul --max_epoch 4 \
    --learning_rate 0.1 --width 10 --depth 3 --rank 3 --init_var 0.1 \
    --headwise --headwise_batch_size 30 --train_no_facts \
    --exps_dir=$LOG --exp_name=$TRIAL \
    --gpu --sparse --no_early_stop
python graph_completion/src/eval/evaluate.py \
    --preds=$LOG/$TRIAL/test_predictions.txt \
    --truths=graph_completion/datasets/$TASK/truths.pckl
