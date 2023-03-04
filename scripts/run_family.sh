LOG=$1
TASK=family
TRIAL=${TASK}
CUDA_VISIBLE_DEVICES=$2

. graph_completion/src/eval/collect_all_facts.sh graph_completion/datasets/$TASK
python graph_completion/src/eval/get_truths.py graph_completion/datasets/$TASK

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PYTHONPATH=. python graph_completion/src/train/main.py \
    --datadir graph_completion/datasets/$TASK \
    --model lerp --soft_logic naive-prob-matmul --max_epoch 30 \
    --learning_rate 0.1 --width 40 --depth 2 --rank 10 --length 3 --init_var 0.1 \
    --dense --headwise --headwise_batch_size 128 \
    --exps_dir=$LOG --exp_name=$TRIAL \
    --gpu --no_early_stop
python graph_completion/src/eval/evaluate.py \
    --preds=$LOG/$TRIAL/test_predictions.txt \
    --truths=graph_completion/datasets/$TASK/truths.pckl
