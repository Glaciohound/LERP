import os
import argparse
import time
import torch
from torch.optim import Adam
from graph_completion.src.train.model_bridged_lerp import Learner \
    as BridgedLerp
from graph_completion.src.train.model_lerp import Learner \
    as Lerp
from graph_completion.src.train.model_transE import Learner as TransE

from graph_completion.src.train.data import \
    Data, DataPlus, DataHeadwise
from graph_completion.src.train.experiment import Experiment

from lerp.utils import set_seed


class Option(object):
    def __init__(self, d):
        self.__dict__ = d

    def save(self):
        with open(os.path.join(self.this_expsdir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(),
                                     key=lambda x: x[0]):
                f.write("%s, %s\n" % (key, str(value)))


def main():
    parser = argparse.ArgumentParser(description="Experiment setup")
    # misc
    parser.add_argument('--seed', default=33, type=int)
    parser.add_argument('--no_train', default=False, action="store_true")
    parser.add_argument('--from_model_ckpt', default=None, type=str)
    parser.add_argument('--rule_thr', default=1e-2, type=float)
    parser.add_argument('--no_preds', default=False, action="store_true")
    parser.add_argument('--get_vocab_embed', default=False,
                        action="store_true")
    parser.add_argument('--exps_dir', default=None, type=str)
    parser.add_argument('--exp_name', default=None, type=str)
    # data property
    parser.add_argument('--datadir', default=None, type=str)
    parser.add_argument('--resplit', default=False, action="store_true")
    parser.add_argument('--no_link_percent', default=0., type=float)
    parser.add_argument('--type_check', default=False, action="store_true")
    parser.add_argument('--domain_size', default=128, type=int)
    parser.add_argument('--no_extra_facts', default=False, action="store_true")
    parser.add_argument('--query_is_language', default=False,
                        action="store_true")
    parser.add_argument('--vocab_embed_size', default=128, type=int)
    # model architecture
    parser.add_argument('--model', type=str, default="lerp")
    parser.add_argument('--num_step', default=3, type=int)
    parser.add_argument('--num_layer', default=1, type=int)
    parser.add_argument('--rank', default=3, type=int)
    parser.add_argument('--rnn_state_size', default=128, type=int)
    parser.add_argument('--query_embed_size', default=128, type=int)
    # optimization
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--print_per_batch', default=3, type=int)
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--min_epoch', default=5, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--no_norm', default=False, action="store_true")
    parser.add_argument('--thr', default=1e-20, type=float)
    parser.add_argument('--dropout', default=0., type=float)
    # evaluation
    parser.add_argument('--get_phead', default=False, action="store_true")
    parser.add_argument('--adv_rank', default=False, action="store_true")
    parser.add_argument('--rand_break', default=False, action="store_true")
    parser.add_argument('--accuracy', default=False, action="store_true")
    parser.add_argument('--top_k', default=10, type=int)
    # Chi Han
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--sparse', action="store_true")
    parser.add_argument('--dense', action="store_true")
    parser.add_argument('--headwise', action="store_true")
    parser.add_argument('--no_rev_in_model', action="store_true")
    parser.add_argument('--soft_logic', type=str, default="minmax-prob")
    parser.add_argument('--limit_supernode', type=int, default=-1)
    parser.add_argument('--empty_cache', action="store_true")
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--length', type=int, default=4)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=50)
    parser.add_argument('--init_var', type=float, default=1e-3)
    parser.add_argument('--regulation', type=float, default=1)
    parser.add_argument('--entropy_regulation', type=float, default=0)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--no_early_stop', action="store_true")
    parser.add_argument('--rank_geq', default=False, action="store_true")
    parser.add_argument('--headwise_batch_size', default=-1, type=int)
    parser.add_argument('--induction', action="store_true")
    parser.add_argument('--data_relink', action="store_true")
    parser.add_argument('--train_no_facts', action="store_true")
    parser.add_argument('--train_only', action="store_true")

    d = vars(parser.parse_args())
    option = Option(d)
    if option.exp_name is None:
        option.tag = time.strftime("%y-%m-%d-%H-%M")
    else:
        option.tag = option.exp_name
    if option.resplit:
        assert not option.no_extra_facts
    if option.accuracy:
        assert option.top_k == 1

    torch.set_num_threads(10)
    if option.gpu:
        option.device = torch.device("cuda:0")
    else:
        option.device = torch.device("cpu")
    # tf.logging.set_verbosity(tf.logging.ERROR)

    # torch.autograd.set_detect_anomaly(True)
    if option.headwise:
        data = DataHeadwise(
            option.datadir, option.seed,
            option.type_check, option.domain_size,
            option.no_extra_facts, option.limit_supernode,
            option.induction, option.data_relink, option.train_no_facts)
        data.reset(option.headwise_batch_size)
    elif not option.query_is_language:
        data = Data(
            option.datadir, option.seed,
            option.type_check, option.domain_size,
            option.no_extra_facts, option.limit_supernode,
            option.induction, option.data_relink, option.train_no_facts)
        data.reset(option.batch_size)
    else:
        data = DataPlus(option.datadir, option.seed)
        data.reset(option.batch_size)
    print("Data prepared.")

    option.num_entity = data.num_entity
    option.num_operator = data.num_operator
    option.num_relation = data.num_relation
    if not option.query_is_language:
        option.num_query = data.num_query
    else:
        option.num_vocab = data.num_vocab
        option.num_word = data.num_word  # the number of words in each query

    option.this_expsdir = os.path.join(option.exps_dir, option.tag)
    if not os.path.exists(option.this_expsdir):
        os.makedirs(option.this_expsdir)
    option.ckpt_dir = os.path.join(option.this_expsdir, "ckpt")
    if not os.path.exists(option.ckpt_dir):
        os.makedirs(option.ckpt_dir)
    option.model_path = os.path.join(option.ckpt_dir, "model")

    option.save()
    print(option.__dict__)
    print("Option saved.")

    set_seed(option.seed)
    if option.model == "lerp":
        learner = Lerp(option)
    elif option.model.startswith("bridged_lerp"):
        learner = BridgedLerp(option)
    elif option.model in ("transE", "logical-transE"):
        learner = TransE(option)
    else:
        raise NotImplementedError()
    learner.to(option.device)
    learner.device = option.device
    optimizer = Adam(learner.parameters(), lr=option.learning_rate)
    print("Learner built.")

    start_epoch = 0
    if option.from_model_ckpt is not None:
        # saver.restore(sess, option.from_model_ckpt)
        ckpt = torch.load(option.from_model_ckpt)
        learner.load_state_dict(ckpt["learner"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["i_epoch"]
        print("Checkpoint restored from model %s" % option.from_model_ckpt)
        print("Start epoch:", start_epoch)

    # experiment = Experiment(sess, saver, option, learner, data)
    experiment = Experiment(option, learner, optimizer, data,
                            start_epoch=start_epoch)
    print("Experiment created.")

    if not option.no_train:
        print("Start training...")
        experiment.train()

    if not option.no_preds:
        print("Start getting test predictions...")
        experiment.get_predictions()

    if option.get_vocab_embed:
        print("Start getting vocabulary embedding...")
        experiment.get_vocab_embedding()

    experiment.close_log_file()
    print("="*36 + "Finish" + "="*36)


if __name__ == "__main__":
    main()
