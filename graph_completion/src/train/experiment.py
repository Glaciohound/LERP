import sys
import os
import time
import pickle
import torch
# from collections import Counter
import numpy as np
from tqdm import tqdm


class Experiment():
    """
    This class handles all experiments related activties,
    including training, testing, early stop, and visualize
    results, such as get attentions and get rules.

    Args:
        sess: a TensorFlow session
        saver: a TensorFlow saver
        option: an Option object that contains hyper parameters
        learner: an inductive learner that can
                 update its parameters and perform inference.
        data: a Data object that can be used to obtain
              num_batch_train/valid/test,
              next_train/valid/test,
              and a parser for get rules.
    """

    # def __init__(self, sess, saver, option, learner, data):
    def __init__(self, option, learner, optimizer, data, start_epoch=0):
        # self.sess = sess
        # self.saver = saver
        self.option = option
        self.learner = learner
        self.optimizer = optimizer
        self.data = data
        # helpers
        self.msg_with_time = lambda msg: \
            "%s Time elapsed %0.2f hrs (%0.1f mins)" \
            % (msg, (time.time() - self.start) / 3600.,
               (time.time() - self.start) / 60.)

        self.start = time.time()
        # self.epoch = 0
        self.epoch = start_epoch
        self.best_valid_loss = np.inf
        self.best_valid_in_top = 0.
        self.train_stats = []
        self.valid_stats = []
        self.test_stats = []
        self.early_stopped = False
        self.log_file = open(os.path.join(
            self.option.this_expsdir, "log.txt"), "w")
        self.no_early_stop = option.no_early_stop

    def one_epoch(self, mode, num_batch, next_fn):
        epoch_loss = []
        epoch_in_top = []
        num_batch=int(num_batch)
        pbar = tqdm(range(num_batch), leave=False)
        for batch in pbar:
            # if (batch+1) % max(1, (num_batch // self.option.print_per_batch)) == 0:
            #     sys.stdout.write("%d/%d\t" % (batch+1, num_batch))
            #     sys.stdout.flush()
            (qq, hh, tt), mdb = next_fn()
            if qq is None:
                continue
            try:
                if mode == "train":
                    self.optimizer.zero_grad()
                    loss, in_top = self.learner.update(qq, hh, tt, mdb)
                    # print("got loss", time.ctime())
                    if not self.option.headwise:
                        loss.mean().backward()
                    else:
                        (loss.sum() / self.option.batch_size).backward()
                    # print("backwarded", time.ctime())
                    # print()
                    torch.nn.utils.clip_grad_norm_(self.learner.parameters(), 5.0,
                                                   "inf")
                    self.optimizer.step()
                else:
                    loss, in_top = self.learner.predict(qq, hh, tt, mdb)

                epoch_loss += list(loss.detach().cpu().numpy())
                epoch_in_top += list(in_top.cpu().numpy())
            except RuntimeError as e:
                print(e)
                raise e
            torch.cuda.empty_cache()
            if mode == "train":
                pbar.set_description(
                    "loss:{:.3f},in_top:{:.3f}".format(
                        self.learner.running_mean["train"]["loss"].value,
                        self.learner.running_mean["train"]["in_top"].value,
                    )
                )

        msg = self.msg_with_time(
                "Epoch %d mode %s Loss %0.4f In top %0.4f."
                % (self.epoch+1, mode, np.mean(epoch_loss), np.mean(epoch_in_top)))
        print(msg)
        self.log_file.write(msg + "\n")
        return epoch_loss, epoch_in_top

    def one_epoch_train(self):
        if self.epoch > 0 and self.option.resplit:
            self.data.train_resplit(self.option.no_link_percent)
        loss, in_top = self.one_epoch("train",
                                      self.data.num_batch_train,
                                      self.data.next_train)

        self.train_stats.append([loss, in_top])

    def one_epoch_valid(self):
        loss, in_top = self.one_epoch("valid",
                                      self.data.num_batch_valid,
                                      self.data.next_valid)
        self.valid_stats.append([loss, in_top])
        self.best_valid_loss = min(self.best_valid_loss, np.mean(loss))
        self.best_valid_in_top = max(self.best_valid_in_top, np.mean(in_top))

    def one_epoch_test(self):
        loss, in_top = self.one_epoch("test",
                                      self.data.num_batch_test,
                                      self.data.next_test)
        self.test_stats.append([loss, in_top])

    def early_stop(self):
        if self.no_early_stop:
            return False
        loss_improve = self.best_valid_loss == np.mean(self.valid_stats[-1][0])
        in_top_improve = self.best_valid_in_top == np.mean(self.valid_stats[-1][1])
        if loss_improve or in_top_improve:
            return False
        else:
            if self.epoch < self.option.min_epoch:
                return False
            else:
                return True

    def train(self):
        if self.epoch >= self.option.max_epoch:
            return
        while (self.epoch < self.option.max_epoch and not self.early_stopped):
            self.one_epoch_train()
            if not self.option.train_only:
                self.one_epoch_valid()
                self.one_epoch_test()
            self.epoch += 1
            model_path = self.option.model_path + str(self.epoch)
            torch.save(
                {"learner": self.learner.state_dict(),
                 "optimizer": self.optimizer.state_dict(),
                 "i_epoch": self.epoch},
                model_path,
            )
            print("Model saved at %s" % model_path)
            if self.epoch < self.option.max_epoch - 1 and not self.option.train_only:
                self.get_predictions()

            if self.early_stop():
                self.early_stopped = True
                print("Early stopped at epoch %d" % (self.epoch))

        if self.option.train_only:
            self.get_predictions()
        all_test_in_top = [np.mean(x[1]) for x in self.test_stats]
        best_test_epoch = np.argmax(all_test_in_top)
        best_test = all_test_in_top[best_test_epoch]

        msg = "Best test in top: %0.4f at epoch %d." % (best_test, best_test_epoch + 1)
        print(msg)
        self.log_file.write(msg + "\n")
        pickle.dump([self.train_stats, self.valid_stats, self.test_stats],
                    open(os.path.join(self.option.this_expsdir, "results.pckl"), "wb"))

    def get_predictions(self):
        if self.option.query_is_language:
            all_accu = []
            all_num_preds = []
            all_num_preds_no_mistake = []

        f = open(os.path.join(self.option.this_expsdir, "test_predictions.txt"), "w")
        if self.option.get_phead:
            f_p = open(os.path.join(self.option.this_expsdir, "test_preds_and_probs.txt"), "w")
        all_in_top = []
        printed = False
        for batch in tqdm(range(self.data.num_batch_test), leave=False):
            torch.cuda.empty_cache()
            if (batch+1) % max(1, (self.data.num_batch_test // self.option.print_per_batch)) == 0:
                sys.stdout.write("%d/%d\t" % (batch+1, self.data.num_batch_test))
                sys.stdout.flush()
            (qq, hh, tt), mdb = self.data.next_test()
            if qq is None:
                continue
            #print(mdb)
            in_top, predictions_this_batch \
                    = self.learner.get_predictions_given_queries(qq, hh, tt, mdb)
            if not printed:
                print(predictions_this_batch)
                printed = True
            predictions_this_batch = predictions_this_batch.\
                detach().cpu().numpy()
            all_in_top += list(in_top.cpu().numpy())

            for i, (q, h, t) in enumerate(zip(qq, hh, tt)):
                p_head = predictions_this_batch[i, h]
                if self.option.adv_rank:
                    eval_fn = lambda t: t[1] >= p_head and (t[0] != h)
                elif self.option.rand_break:
                    eval_fn = lambda t: (t[1] >= p_head) or ((t[1] == p_head) and (t[0] != h) and (np.random.uniform() < 0.5))
                elif self.option.rank_geq:
                    eval_fn = lambda t: (t[1] >= p_head)
                else:
                    eval_fn = lambda t: (t[1] > p_head)
                this_predictions = filter(eval_fn, enumerate(predictions_this_batch[i, :]))
                this_predictions = sorted(this_predictions, key=lambda x: x[1], reverse=True)
                if self.option.query_is_language:
                    all_num_preds.append(len(this_predictions))
                    mistake = False
                    for k, _ in this_predictions:
                        assert(k != h)
                        if not self.data.is_true(q, k, t):
                            mistake = True
                            break
                    all_accu.append(not mistake)
                    if not mistake:
                        all_num_preds_no_mistake.append(len(this_predictions))
                else:
                    this_predictions.append((h, p_head))
                    this_predictions = [self.data.number_to_entity[j] for j, _ in this_predictions]
                    q_string = self.data.parser["query"][q]
                    h_string = self.data.number_to_entity[h]
                    t_string = self.data.number_to_entity[t]
                    to_write = [q_string, h_string, t_string] + this_predictions
                    f.write(",".join(to_write) + "\n")
                    if self.option.get_phead:
                        f_p.write(",".join(to_write + [str(p_head)]) + "\n")
        f.close()
        if self.option.get_phead:
            f_p.close()

        if self.option.query_is_language:
            print("Averaged num of preds", np.mean(all_num_preds))
            print("Averaged num of preds for no mistake", np.mean(all_num_preds_no_mistake))
            msg = "Accuracy %0.4f" % np.mean(all_accu)
            print(msg)
            self.log_file.write(msg + "\n")

        msg = "Test in top %0.4f" % np.mean(all_in_top)
        msg += self.msg_with_time("\nTest predictions written.")
        print(msg)
        self.log_file.write(msg + "\n")

    def get_vocab_embedding(self):
        vocab_embedding = self.learner.get_vocab_embedding()
        vocab_embedding = vocab_embedding.detach().cpu().numpy()
        msg = self.msg_with_time("Vocabulary embedding retrieved.")
        print(msg)
        self.log_file.write(msg + "\n")

        vocab_embed_file = os.path.join(self.option.this_expsdir, "vocab_embed.pckl")
        pickle.dump({"embedding": vocab_embedding, "labels": self.data.query_vocab_to_number}, open(vocab_embed_file, "wb"))
        msg = self.msg_with_time("Vocabulary embedding stored.")
        print(msg)
        self.log_file.write(msg + "\n")

    def close_log_file(self):
        self.log_file.close()
