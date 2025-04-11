import numpy as np
import datetime
import os
import csv
import h5py
import copy
import torch
import os.path as osp
from trainer import Trainer
from utils import Averager, ensure_path, get_metrics
from sklearn.model_selection import KFold
import pickle
from os.path import exists

ROOT = os.getcwd()


class CrossValidation:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        # Log the results per subject
        result_path = osp.join(args.save_path, 'result')
        ensure_path(result_path)
        self.text_file = osp.join(result_path,
                                  "results_{}_{}.txt".format(args.dataset, args.gpu))
        file = open(self.text_file, 'a')
        file.write("\n" + str(datetime.datetime.now()) +
                   "\nTrain:Parameter setting for " + str(args.model) + ' on ' + str(args.dataset) +
                   "\n1)number_class:" + str(args.num_class) +
                   "\n2)random_seed:" + str(args.random_seed) +
                   "\n3)learning_rate:" + str(args.learning_rate) +
                   "\n4)pool:" + str(args.pool) +
                   "\n5)num_epochs:" + str(args.max_epoch) +
                   "\n6)batch_size:" + str(args.batch_size) +
                   "\n7)dropout:" + str(args.dropout) +
                   "\n8)hidden_node:" + str(args.hidden) +
                   "\n9)input_shape:" + str(args.input_shape) +
                   "\n10)class:" + str(args.label_type) +
                   "\n11)T:" + str(args.T) +
                   "\n12)graph-type:" + str(args.graph_type) + '\n')
        file.close()

    def load_per_subject(self, sub):
        """
        load data for sub
        :param sub: which subject's data to load
        :return: data and label
        """
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        sub_code = 'sub' + str(sub) + '.hdf'
        path = osp.join(save_path, data_type, sub_code)
        dataset = h5py.File(path, 'r')
        data = np.array(dataset['data'])
        label = np.array(dataset['label'])
        ###print('>>> Data:{} Label:{}'.format(data.shape, label.shape))
        return data, label

    def prepare_data(self, idx_train, idx_test, data, label):  ###old
        """
        1. get training and testing data according to the index
        2. numpy.array-->torch.tensor
        :param idx_train: index of training data
        :param idx_test: index of testing data
        :param data: (segments, 1, channel, data)
        :param label: (segments,)
        :return: data and label
        """
        data_train = data[idx_train]
        label_train = label[idx_train]
        data_test = data[idx_test]
        label_test = label[idx_test]

        if self.args.dataset == 'Att' or self.args.dataset == 'DEAP':
            """
            For DEAP we want to do trial-wise 10-fold, so the idx_train/idx_test is for
            trials.
            data: (trial, segment, 1, chan, datapoint)
            To use the normalization function, we should change the dimension from
            (trial, segment, 1, chan, datapoint) to (trial*segments, 1, chan, datapoint)
            """
            data_train = np.concatenate(data_train, axis=0)
            label_train = np.concatenate(label_train, axis=0)
            if len(data_test.shape) > 4:
                """
                When leave one trial out is conducted, the test data will be (segments, 1, chan, datapoint), hence,
                no need to concatenate the first dimension to get trial*segments
                """
                data_test = np.concatenate(data_test, axis=0)
                label_test = np.concatenate(label_test, axis=0)

        data_train, data_test = self.normalize(train=data_train, test=data_test)
        # Prepare the data format for training the model using PyTorch
        data_train = torch.from_numpy(data_train).float()
        label_train = torch.from_numpy(label_train).long()

        data_test = torch.from_numpy(data_test).float()
        label_test = torch.from_numpy(label_test).long()
        return data_train, label_train, data_test, label_test

    def prepare_data1(self, data_train, label_train, data_test, label_test):
        data_train, label_train = np.concatenate(data_train), np.concatenate(label_train)
        data_test, label_test = np.concatenate(data_test), np.concatenate(label_test)

        data_train = torch.from_numpy(data_train).float()
        label_train = torch.from_numpy(label_train).long()

        data_test = torch.from_numpy(data_test).float()
        label_test = torch.from_numpy(label_test).long()
        return data_train, label_train, data_test, label_test

    def normalize(self, train, test):
        """
        this function do standard normalization for EEG channel by channel
        :param train: training data (sample, 1, chan, datapoint)
        :param test: testing data (sample, 1, chan, datapoint)
        :return: normalized training and testing data
        """
        # data: sample x 1 x channel x data
        for channel in range(train.shape[2]):
            mean = np.mean(train[:, :, channel, :])
            std = np.std(train[:, :, channel, :])
            train[:, :, channel, :] = (train[:, :, channel, :] - mean) / std
            test[:, :, channel, :] = (test[:, :, channel, :] - mean) / std
        return train, test

    def split_balance_class(self, data, label, train_rate, random):
        """
        Get the validation set using the same percentage of the two classe samples
        :param data: training data (segment, 1, channel, data)
        :param label: (segments,)
        :param train_rate: the percentage of trianing data
        :param random: bool, whether to shuffle the training data before get the validation data
        :return: data_trian, label_train, and data_val, label_val
        """
        # Data dimension: segment x 1 x channel x data
        # Label dimension: segment x 1
        np.random.seed(0)
        # data : segments x 1 x channel x data
        # label : segments

        index_0 = np.where(label == 0)[0]
        index_1 = np.where(label == 1)[0]

        # for class 0
        index_random_0 = copy.deepcopy(index_0)

        # for class
        # .1
        index_random_1 = copy.deepcopy(index_1)

        if random == True:
            np.random.shuffle(index_random_0)
            np.random.shuffle(index_random_1)

        index_train = np.concatenate((index_random_0[:int(len(index_random_0) * train_rate)],
                                      index_random_1[:int(len(index_random_1) * train_rate)]),
                                     axis=0)
        index_val = np.concatenate((index_random_0[int(len(index_random_0) * train_rate):],
                                    index_random_1[int(len(index_random_1) * train_rate):]),
                                   axis=0)

        # get validation
        val = data[index_val]
        val_label = label[index_val]

        train = data[index_train]
        train_label = label[index_train]

        return train, train_label, val, val_label


    def leave_sub_out(self, subject=[], shuffle=True, reproduce=False):
        """
        this function achieves leave one subject out cross-validation
        :param subject: the list of subject index
        :param shuffle: whether to shuffle the data before train-val split (not trian-test split!!!)
        :param reproduce: whether to reproduce the previous results
        """
        # Train and evaluate the model subject by subject
        tta = []  # total test accuracy
        tva = []  # total validation accuracy
        ttf = []  # total test f1
        tvf = []  # total validation f1

        for sub in subject:
            data_train, label_train = [], []
            data_test, label_test = self.load_per_subject(sub)
            for sub_ in subject:
                if sub != sub_:
                    data_temp, label_temp = self.load_per_subject(sub_)
                    data_train.extend(data_temp)
                    label_train.extend(label_temp)

            va_val = Averager()
            vf_val = Averager()
            preds, acts = [], []
            idx_fold = 0
            data_train, label_train, data_test, label_test = self.prepare_data1(
                data_train=data_train, label_train=label_train, data_test=data_test, label_test=label_test
            )
            print("Training:{}  Test: {}".format(data_train.size(), data_test.size()))
            trainer = Trainer(args=self.args, subject=sub, fold_ID=idx_fold)
            if self.args.reproduce:
                # to reproduce the reported ACC
                acc_test, pred, act = trainer.test(args=self.args, data=data_test, label=label_test,
                                                   reproduce=self.args.reproduce,
                                                   subject=sub, fold=idx_fold)
                acc_val = 0
                f1_val = 0
            else:
                # to train new models
                acc_val, f1_val = self.train_with_test(data=data_train, label=label_train,
                                                       data_test=data_test, label_test=label_test,
                                                       subject=sub, fold=idx_fold, trainer=trainer)


                acc_test, pred, act = trainer.test(args=self.args, data=data_test, label=label_test,
                                                   reproduce=self.args.reproduce,
                                                   subject=sub, fold=idx_fold)
                va_val.add(acc_val)
                vf_val.add(f1_val)
                preds.extend(pred)
                acts.extend(act)

            tva.append(va_val.item())
            tvf.append(vf_val.item())
            acc, f1, _ = get_metrics(y_pred=preds, y_true=acts)
            tta.append(acc)
            ttf.append(f1)
            result = '{},{}'.format(tta[-1], f1)
            self.log2txt(result)

        # prepare final report
        tta = np.array(tta)
        ttf = np.array(ttf)
        tva = np.array(tva)
        tvf = np.array(tvf)
        mACC = np.mean(tta)
        std = np.std(tta)
        mF1 = np.mean(ttf)
        stdf = np.std(ttf)
        mACC_val = np.mean(tva)
        std_val = np.std(tva)
        mF1_val = np.mean(tvf)
        stdf_val = np.std(tvf)

        print('Final: test mean ACC:{} std:{}'.format(mACC, std))
        print('Final: test mean F1:{} std:{}'.format(mF1, stdf))
        print('Final: val mean ACC:{} std:{}'.format(mACC_val, std_val))
        print('Final: val mean F1:{} std:{}'.format(mF1_val, stdf_val))
        results = 'test mAcc={} std:{} mF1={} std:{} val mAcc={} std:{} val F1={} std:{}'.format(mACC, std,
                                                                     mF1, stdf, mACC_val, std_val, mF1_val, stdf_val)
        self.log2txt(results)

    def train_with_test(self, data, label, data_test, label_test, subject, fold, trainer):
        """
        this function achieves n-fold-CV to:
            1. select hyper-parameters on training data
            2. get the model for evaluation on testing data
        :param data: (segments, 1, channel, data)
        :param label: (segments,)
        :param subject: which subject the data belongs to
        :param fold: which fold the data belongs to
        :return: mean validation accuracy
        """
        # save the best performance model and the corresponding acc for the second stage
        # data: trial x 1 x channel x time

        va = Averager()
        vf = Averager()
        va_item = []

        data_train, label_train, data_val, label_val = self.split_balance_class(
            data=data, label=label, train_rate=0.8, random=True
        )
        if self.args.model == 'SERA':
            acc_val, F1_val = trainer.train_with_DA(args=self.args,
                                                    data_train=data_train,
                                                    label_train=label_train,
                                                    data_val=data_val,
                                                    label_val=label_val,
                                                    data_test=data_test,
                                                    label_test=label_test,
                                                    subject=subject,
                                                    fold=fold)
        else:
            acc_val, F1_val = trainer.train(args=self.args,
                                            data_train=data_train,
                                            label_train=label_train,
                                            data_val=data_val,
                                            label_val=label_val,
                                            data_test=data_test,
                                            label_test=label_test,
                                            subject=subject,
                                            fold=fold)
        print('New max ACC model saved, with the val ACC being:{}'.format(acc_val))

        return acc_val, F1_val

    def log2txt(self, content):
        """
        this function log the content to results.txt
        :param content: string, the content to log
        """
        file = open(self.text_file, 'a')
        file.write(str(content) + '\n')
        file.close()
