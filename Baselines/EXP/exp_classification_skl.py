from data_provider.data_factory import data_provider
from EXP.exp_basic import Exp_Basic
import numpy as np
import torch
import os
import warnings
import pickle as pk
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.model_id = args.model_id

    def _build_model(self):
        model_dict = {
            'Logistic': LogisticRegression(),
            'LinearSVC': LinearSVC(),
        }
        model = model_dict[self.args.model]

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def train(self, seed):
        train_data, train_loader = self._get_data(flag='training')
        vali_data, vali_loader = self._get_data(flag='validation')

        path = os.path.join(self.args.checkpoints, self.model_id, str(seed))

        training_data_, training_label_ = [], []
        for i, (pos, label, traj, _) in enumerate(train_loader):
            
            pos = pos.float().to(self.device)
            label = label.float().to(self.device)
            traj = traj.float().to(self.device)
            # print(f'Batch {i}:', pos.shape, label.shape, traj.shape)

            # feature = torch.concat([pos, traj], dim=1).flatten(start_dim=1).cpu().numpy()
            feature = pos.flatten(start_dim=1).cpu().numpy()
            label = label.cpu().numpy()
            training_data_.append(feature)
            training_label_.append(label)

        for i, (pos, label, traj, _) in enumerate(vali_loader):
            
            pos = pos.float().to(self.device)
            label = label.float().to(self.device)
            traj = traj.float().to(self.device)
            # print(f'Batch {i}:', pos.shape, label.shape, traj.shape)

            # feature = torch.concat([pos, traj], dim=1).flatten(start_dim=1).cpu().numpy()
            feature = pos.flatten(start_dim=1).cpu().numpy()
            label = label.cpu().numpy()
            training_data_.append(feature)
            training_label_.append(label)

        training_data_ = np.concatenate(training_data_)
        training_label_ = np.concatenate(training_label_)

        self.scaler = StandardScaler()
        training_data_trfd = self.scaler.fit_transform(training_data_)

        # print(len(train_data), len(vali_data), training_data_.shape, training_label_.shape)

        self.model.fit(training_data_trfd, training_label_)
        # train_accuracy = self.model.score(training_data_, training_label_)
        # print(f'Training Accuracy: {train_accuracy:.2f}')

        best_model_path = os.path.join(path, 'checkpoint.pkl')
        with open(best_model_path, 'wb') as file:
            pk.dump(self.model, file)
        with open(best_model_path, 'rb') as file:
            self.model = pk.load(file)

        return

    def test(self, seed, test=0):
        test_data, test_loader = self._get_data(flag='testing')
        if test:
            print('loading model')
            with open(os.path.join(f'./checkpoints/{self.model_id}/{seed}', 'checkpoint.pkl'), 'rb') as file:
                self.model = pk.load(file)
        
        # preds = []
        # trues = []
        # ipts = []

        testing_data_, testing_label_ = [], []
        for i, (pos, label, traj, _) in enumerate(test_loader):
            pos = pos.float().to(self.device)
            label = label.float().to(self.device)
            traj = traj.float().to(self.device)
            # print(f'Batch {i}:', pos.shape, label.shape, traj.shape)

            # feature = torch.concat([pos, traj], dim=1).flatten(start_dim=1).cpu().numpy()
            feature = pos.flatten(start_dim=1).cpu().numpy()
            label = label.cpu().numpy()
            testing_data_.append(feature)
            testing_label_.append(label)

        testing_data_ = np.concatenate(testing_data_)
        testing_label_ = np.concatenate(testing_label_)

        testing_data_trfd = self.scaler.transform(testing_data_)

        preds = self.model.predict(testing_data_trfd)
        # trues = testing_label_
        # ipts = testing_data_

        accuracy = accuracy_score(testing_label_, preds)
        report = classification_report(testing_label_, preds)
    
        # preds = np.concatenate(preds, axis=0)
        # trues = np.concatenate(trues, axis=0)
        # ipts = np.concatenate(ipts, axis=0)
        # print('test shape:', preds.shape, trues.shape)

        # result save
        # folder_path = f'./checkpoints/{self.model_id}/{seed}/'

        # acc = np.mean(np.argmax(preds, axis=1) == np.argmax(trues, axis=1))
        # print('Accuracy:{}'.format(acc))
        f = open(f'{self.model_id}.txt', 'a')
        f.write(f'{seed}\n')
        f.write('Accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', acc)
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'inputs.npy', ipts)

        return accuracy
