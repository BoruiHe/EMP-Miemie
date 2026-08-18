import os
import time
import torch
import numpy as np
import torch.nn as nn
from dataset import Miedataset
from sklearn.svm import LinearSVC
from model.Our_model import Model
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
from sklearn.preprocessing import Normalizer
from dataset_Egofalls import Miedataset_Egofalls
from sklearn.model_selection import cross_validate
from sklearn.metrics import balanced_accuracy_score
from torchvision.models import resnet50, ResNet50_Weights
from utils.miscellaneous import EarlyStopping, adjust_learning_rate, get_params_dict


def train_encoder(args):
    train_data = Miedataset(args, mode='training')
    vali_data = Miedataset(args, mode='validation')
    test_data = Miedataset(args, mode='testing')
    train_loader = DataLoader(train_data,
                             num_workers=args.num_workers,
                             shuffle=True,
                             batch_size=args.batch_size,
                             drop_last=False)
    vali_loader = DataLoader(vali_data,
                             num_workers=args.num_workers,
                             shuffle=True,
                             batch_size=args.batch_size,
                             drop_last=False)
    test_loader = DataLoader(test_data,
                             num_workers=args.num_workers,
                             shuffle=True,
                             batch_size=args.batch_size,
                             drop_last=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(args).to(device)
    print(f'Running experiments on {device}.')
    path = os.path.join(args.checkpoints, str(args.seed))

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    params_dict_ls = get_params_dict(args, model)
    model_optim = torch.optim.Adam(params_dict_ls) # model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    criterion = nn.CrossEntropyLoss()

    time_consumption, num_eps = 0., 0
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (actuals, future, imgs, planned_traj, labels) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()

            actuals = actuals.float().to(device)
            future = future.float().to(device)
            labels = labels.float().to(device)
            imgs = imgs.float().to(device)
            planned_traj = planned_traj.float().to(device)

            outputs = model(actuals, imgs, planned_traj)

            loss = criterion(outputs, labels)
            train_loss.append(loss.item())

            loss.backward()
            model_optim.step()

        ep_time_consump = time.time() - epoch_time
        time_consumption += ep_time_consump
        num_eps += 1
        print("Epoch: {} cost time: {}".format(epoch + 1, ep_time_consump))
        train_loss = np.average(train_loss)
        vali_loss, vali_acc = vali(model, vali_loader, criterion)
        test_loss, test_acc = vali(model, test_loader, criterion)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Vali Acc: {4:.3f} Test Loss: {5:.7f} Test Acc: {6:.3f}".format(
            epoch + 1, train_steps, train_loss, vali_loss, vali_acc, test_loss, test_acc))
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        adjust_learning_rate(model_optim, epoch + 1, args)

    f = open(f'result_{args.exp_tag}.txt', 'a')
    f.write(f'------ {args.exp_tag} ------\n')
    f.write('Seed:{}, Time:{}'.format(args.seed, time_consumption/num_eps))
    # f.write('\n')
    # f.write('\n')
    f.close()

    return num_eps

def vali(model, vali_loader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_loss = []
    preds = []
    trues = []
    model.eval()
    with torch.no_grad():
        for i, (actuals, future, imgs, planned_traj, labels) in enumerate(vali_loader):
            actuals = actuals.float().to(device)
            future = future.float().to(device)
            labels = labels.float().to(device)
            imgs = imgs.float().to(device)
            planned_traj = planned_traj.float().to(device)

            outputs = model(actuals, imgs, planned_traj)

            loss = criterion(outputs.detach(), labels)
            total_loss.append(loss.item())

            preds.append(outputs.detach().cpu().numpy())
            trues.append(labels.detach().cpu().numpy())

    total_loss = np.average(total_loss)

    preds = np.concatenate(preds, 0)
    trues = np.concatenate(trues, 0)
    acc = np.mean(np.argmax(preds, axis=1) == np.argmax(trues, axis=1))

    model.train()
    return total_loss, acc

def test_encoder(args):
    test_data = Miedataset(args, mode='testing')
    test_loader = DataLoader(test_data,
                             num_workers=args.num_workers,
                             shuffle=True,
                             batch_size=args.batch_size,
                             drop_last=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(args).to(device)
    print('loading model')
    model.load_state_dict(torch.load(os.path.join(args.checkpoints, str(args.seed), 'checkpoint.pth')))
    
    preds = []
    trues = []
    ipts = []

    model.eval()
    with torch.no_grad():
        for i, (actuals, future, imgs, planned_traj, labels) in enumerate(test_loader):
            actuals = actuals.float().to(device)
            future = future.float()
            labels = labels.float()
            imgs = imgs.float().to(device)
            planned_traj = planned_traj.float().to(device)

            outputs = model(actuals, imgs, planned_traj)

            preds.append(outputs.detach().cpu().numpy())
            trues.append(labels.numpy())
            ipts.append(actuals.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    ipts = np.concatenate(ipts, axis=0)

    # result save
    folder_path = os.path.join(args.checkpoints, str(args.seed))

    acc = np.mean(np.argmax(preds, axis=1) == np.argmax(trues, axis=1))
    print('Accuracy:{}'.format(acc))
    f = open(f'result_{args.exp_tag}.txt', 'a')
    f.write('Seed:{}, Accuracy:{}'.format(args.seed, acc))
    f.write('\n')
    f.write('\n')
    f.close()

    # np.save(folder_path + '/metrics.npy', acc)
    # np.save(folder_path + '/pred.npy', preds)
    # np.save(folder_path + '/true.npy', trues)
    # np.save(folder_path + '/inputs.npy', ipts)

    return acc

'''
Compatible methods are developed for Egofalls only.
'''
def Egofalls_cpt(args):
    train_data = Miedataset_Egofalls(args, mode='training')
    vali_data = Miedataset_Egofalls(args, mode='validation')
    test_data = Miedataset_Egofalls(args, mode='testing')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    model = model.to(device)

    features_train, labels_train = [], []
    features_test, labels_test = [], []

    train_loader = DataLoader(train_data,
                             num_workers=args.num_workers,
                             shuffle=True,
                             batch_size=args.batch_size,
                             drop_last=False)
    vali_loader = DataLoader(vali_data,
                             num_workers=args.num_workers,
                             shuffle=True,
                             batch_size=args.batch_size,
                             drop_last=False)
    test_loader = DataLoader(test_data,
                             num_workers=args.num_workers,
                             shuffle=True,
                             batch_size=args.batch_size,
                             drop_last=False)
    print(f'Running experiments on {device}.')
    path = os.path.join(args.checkpoints, str(args.seed))

    for i, (_, _, imgs, _, labels) in enumerate(train_loader):

        imgs = imgs.float().to(device)

        for frames in imgs:
            res_feature = model(frames.permute(1,0,2,3))
            res_feature = res_feature.detach().cpu()
            features_train.append(res_feature.flatten().numpy())
        labels_train.append(labels.numpy())
    for i, (_, _, imgs, _, labels) in enumerate(vali_loader):

        imgs = imgs.float().to(device)

        for frames in imgs:
            res_feature = model(frames.permute(1,0,2,3))
            res_feature = res_feature.detach().cpu()
            features_train.append(res_feature.flatten().numpy())
        labels_train.append(labels.numpy())
    for i, (_, _, imgs, _, labels) in enumerate(test_loader):

        imgs = imgs.float().to(device)

        for frames in imgs:
            res_feature = model(frames.permute(1,0,2,3))
            res_feature = res_feature.detach().cpu()
            features_test.append(res_feature.flatten().numpy())
        labels_test.append(labels.numpy())

    features_train = np.array(features_train)
    labels_train = np.concatenate(labels_train)
    # print(features_train.shape, labels_train.shape)

    features_test = np.array(features_test)
    labels_test = np.concatenate(labels_test)
    # print(features_test.shape, labels_test.shape)

    clf = LinearSVC(dual='auto', C=args.C, class_weight='balanced', random_state=args.seed)
    pipe = Pipeline([('scaler', Normalizer()), ('estimator', clf)])
    # labels_train = np.argmax(labels_train, axis=1)
    # results = cross_validate(pipe, features_train, labels_train, scoring=('accuracy', 'balanced_accuracy'), cv=args.kfold)
    # print('testing accuracy: {}; testing balanced accuracy: {}.'.format(round(results['test_accuracy'].mean()*100, 2), round(results['test_balanced_accuracy'].mean()*100, 2)))
    scaler = Normalizer()
    normalized_features = scaler.fit_transform(features_train)
    clf.fit(normalized_features, labels_train)

    scaler = Normalizer()
    normalized_features = scaler.fit_transform(features_test)
    prediction = clf.predict(normalized_features)
    # labels_test = np.argmax(labels_test, axis=1)
    results = balanced_accuracy_score(labels_test, prediction)
    np.save(path + '/pred.npy', prediction)
    np.save(path + '/true.npy', labels_test)
    
    return results