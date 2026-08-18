import numpy as np
import torch.nn as nn


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

def comprehensive_metric(pred, true, scaler_act, scaler_fut):
    pred = scaler_act.inverse_transform(pred.reshape(-1,25)).reshape(-1,30,25)*(180/np.pi)
    true = scaler_fut.inverse_transform(true.reshape(-1,25)).reshape(-1,30,25)*(180/np.pi)

    mae = MAPE(pred, true)
    mse = MSE(pred, true)

    return np.stack([mae, mse])


if __name__ == "__main__":
    from tools import load_scaler


    path = 'checkpoints/Ours_VP_np'
    mae, mse = 0, 0
    for seed in [608, 1247, 3224]:
        path_scaler = path + f'/{seed}'
        path_pred = path + f'/{seed}/results/preds.npy'
        path_true = path + f'/{seed}/results/gts.npy'

        scaler_act, scaler_fut = load_scaler(path_scaler, 'VP', 60, 30)
        pred = np.load(path_pred)
        true = np.load(path_true)
        metrics = comprehensive_metric(pred, true, scaler_act, scaler_fut)

        mae += metrics[0].item()
        mse += metrics[1].item()

    mae /= 3
    mse /= 3
    print(mae)