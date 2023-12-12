import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import joblib
from sklearn import preprocessing
import os
from argparse import RawDescriptionHelpFormatter
import argparse
import datetime

class Mydataset(Dataset):
    def __init__(self, x, y):
        self.feature = x
        self.label = y
    
    def __getitem__(self, item):
        return torch.tensor(self.feature[item]),self.label[item]
    
    def __len__(self):
        return len(self.feature)

def get_dataloader(df, name, batch_size):
    """
    df: pd.DataFrame
    name: train or eval
    """
    values = df.values
    if name == "train":
        scaler = preprocessing.StandardScaler()
        feats = scaler.fit_transform(values[:, :-1])
        joblib.dump(scaler, "train_scaler.scaler")
    else:
        scaler = joblib.load("train_scaler.scaler")
        feats = scaler.transform(values[:, :-1])

    feats = torch.from_numpy(feats).to(torch.float32)
    label = torch.from_numpy(values[:, -1]).to(torch.float32)
    dataset = Mydataset(feats, label)
    shuffle = True if name == "train" else False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def RMSE(y_true, y_pred):
    return torch.sqrt(torch.mean(torch.pow(y_pred - y_true, 2), axis=-1))
    
def PCC(y_true, y_pred):
    fsp = y_pred - torch.mean(y_pred)
    fst = y_true - torch.mean(y_true)

    devP = torch.std(y_pred, unbiased=False)
    devT = torch.std(y_true, unbiased=False)

    pcc = torch.mean(fsp * fst) / (devP * devT)
    return pcc


class DeepRMSD(nn.Module):
    def __init__(self, rate):
        super(DeepRMSD, self).__init__()

        # self.flatten = torch.flatten()  # 128 * 7 * 210
        self.fc1 = nn.Sequential(
            nn.Linear(1470, 1024),
            nn.ReLU(),
            nn.Dropout(p=rate),
            nn.BatchNorm1d(1024),
        )
    
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=rate),
            nn.BatchNorm1d(512),
            )

        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=rate),
            nn.BatchNorm1d(256),
            )

        self.fc4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=rate),
            nn.BatchNorm1d(128),
        )

        self.fc5 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=rate),
            nn.BatchNorm1d(64),
        )

        self.out = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        out = self.out(x)
        return out

if __name__ == "__main__":
    print("Start training ...")

    d = """
        Usage:
            python train.py -train_file $train_file -valid_file $valid_file
   
    """

    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("-train_file", type=str, default="train_features_label.csv",
                        help="Input. This input file should include the features and label \n"
                            "of complexes in the training set.")
    parser.add_argument("-valid_file", type=str, default="valid_features_label.csv",
                        help="Input. This input file shuould include the features and label \n"
                            "of complexes in the validating set.")
    parser.add_argument("-lr", type=float, default=0.001,
                        help="Input. The learning rate.")
    parser.add_argument("-batchsz", type=int, default=64,
                        help="Input. The number of samples processed per batch.")
    parser.add_argument("-rate", type=float, default=0.0,
                        help="Input. The dropout rate.")
    parser.add_argument("-epochs", type=int, default=300,
                        help="Input. The number of times all samples in the training set pass the CNN model.")
    parser.add_argument("-out", type=str, default="predicted_pKa.csv",
                        help="Output. The predicted pKa of the complex on the test set.")
    parser.add_argument("-device", type=str, default="cuda:0",
                        help="Output. The device cpu or cuda.")

    args = parser.parse_args()

    # load datasets
    train_data = pd.read_pickle(args.train_file)
    print("train data loaded ...")
    valid_data = pd.read_pickle(args.valid_file)
    print("valid data loaded ...")

    scaler = preprocessing.StandardScaler()
    
    # params
    batch_size = args.batchsz
    rate = args.rate
    lr = args.lr
    EPOCH = 150
    end = 1470
    shape = [-1, 1470]
    patience = 20
    min_delta = 0.001
    device = args.device

    train_dataloader = get_dataloader(train_data, "train", batch_size)
    valid_dataloader = get_dataloader(valid_data, "eval", batch_size)
    
    # create model
    model = DeepRMSD(rate=rate).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_function = nn.MSELoss()

    # training
    if os.path.exists('logfile'):
        os.remove('logfile')

    valid_min_loss = []
    last_epoch_number = 0

    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for epoch in range(EPOCH):
        # training
        model.train()
        for step, (x, y) in enumerate(train_dataloader):
            b_x = x.to(device)
            b_y = y.to(device)

            train_pred = model(b_x).reshape(-1)
            train_loss = loss_function(train_pred, b_y.reshape(-1))

            train_PCC = PCC(b_y.reshape(-1), train_pred).cpu().data.numpy()
            train_RMSE = RMSE(b_y.reshape(-1), train_pred).cpu().data.numpy()
            
            optimizer.zero_grad()  # clear gradients for this training step
            train_loss.backward()  # backpropagation, compute gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()

            if step % 500 == 0:
                print('Epoch: ', epoch, '| train_loss: %.4f' % train_loss, '| train_PCC: %.4f' % train_PCC, '| train_RMSE: %.4f' % train_RMSE)
        
        # Validation
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            valid_PCC = 0
            valid_RMSE = 0
            
            for step, (x, y) in enumerate(valid_dataloader):
                b_x = x.to(device)
                b_y = y.to(device)

                valid_pred = model(b_x).reshape(-1)
                valid_loss += loss_function(valid_pred, b_y.reshape(-1))
                valid_PCC += PCC(valid_pred, b_y.reshape(-1))
                valid_RMSE += RMSE(valid_pred, b_y.reshape(-1))

            valid_loss = valid_loss / (step + 1)
            valid_PCC = valid_PCC / (step + 1)
            valid_RMSE = valid_RMSE / (step + 1)

            print('Epoch: ', epoch, '| valid_loss: %.4f' % valid_loss, '| valid_PCC: %.4f' % valid_PCC, '| valid_RMSE: %.4f' % valid_RMSE)

            with open('logfile', 'a') as f:
                if epoch == 0:
                    f.writelines("epoch,train_PCC,train_RMSE,train_loss,valid_PCC,valid_RMSE,valid_loss\n")
                line = "{},{},{},{},{},{},{}".format(epoch, train_PCC, train_RMSE, train_loss, valid_PCC, valid_RMSE, valid_loss)
                f.writelines(line + '\n')
                
            # EarlyStopping
            if epoch == 0:
                valid_min_loss.append(valid_loss)
                torch.save(model, "bestmodel.pth")
                print("Epoch {}: val_loss improved from inf to {}, saving model to bestmodel.pth".format(epoch, valid_loss))
            else:
                if valid_loss < valid_min_loss[0] - min_delta:
                    torch.save(model, "bestmodel.pth")
                    print("Epoch {}: val_loss improved from {} to {}, saving model to bestmodel.pth".format(epoch, valid_min_loss[0], valid_loss))
                    valid_min_loss[0] = valid_loss
                    last_epoch_number = 0
                else:
                    last_epoch_number += 1
                    print("Epoch {}: val_loss did not improve from {}".format(epoch, valid_min_loss[0]))

                if last_epoch_number == patience:
                    print("EarlyStopping!")
                    break

    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('time_running.dat', 'w') as f:
        f.writelines('Start Time:  ' + start_time + '\n')
        f.writelines('End Time:  ' + end_time)
