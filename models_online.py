import pandas as pd
import numpy as np
from sklearn import svm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import csv

def write_csv(fileName,data):
    with open(fileName, 'a') as outfile:
        writer = csv.writer(outfile,  lineterminator='\n')
        writer.writerow(data)

# input is imgStr 10*10 np.array
def SVM_load():
    file_path = 'GasArrayStm32V1_1/dataset/data.csv'
    sheet_name = 'raw data'
    C = 10000  # SVM regularization parameter, 1;
    gamma = 0.065 # 0.7
    ####################################################

    # gas = pd.read_excel(file_path, sheet_name=sheet_name)
    data = np.genfromtxt(file_path, delimiter=',') # delimiter=','
    x = data[1:, 4:104]
    y = np.ravel(data[1:, 1])
    print(x,y)
    # x = StandardScaler().fit_transform(x)
    clf = svm.SVC(kernel='rbf', gamma=gamma, C=C, decision_function_shape='ovo')
    models = clf.fit(x, y)
    return models

def SVM_pre(models, gas_img):
    pred = models.predict(np.array(gas_img).reshape(1, -1))
    # print(models.predict(X[3, :].reshape(1, -1)))
    return pred

def CNN_load(dataset=True):
    if dataset == True:
        file_path = 'dataset/data.csv'
        sheet_name = 'raw data'
        data = np.genfromtxt(file_path, delimiter=',') # delimiter=','
        y = np.ravel(data[1:, 1])
        target = 50
        target_num = y[-1] + 1
        model_path = "dataset/model20211111_104758.pth"
    else:
        target = 100
        target_num = 24
        model_path = "dataset/model20211027odor10.pth"
    ####################################################

    class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=2)
          self.conv2 = nn.Conv2d(10, 20, kernel_size=3, padding=2)
          self.conv2_drop = nn.Dropout2d()
          self.fc1 = nn.Linear(320, target)
          self.fc2 = nn.Linear(target, target_num)

      def forward(self, x):
          x = F.relu(F.max_pool2d(self.conv1(x), 2))
          x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
          x = x.view(-1, 320)
          x = F.relu(self.fc1(x))
          x = F.dropout(x, training=self.training)
          x = self.fc2(x)
          return F.log_softmax(x)
    net = Net()
    # print(net)
    net.load_state_dict(torch.load(model_path)) # model20211111_104758
    return net

def CNN_pre(models, gas_img):
    gas_img = Variable(torch.from_numpy(np.array(gas_img).reshape((1,1,10,10))))
    output = models(gas_img.float()) 
    # print(output)
    pred = output.data.max(1, keepdim=True)[1]
    return pred
    
# # ### debug part ###########################
# file_path = 'dataset/data.xlsx'
# sheet_name = 'raw data'
# gas = pd.read_excel(file_path, sheet_name=sheet_name)
# x = gas.iloc[:, 1:101].values
# y = np.ravel(gas.iloc[:, 101].values)
# # print(np.array(x[3,:]).shape)
# models = CNN_load()
# output = CNN_pre(models, x[203,:])
# if output == 0:
#     print("ssss")
# print(output)
# # ##########################################