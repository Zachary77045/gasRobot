# classify 8 gases in RH

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.utils.data as Data
import pandas as pd
import os, time
import numpy as np

def load_data(file_path):
    # 8 gases in dry air and humidity
    gas = pd.read_excel(file_path, sheet_name='raw data')
    x = gas.iloc[:, 4:104].values
    y = np.ravel(gas.iloc[:, 1].values)
    return x,y

def main(n_epochs):
  batch_size_train = 32
  learning_rate = 0.001
  momentum = 0.5
  log_interval = 32
  train_test_split_ratio = 0.8
  batch_size_test = 100
  # target_num = 114
  timestr = time.strftime("%Y%m%d_%H%M%S")

  random_seed = 1
  torch.backends.cudnn.enabled = False
  torch.manual_seed(random_seed)
  ######### parameters to input ######################
  file_path = 'dataset/data.xlsx'
  x,y = load_data(file_path=file_path)
  target_num = y[-1] + 1
  print("total gas species:", target_num)
  data_length = len(x)
  x = x.reshape((data_length,1,10,10))
  # torch can only train on Variable, so convert them to Variable
  x, y = Variable(torch.from_numpy(x)), Variable(torch.from_numpy(y))
  print("x.shape, y.shape: ", x.shape, y.shape)

  torch_dataset = Data.TensorDataset(x, y)
  train_dataset, test_dataset = torch.utils.data.random_split(torch_dataset, \
      (int(data_length*train_test_split_ratio), data_length-int(data_length*train_test_split_ratio)))
  train_loader = Data.DataLoader(
      dataset=train_dataset, 
      batch_size=batch_size_train, 
      shuffle=True, num_workers=0,)
  test_loader = Data.DataLoader(
      dataset=test_dataset, 
      batch_size=batch_size_test, 
      shuffle=True, num_workers=0,)

  class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=2)
          self.conv2 = nn.Conv2d(10, 20, kernel_size=3, padding=2)
          self.conv2_drop = nn.Dropout2d()
          self.fc1 = nn.Linear(320, 50)
          self.fc2 = nn.Linear(50, target_num)

      def forward(self, x):
          x = F.relu(F.max_pool2d(self.conv1(x), 2))
          x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
          x = x.view(-1, 320)
          x = F.relu(self.fc1(x))
          x = F.dropout(x, training=self.training)
          x = self.fc2(x)
          return F.log_softmax(x,dim=1)

  network = Net()
  optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                        momentum=momentum)

  train_losses = []
  train_counter = []
  test_losses = []
  test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
  accuracy = []

  def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      # print(data.shape)
      output = network(data.float())
      # print(output.shape, target.long().shape)
      loss = F.nll_loss(output, target.long())
      loss.backward()
      optimizer.step()
      if batch_idx % log_interval == 0:
        train_losses.append(loss.item())
        train_counter.append((batch_idx*32) + ((epoch-1)*len(train_loader.dataset)))
    if epoch%10 == 0:
      print('Train Epoch: {} [len(data): {}]\tLoss: {:.6f}'.format(epoch, len(train_loader.dataset), loss.item()))
      torch.save(network.state_dict(), 'dataset/model'+timestr+'.pth')
      torch.save(optimizer.state_dict(), 'dataset/optimizer'+timestr+'.pth')

  def test(epoch):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in test_loader:
          # print(target)
          output = network(data.float())
          test_loss += F.nll_loss(output, target.long(), size_average=False).item()
          pred = output.data.max(1, keepdim=True)[1]
          correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    accuracy.append(100. * correct / len(test_loader.dataset))
    if epoch%10 == 0:
      print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

  test(0)
  for epoch in range(1, n_epochs + 1):
    train(epoch)
    acc = test(epoch)

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(train_counter, train_losses, color='blue', label='Train Loss')
  ax.scatter(test_counter, test_losses, color='red', label='Test Loss')
  ax0 = ax.twinx()
  ax0.set_ylabel('accuracy (%)', fontsize=20)
  ax0.plot(test_counter, accuracy, color='red', label = 'accuracy')
  ax.set_xlabel('number of training examples seen', fontsize=20)
  ax.set_ylabel('negative log likelihood loss', fontsize=20)
  fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)

  plt.show()

  # train_losses = pd.DataFrame(train_losses)
  # test_losses = pd.DataFrame(test_losses)
  # accuracy = pd.DataFrame(accuracy)
  # with pd.ExcelWriter('E:\data\Vinson\mixed gas-toluene&acetone/cnn_output_allcs%d.xlsx'%chop_size) as writer:  
  #     train_losses.to_excel(writer,sheet_name='train_losses')
  #     test_losses.to_excel(writer,sheet_name='test_losses')
  #     accuracy.to_excel(writer,sheet_name='accuracy')

main(n_epochs=100)