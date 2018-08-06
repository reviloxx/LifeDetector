import torch
import torchvision
from torchvision import transforms
from PIL import Image
from os import listdir
import random
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import os
from PIL import ImageFont
from PIL import ImageDraw

batch_size = 256
use_cuda = True

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    normalize])

train_data = []


# TARGET: [valid, fraud]
def load_training_data():
    train_data_list = []
    target_list = []

    files = listdir('data/train/')
    for i in range(len(listdir('data/train/'))):
        f = random.choice(files)
        files.remove(f)
        img = Image.open("data/train/" + f)
        img_tensor = transform(img)
        train_data_list.append(img_tensor)

        valid = 1 if 'valid' in f else 0
        fraud = 1 if 'fraud' in f else 0
        target = [valid, fraud]
        target_list.append(target)
        if len(train_data_list) >= batch_size:
            train_data.append((torch.stack(train_data_list), target_list))
            train_data_list = []
            target_list = []
            print('Loaded batch', len(train_data), 'of ', (int(len(listdir('data/train/')) / batch_size)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3)
        self.conv3 = nn.Conv2d(12, 24, kernel_size=3)
        self.conv4 = nn.Conv2d(24, 48, kernel_size=3)
        self.conv5 = nn.Conv2d(48, 96, kernel_size=3)
        self.conv6 = nn.Conv2d(96, 192, kernel_size=3)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.dropout1(x)
        x = x.view(-1, 768)
        x = F.relu(self.dropout2(self.fc1(x)))
        x = self.fc2(x)
        return F.sigmoid(x)


model = Net()
if use_cuda:
    model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()
    batch_id = 0
    for data, target in train_data:
        target = torch.Tensor(target)
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = F.binary_cross_entropy
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, (batch_id + 1) * len(data), len(train_data) * batch_size,
                   100. * (batch_id + 1) / len(train_data), loss.data.item()))
        batch_id = batch_id + 1


def test():
    model.eval()
    files = listdir('data/test/')

    while True:
        f = random.choice(files)
        img = Image.open('data/test/' + f)
        img_eval_tensor = transform(img)
        img_eval_tensor.unsqueeze_(0)
        data = img_eval_tensor
        if use_cuda:
            data = data.cuda()
        data = Variable(data)
        out = model(data)
        # print(str(f) + ": " + str(out.data.max(1, keepdim=True)[1]))
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        # font = ImageFont.truetype("Roboto-Bold.ttf", 50)
        # draw.text((x, y),"Sample Text",(r,g,b))
        text = "Valid"
        print(out.data.max(1, keepdim=True)[1].cpu().numpy()[0])
        if out.data.max(1, keepdim=True)[1].cpu().numpy()[0] != 0:
            text = "Fraud"
        draw.text((0, 0), text, (255, 255, 255))
        img.show()
        x = input('')


if not os.path.isfile('model.pt'):
    load_training_data()
    for epoch in range(1, 30):
        train(epoch)
        torch.save(model, 'model.pt')
else:
    model = torch.load('model.pt')
    if use_cuda:
        model = model.cuda()
test()
