import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

train = datasets.CIFAR10(root = './data', train = True, download = True, transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.CIFAR10(root = './data', train = False, download = True, transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
testset = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = False)

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 16, 4)
		self.pool = nn.MaxPool2d(2, 2)
		self.drop = nn.Dropout(p = 0.25)
		self.norm1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, 4)
		self.norm2 = nn.BatchNorm2d(32)
		self.fc1 = nn.Linear(32*5*5, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 10)

	def forward(self, x):
		#x = self.drop(self.pool(F.relu(self.conv1(x))))
		#x = self.drop(self.pool(F.relu(self.conv2(x))))
		x = self.norm1(self.pool(F.relu(self.conv1(x))))
		x = self.norm2(self.pool(F.relu(self.conv2(x))))
		#x = self.norm(self.drop(self.pool(F.relu(self.conv1(x)))))
		#x = self.norm(self.drop(self.pool(F.relu(self.conv2(x)))))
		#x = self.pool(F.relu(self.conv1(x)))
		#x = self.pool(F.relu(self.conv2(x)))
		x = torch.flatten(x, 1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.log_softmax(x, dim = 1)

net = Net()
#print(net)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

for epoch in range(10):
	correct = 0
	total = 0
	for data in trainset:
		X, y = data
		net.zero_grad()
		output = net(X)
		loss = F.nll_loss(output, y)
		loss.backward()
		optimizer.step()
		for idx, i in enumerate(output):
			if torch.argmax(i) == y[idx]:
				correct += 1
			total += 1
	print(loss, "Train accuracy: ", round(correct/total, 3))

correct = 0
total = 0

with torch.no_grad():
	for data in testset:
		X, y = data
		output = net(X)
		#print(output)
		for idx, i in enumerate(output):
			if torch.argmax(i) == y[idx]:
				correct += 1
			total += 1

print("Test accuracy: ", round(correct/total, 3))