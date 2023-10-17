import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_same_index(target, label):
	label_indices = []
	for i in range(len(target)):
		if target[i][1] == label:
			label_indices.append(target[i])
	return label_indices

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

train = datasets.CIFAR10(root = './data', train = True, download = True, transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.CIFAR10(root = './data', train = False, download = True, transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)

testset = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = False)

test_indices = get_same_index(test, 0)
testplane = torch.utils.data.DataLoader(test_indices, batch_size = 10, shuffle = False)

test_indices = get_same_index(test, 1)
testcar = torch.utils.data.DataLoader(test_indices, batch_size = 10, shuffle = False)

test_indices = get_same_index(test, 2)
testbird= torch.utils.data.DataLoader(test_indices, batch_size = 10, shuffle = False)

test_indices = get_same_index(test, 3)
testcat = torch.utils.data.DataLoader(test_indices, batch_size = 10, shuffle = False)

test_indices = get_same_index(test, 4)
testdeer = torch.utils.data.DataLoader(test_indices, batch_size = 10, shuffle = False)

test_indices = get_same_index(test, 5)
testdog = torch.utils.data.DataLoader(test_indices, batch_size = 10, shuffle = False)

test_indices = get_same_index(test, 6)
testfrog = torch.utils.data.DataLoader(test_indices, batch_size = 10, shuffle = False)

test_indices = get_same_index(test, 7)
testhorse = torch.utils.data.DataLoader(test_indices, batch_size = 10, shuffle = False)

test_indices = get_same_index(test, 8)
testship = torch.utils.data.DataLoader(test_indices, batch_size = 10, shuffle = False)

test_indices = get_same_index(test, 9)
testtruck = torch.utils.data.DataLoader(test_indices, batch_size = 10, shuffle = False)

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

print("Total test accuracy: ", round(correct/total, 3))

correct = 0
total = 0

with torch.no_grad():
	for data in testplane:
		X, y = data
		output = net(X)
		#print(output)
		for idx, i in enumerate(output):
			if torch.argmax(i) == y[idx]:
				correct += 1
			total += 1

print("Plane accuracy: ", round(correct/total, 3))

correct = 0
total = 0

with torch.no_grad():
	for data in testcar:
		X, y = data
		output = net(X)
		#print(output)
		for idx, i in enumerate(output):
			if torch.argmax(i) == y[idx]:
				correct += 1
			total += 1

print("Car accuracy: ", round(correct/total, 3))

correct = 0
total = 0

with torch.no_grad():
	for data in testbird:
		X, y = data
		output = net(X)
		#print(output)
		for idx, i in enumerate(output):
			if torch.argmax(i) == y[idx]:
				correct += 1
			total += 1

print("Bird accuracy: ", round(correct/total, 3))

correct = 0
total = 0

with torch.no_grad():
	for data in testcat:
		X, y = data
		output = net(X)
		#print(output)
		for idx, i in enumerate(output):
			if torch.argmax(i) == y[idx]:
				correct += 1
			total += 1

print("Cat accuracy: ", round(correct/total, 3))

correct = 0
total = 0

with torch.no_grad():
	for data in testdeer:
		X, y = data
		output = net(X)
		#print(output)
		for idx, i in enumerate(output):
			if torch.argmax(i) == y[idx]:
				correct += 1
			total += 1

print("Deer accuracy: ", round(correct/total, 3))

correct = 0
total = 0

with torch.no_grad():
	for data in testdog:
		X, y = data
		output = net(X)
		#print(output)
		for idx, i in enumerate(output):
			if torch.argmax(i) == y[idx]:
				correct += 1
			total += 1

print("Dog accuracy: ", round(correct/total, 3))

correct = 0
total = 0

with torch.no_grad():
	for data in testfrog:
		X, y = data
		output = net(X)
		#print(output)
		for idx, i in enumerate(output):
			if torch.argmax(i) == y[idx]:
				correct += 1
			total += 1

print("Frog accuracy: ", round(correct/total, 3))

correct = 0
total = 0

with torch.no_grad():
	for data in testhorse:
		X, y = data
		output = net(X)
		#print(output)
		for idx, i in enumerate(output):
			if torch.argmax(i) == y[idx]:
				correct += 1
			total += 1

print("Horse accuracy: ", round(correct/total, 3))

correct = 0
total = 0

with torch.no_grad():
	for data in testship:
		X, y = data
		output = net(X)
		#print(output)
		for idx, i in enumerate(output):
			if torch.argmax(i) == y[idx]:
				correct += 1
			total += 1

print("Ship accuracy: ", round(correct/total, 3))

correct = 0
total = 0

with torch.no_grad():
	for data in testtruck:
		X, y = data
		output = net(X)
		#print(output)
		for idx, i in enumerate(output):
			if torch.argmax(i) == y[idx]:
				correct += 1
			total += 1

print("Truck accuracy: ", round(correct/total, 3))