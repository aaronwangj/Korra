import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
from preprocess import tokenize, stem, bag_of_words
from model import ffNeuralNet

with open('data.json', 'r',) as x:
    data = json.load(x)

words_list = []
tags = []
patterns = []

for points in data['data']:
    topic = points['topic']
    tags.append(topic)
    for pattern in points['patterns']:
        word = tokenize(pattern)
        words_list.extend(word)
        patterns.append((word, topic))

remove_punctuation = ['.',',','?','!',':',';']

words_list = sorted(set([stem(w) for w in words_list if w not in remove_punctuation]))
tags = sorted(set(tags))

trainx = []
trainy = []

for (p, t) in patterns:
    bag = bag_of_words(p, words_list)
    trainx.append(bag)

    label = tags.index(t)
    trainy.append(label) #CrossEntropyLoss

trainx = np.array(trainx)
trainy = np.array(trainy)

#HyperParameters
batch_size=32
hidden_size = 10
output_size = len(tags)
input_size = len(trainx[0])
learning_rate = 0.001
num_epochs = 500

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(trainx)
        self.x_data = trainx
        self.y_data = trainy
    
    def __len__(self):
        return self.n_samples        

    def __getitem__(self, index):
        return self.x_data[index],  self.y_data[index]


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ffNeuralNet(input_size, hidden_size, output_size).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for e in range(num_epochs):
    for (w, l) in train_loader:
        w = w.to(device)
        l = l.to(device=device, dtype=torch.int64)

        #forward
        outputs = model(w)
        loss = criterion(outputs, l)

        #backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (e+1) % 5 == 0:
        print(f'Loss= {loss.item():.4f} [{int(100*(e+1)/num_epochs)}%] ')

print(f'Final loss: {loss.item():.4f}')
    
datafinal = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "words_list": words_list,
    "tags": tags
}

FILE = "datafinal.pth"
torch.save(datafinal, FILE)
print(f'The training complete! Your file is saved as {FILE}')