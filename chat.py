import torch
import json
import random
from model import ffNeuralNet
from preprocess import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('data.json', 'r',) as x:
    data = json.load(x)

FILE = "datafinal.pth"
datafinal = torch.load(FILE)

input_size = datafinal["input_size"]
hidden_size = datafinal["hidden_size"]
output_size = datafinal["output_size"]
words_list = datafinal["words_list"]
model_state = datafinal["model_state"]
tags = datafinal["tags"]

model = ffNeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

name = "Korra"
print("Hi, My name is Korra and I'm here to answer any question you have about COVID-19! Ask away.\n(Type quit to leave)")

while True:
    userInput = input("You: ")
    if userInput == "quit":
        break

    userInput = tokenize(userInput)
    pre = bag_of_words(userInput, words_list)
    pre = pre.reshape(1, pre.shape[0])
    pre = torch.from_numpy(pre).to(device)

    output = model(pre)

    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probability = torch.softmax(output, dim=1)

    p = probability[0][predicted.item()]
    if p.item()>=0.70:
        for point in data["data"]:
            if tag == point["topic"]:
                print(f"{name}: {random.choice(point['responses'])}")
    else:
        print('{}: Hm... I am not sure. Try asking another question!'.format(name))
    
