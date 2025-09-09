# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="1002" height="817" alt="image" src="https://github.com/user-attachments/assets/08836fbd-3598-4a8d-a31b-892329c5d97a" />

## DESIGN STEPS

### STEP 1
Load and preprocess the dataset (handle missing values, encode categorical features, scale numeric data).

### STEP 2
Split the dataset into training and testing sets, convert to tensors, and create DataLoader objects.

### STEP 3
Build the neural network model, train it with CrossEntropyLoss and Adam optimizer, then evaluate with confusion matrix and classification report.

## PROGRAM
### Name: MONISH N
### Register Number: 212223240097

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,4)


    def forward(self,x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x

```
```python
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)
```
```python
def train_model(model, train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()
      output = model(X_batch)
      loss = criterion(output,y_batch)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
      print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
```

## Dataset Information

<img width="1299" height="273" alt="image" src="https://github.com/user-attachments/assets/a1f2691f-e21e-4477-a134-e607d58f8787" />


## OUTPUT

### Confusion Matrix

<img width="322" height="173" alt="image" src="https://github.com/user-attachments/assets/e232888c-f275-46b5-a2f1-119469af9701" />

<img width="685" height="561" alt="image" src="https://github.com/user-attachments/assets/418abed8-1fce-4d4b-b58e-e9cf25e4b7bf" />


### Classification Report

<img width="703" height="244" alt="image" src="https://github.com/user-attachments/assets/6dbd8553-31b7-404d-9825-7324c096a316" />

### New Sample Data Prediction

<img width="1001" height="353" alt="image" src="https://github.com/user-attachments/assets/cebc8032-6a93-4766-92c2-4a84faacc67e" />


## RESULT
The neural network model was successfully built and trained to handle classification tasks.
