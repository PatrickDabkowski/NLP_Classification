import torch.nn

# most basic MLP for classification purpose
class MLP(torch.nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, input_size // 2)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(input_size // 2, input_size // 4)
        self.fc3 = torch.nn.Linear(input_size // 4, input_size // 8)
        self.fc3 = torch.nn.Linear(input_size // 8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)

        return x

if __name__ != "__main__":
    print(f"file: {__name__}")