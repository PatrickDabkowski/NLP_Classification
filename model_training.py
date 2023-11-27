import torch
import load_MLP
import data_load
import llama_load
import tokenizer_load

def train(tokenizer, llm, model, train_loader, criterion, optimizer, device):
    model.train()
    tot_loss = 0

    # process input to token and token by LLM without an update of gradient and weights
    for data in train_loader:

        with torch.no_grad():

            x_token = tokenizer(data[1], return_tensors='pt', padding=True, truncation=True, max_length=512)
            x_out = llm(**x_token)
            print(x_out)

        # train the classifier model
        out = model(x_out.to(device))
        print(out)
        loss = criterion(out, data[0].float().unsqueeze(1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        tot_loss = tot_loss + loss

    return tot_loss

def test(tokenizer, llm, model, loader, device):
    model.eval()
    total_acc = 0
    total_samples = 0

    with torch.no_grad():
        for data in loader:

            x_token = tokenizer(data[1], return_tensors='pt', padding=True, truncation=True, max_length=512)
            x_out = llm(**x_token)

            out = model(x_out.to(device))
            out = torch.sigmoid(out)

            acc = torch.mean((out.to(device).round().squeeze() == data[0].to(device)).float())
            total_acc += acc.item()
            total_samples += 1

    return total_acc / total_samples

def training(tokenizer, llm, model, device, epochs, criterion, optimizer, stop_criterion, train_loader, valid_loader, test_loader):
    print("\nmodel training...\n")
    model.to(device)

    best_acc = 0
    best_model = None
    early_stopping = 0

    for epoch in range(1, epochs):
        loss = train(tokenizer, llm, model, train_loader, criterion, optimizer, device)
        train_acc = test(tokenizer, llm, model, train_loader, device)
        test_acc = test(tokenizer, llm, model, valid_loader, device)
        if test_acc <= best_acc:
            early_stopping = early_stopping + 1
        else:
            early_stopping = 0
            best_acc = test_acc
            best_model = model
        if early_stopping >= stop_criterion:
            print(f'Early stopping in epoch: {epoch}, with accuracy: {best_acc}')
            break
        print(f'Epoch: {epoch}, Train Loss: {loss}, Train Acc: {train_acc}, Valid Acc: {test_acc}')

    test_acc = test(tokenizer, llm, model, test_loader, device)
    print(f'Test Acc: {test_acc}')

    return best_model, test_acc

if __name__ == "__main__":
    # !!! device for Mac
    '''print('GPU: ', torch.backends.mps.is_available())
    print('GPU built: ', torch.backends.mps.is_built())
    device = torch.device("cpu")
    print('device: ', device)'''
    device = "cpu"

    # Data loaders
    train_iter, valid_iter, test_iter = data_load.load()

    # tokenizer and LLM
    tokenizer = tokenizer_load.load()
    tokenizer.pad_token = tokenizer.eos_token

    llama = llama_load.load(info=True) # output 32016

    # binary classification MLP
    model = load_MLP.MLP(32016)

    # training hyperparameters
    o = torch.optim.Adam(model.parameters(), lr=0.001)
    c = torch.nn.BCEWithLogitsLoss()

    training(tokenizer, llama, model, device, 300, c, o, 15, train_iter, valid_iter, test_iter)