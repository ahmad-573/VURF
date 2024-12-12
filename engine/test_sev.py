import torch
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    x = torch.rand(100000,10000).to(device)
    y = torch.rand(10000,10000).to(device)
    while(1):
        z = x @ y.T

