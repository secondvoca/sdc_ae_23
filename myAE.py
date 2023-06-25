import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from tqdm import tqdm

from collections import OrderedDict


class My_Encoder(nn.Module):
    def __init__(self, dim_encoder_output, activation):
        super().__init__()
        if activation == "tanh":
            self.activation = torch.tanh
        self.l0 = nn.Linear(784, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, dim_encoder_output)

    def forward(self, x):
        z = self.activation(self.l0(x))
        z = self.activation(self.l1(z))
        z = self.l2(z)
        return z


class My_Decoder(nn.Module):
    def __init__(self, dim_decoder_input, activation):
        super().__init__()
        if activation == "tanh":
            self.activation = torch.tanh
        self.l0 = nn.Linear(dim_decoder_input, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 784)

    def forward(self, x):
        z = self.activation(self.l0(x))
        z = self.activation(self.l1(z))
        z = torch.sigmoid(self.l2(z))
        return z


class SDC_AE:
    def __init__(self, kind="ae", activation="tanh"):
        super().__init__()
        self.kind = kind
        self.activation = "tanh"

    def prepare_data(self, less_than=10, batch_size=128, shuffle=True):
        # Download training data from open datasets.
        training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        training_data.data = training_data.data[training_data.targets < less_than]
        training_data.targets = training_data.targets[training_data.targets < less_than]

        # Create data loaders.
        self.train_dataloader = DataLoader(
            training_data, batch_size=batch_size, shuffle=shuffle
        )

    def prepare_model(self, dim_encoder_output, dim_decoder_input, activation="tanh"):
        if self.kind == "ae":
            encoder = My_Encoder(
                dim_encoder_output=dim_encoder_output, activation=activation
            )
            decoder = My_Decoder(
                dim_decoder_input=dim_decoder_input, activation=activation
            )
            self.model = nn.Sequential(
                OrderedDict(
                    [
                        ("encoder", encoder),
                        ("decoder", decoder),
                    ]
                )
            )
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def get_cuda_device_or_cpu(self):
        if torch.cuda.is_available():
            cuda_count = torch.cuda.device_count()

            no = 0
            mem_available = 0

            for i in range(cuda_count):
                tmp_available = torch.cuda.mem_get_info(i)[0]
                if mem_available < tmp_available:
                    no = i
                    mem_available = tmp_available
            return f"cuda:{no}"
        return "cpu"

    def run(self, model, dataloader, optimizer, device, calc_loss):
        hist = torch.zeros(len(dataloader))

        for batch, (x, y) in enumerate(dataloader):
            x = x.view([-1, 28 * 28]).to(device)

            # Compute prediction error
            loss = calc_loss(model, x, F)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            hist[batch] = loss.item()

        return hist

    def train(self, calc_loss, epochs=5):
        device = self.get_cuda_device_or_cpu()
        print(device)

        self.model.to(device)
        self.model.train()

        hist = torch.zeros(0)

        for _ in tqdm(range(epochs)):
            tmp = self.run(self.model, self.train_dataloader, self.optimizer, device, calc_loss)
            hist = torch.cat([hist, tmp])

        return hist
