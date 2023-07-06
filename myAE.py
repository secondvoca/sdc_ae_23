import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from tqdm import tqdm
from celluloid import Camera

from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns

import datetime


class Manager:
    def prepare_data(self, less_than=10, batch_size=128, shuffle=True):
        training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        training_data.data = training_data.data[training_data.targets < less_than]
        training_data.targets = training_data.targets[training_data.targets < less_than]

        self.training_data = training_data.data.unsqueeze(dim=1) / 255.0
        self.training_targets = training_data.targets

        self.training_data_length = len(training_data.data)

        self.train_dataloader = DataLoader(
            training_data, batch_size=batch_size, shuffle=shuffle
        )

    def set_model(self, encoder, decoder):
        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("encoder", encoder),
                    ("decoder", decoder),
                ]
            )
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def print_the_number_of_parameters(self, trainalbe=True):
        if trainalbe:
            count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"There are {count:,d} trainable parameters.")
        else:
            count = sum(p.numel() for p in self.model.parameters())
            print(f"There are {count:,d} parameters.")

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

    def save_current_model(self, name):
        torch.save(
            self.model,
            f'./models/{datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")}_{name}.pt',
        )

    def load_model(self, name):
        self.model = torch.load(f"./models/{name}.pt")

    def run(self, model, dataloader, optimizer, device, calc_loss):
        hist = torch.zeros(len(dataloader))

        for batch, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            # Compute prediction error
            loss = calc_loss(model, x, y, F, device=device)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            hist[batch] = loss.item()

        return hist

    def train(self, calc_loss, epochs=5):
        try:
            device = self.get_cuda_device_or_cpu()
        except:
            device = "cpu"
        print(f"Now, it is working on {device}.")

        self.model.to(device)
        self.model.train()

        hist = torch.zeros(0)

        for _ in tqdm(range(epochs)):
            tmp = self.run(
                self.model, self.train_dataloader, self.optimizer, device, calc_loss
            )
            hist = torch.cat([hist, tmp])

        return hist

    def run_with_record(
        self,
        model,
        dataloader,
        optimizer,
        device,
        calc_loss,
        encode_fn,
        record_step_ratio,
        data_ratio,
    ):
        hist = torch.zeros(len(dataloader))
        record = None

        record_step = int(len(dataloader) * record_step_ratio)

        count = 0
        data_index_to = int(self.training_data_length * data_ratio)

        for batch, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            # Compute prediction error
            loss = calc_loss(model, x, y, F, device=device)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            hist[batch] = loss.item()

            count += 1

            if count >= record_step:
                with torch.no_grad():
                    tmp_y = self.training_targets[:data_index_to].to(device)
                    zs = encode_fn(
                        model,
                        self.training_data[:data_index_to].to(device),
                        tmp_y,
                    )
                    z = torch.cat([zs[0], tmp_y.unsqueeze(1)], dim=1).cpu().unsqueeze(0)

                    if record is not None:
                        record = torch.cat([record, z])
                    else:
                        record = z
                count = 0

        return hist, record

    def train_with_record(
        self,
        calc_loss,
        encode_fn,
        record_step_ratio=0.1,
        data_ratio=0.2,
        epochs=5,
    ):
        try:
            device = self.get_cuda_device_or_cpu()
        except:
            device = "cpu"
        print(f"Now, it is working on {device}.")

        self.model.to(device)
        self.model.train()

        hist = torch.zeros(0)
        record = None

        for _ in tqdm(range(epochs)):
            tmp, tmp_r = self.run_with_record(
                self.model,
                self.train_dataloader,
                self.optimizer,
                device,
                calc_loss,
                encode_fn,
                record_step_ratio,
                data_ratio,
            )
            hist = torch.cat([hist, tmp])
            if record is not None:
                record = torch.cat([record, tmp_r])
            else:
                record = tmp_r

        return hist, record

    def show_latent_space(
        self,
        title,
        encode,
        data_ratio=0.2,
        figsize=[9, 9],
        xlim=[-10, 10],
        ylim=[-10, 10],
        rect_start=[-3, -3],
        rect_height=6,
        rect_width=6,
        rect_linewidth=2,
    ):
        self.model.to("cpu")
        self.model.eval()

        loading_count = int(len(self.train_dataloader) * data_ratio)

        df = pd.DataFrame(columns=["x", "y", "label"])

        with torch.no_grad():
            idx = 0
            for x, y in self.train_dataloader:
                zs = encode(self.model, x, y)

                tmp = pd.DataFrame({"x": zs[0][:, 0], "y": zs[0][:, 1], "label": y})
                df = pd.concat([df, tmp], ignore_index=True)

                idx += 1
                if idx > loading_count:
                    break

        fig, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(data=df, x="x", y="y", hue="label", palette="deep", ax=ax)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        rect = patches.Rectangle(
            rect_start,
            rect_width,
            rect_height,
            linewidth=rect_linewidth,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

        plt.title(title)
        plt.show()

    def plot_generated_images(
        self, title, xlim=[-3, 3], xsteps=11, ylim=[-3, 3], ysteps=11, figsize=[9, 9]
    ):
        self.model.to("cpu")
        self.model.eval()

        grid_x, grid_y = torch.meshgrid(
            torch.linspace(*xlim, xsteps), torch.linspace(*ylim, ysteps), indexing="xy"
        )
        points = torch.stack([grid_x, grid_y], dim=2)

        w = 28
        n = len(points)
        img = torch.zeros((n * w, n * w))
        for i, r in enumerate(points):
            with torch.no_grad():
                tmps = self.model.get_submodule("decoder")(r).view([-1, 1, 28, 28])
                for j, tmp in enumerate(tmps):
                    img[
                        (n - 1 - i) * w : (n - 1 - i + 1) * w, j * w : (j + 1) * w
                    ] = tmp[0]
        plt.figure(figsize=figsize)
        plt.axis("off")
        plt.title(title)
        plt.imshow(img)

    def plot_generated_images_for_10_classes(
        self,
        title,
        class_title=False,
        xlim=[-3, 3],
        xsteps=11,
        ylim=[-3, 3],
        ysteps=11,
        figsize=[20, 8.5],
    ):
        self.model.to("cpu")
        self.model.eval()

        grid_x, grid_y = torch.meshgrid(
            torch.linspace(*xlim, xsteps), torch.linspace(*ylim, ysteps), indexing="xy"
        )
        points = torch.stack([grid_x, grid_y], dim=2)

        _, (ax_1, ax_2) = plt.subplots(nrows=2, ncols=5, figsize=figsize)

        decoder = self.model.get_submodule("decoder")

        with torch.no_grad():
            for idx in range(5):
                w = 28
                n = len(points)
                img = torch.zeros((n * w, n * w))
                for i, r in enumerate(points):
                    p = F.one_hot(torch.tensor([idx] * len(r)), 10)
                    tmps = decoder(r, p).view([-1, 1, 28, 28])
                    for j, tmp in enumerate(tmps):
                        img[
                            (n - 1 - i) * w : (n - 1 - i + 1) * w, j * w : (j + 1) * w
                        ] = tmp[0]
                ax_1[idx].axis("off")
                ax_1[idx].imshow(img)
                if class_title:
                    ax_1[idx].title.set_text(idx)

            for idx in range(5, 10):
                w = 28
                n = len(points)
                img = torch.zeros((n * w, n * w))
                for i, r in enumerate(points):
                    p = F.one_hot(torch.tensor([idx] * len(r)), 10)
                    tmps = decoder(r, p).view([-1, 1, 28, 28])
                    for j, tmp in enumerate(tmps):
                        img[
                            (n - 1 - i) * w : (n - 1 - i + 1) * w, j * w : (j + 1) * w
                        ] = tmp[0]
                ax_2[idx - 5].axis("off")
                ax_2[idx - 5].imshow(img)
                if class_title:
                    ax_2[idx - 5].title.set_text(idx)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def make_video_of(
        self,
        file_name,
        record,
        figsize=[9, 9],
        xlim=[-10, 10],
        ylim=[-10, 10],
    ):
        fig, ax = plt.subplots(figsize=figsize)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        camera = Camera(fig)

        for idx in range(len(record)):
            df = pd.DataFrame(
                {
                    "x": record[idx][:, 0],
                    "y": record[idx][:, 1],
                    "label": record[idx][:, 2],
                }
            )

            sns.scatterplot(
                data=df, x="x", y="y", hue="label", palette="deep", ax=ax, legend=False
            )

            camera.snap()

        animation = camera.animate()
        animation.save(
            f'./video/{datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")}_{file_name}.mp4'
        )
