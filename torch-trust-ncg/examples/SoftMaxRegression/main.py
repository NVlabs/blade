# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling BLADE or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import torch

from torch.autograd import Variable

import matplotlib.pyplot as plt

import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torchtrustncg import TrustRegion


class SoftMaxRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SoftMaxRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


def acc(model, data_loader, device):
    with torch.no_grad():
        correct = 0
        total = 0
        for samples, labels in data_loader:
            #######################
            #  USE GPU FOR MODEL  #
            #######################
            samples = Variable(samples.view(-1, 28 * 28)).to(device)
            labels = labels.to(dtype=torch.float32).to(device)
            outputs = model(samples)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            #######################
            #  USE GPU FOR MODEL  #
            #######################
            # Total correct predictions
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.cpu()).sum()
            else:
                correct += (predicted == labels).sum()

        accuracy = 100 * correct / total
        return accuracy


def main():
    #######################
    #  USE GPU FOR MODEL  #
    #######################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'  # Fixed in this case
    # print('Device to run: {}'.format(device))

    model = SoftMaxRegression(input_dim, output_dim)
    model.to(device)

    # computes softmax and then the cross entropy
    criterion = torch.nn.CrossEntropyLoss()

    opt_method = 'krylov'
    optimizer = TrustRegion(
        model.parameters(), max_trust_radius=1000, initial_trust_radius=.005,
        eta=0.15, kappa_easy=0.01, max_newton_iter=150, max_krylov_dim=150,
        lanczos_tol=1e-5, gtol=1e-05, hutchinson_approx=True,
        opt_method=opt_method)

    tr_losses = []
    tr_accuracies, tst_accuracies = [], []
    n_iter = 0
    best_acc = 0.0
    n_runs = 1  # For mean efficiency
    for r in range(n_runs):
        for n_epoch in range(int(epochs)):
            running_loss = 0
            running_samples = 0
            for i, (samples, labels) in enumerate(train_loader):
                samples = Variable(samples.view(-1, 28 * 28),
                                   requires_grad=False).to(device)
                labels = Variable(labels, requires_grad=False).to(device)

                def closure(backward=True):
                    if torch.is_grad_enabled() and backward:
                        optimizer.zero_grad()
                    model_outputs = model(samples)
                    cri_loss = criterion(model_outputs, labels)
                    if cri_loss.requires_grad and backward:
                        cri_loss.backward(retain_graph=True, create_graph=True)
                    return cri_loss

                tr_loss = optimizer.step(closure=closure)

                batch_loss = tr_loss.detach().cpu()
                running_loss += batch_loss * train_loader.batch_size
                running_samples += train_loader.batch_size

                n_iter += 1
                # if n_iter % 500 == 0:
                #     _tst_acc = acc(model, test_loader, device)
                #     print("n_iteration: {}. Tr. Loss: {}. Tst. Accuracy: {}.".format(n_iter,
                #                                                                      running_loss / running_samples,
                #                                                                      _tst_acc))

            tr_acc = acc(model, train_loader, device)
            tst_acc = acc(model, test_loader, device)

            print("n_iteration: {} - n_epoch {} - Tr. Loss: {} - Tr. Accuracy: {} - Tst. Accuracy: {}".
                  format(n_iter, n_epoch + 1, running_loss / running_samples, tr_acc, tst_acc))

            if tst_acc > best_acc:
                best_acc = tst_acc

            tr_losses.append(running_loss / len(train_loader.sampler))
            tr_accuracies.append(tr_acc)
            tst_accuracies.append(tst_acc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(tr_losses, 'o-', label='Train Loss')
    ax1.title.set_text('Loss using {}'.format(opt_method))
    ax1.set_yscale('log')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('log Loss')
    ax1.legend()
    ax1.grid()

    ax2.plot(tr_accuracies, 'o-', label='Train acc.')
    ax2.plot(tst_accuracies, 'o-', label='Test acc.')
    ax2.title.set_text('Acc using {}'.format(opt_method))
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid()

    plt.savefig('Results/{}.png'.format(opt_method))
    plt.show()


if __name__ == "__main__":
    batch_size = 512
    epochs = 15
    input_dim = 784
    output_dim = 10
    lr_rate = 0.01

    train_dataset = dsets.MNIST(
        root='../../Datasets', train=True, transform=transforms.ToTensor(),
        download=True)
    test_dataset = dsets.MNIST(
        root='../../Datasets', train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False)

    main()
