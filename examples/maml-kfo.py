#!/usr/bin/env python3

"""
File: maml-kfo.py
Author: Seb Arnold - seba1511.net
Email: smr.arnold@gmail.com
Github: seba-1511
Description: Implementation of MAML with KFO meta-optimizer.
"""

import random
import numpy as np
import torch
import learn2learn as l2l
import wandb
import typer
import tqdm

app = typer.Typer()


class CIFARCNN(torch.nn.Module):
    """
    Example of a 4-layer CNN network for FC100/CIFAR-FS.
    """

    def __init__(
        self,
        output_size=5,
        hidden_size=32,
        layers=4,
        dataset='cifarfs',
    ):
        super(CIFARCNN, self).__init__()
        self.hidden_size = hidden_size
        features = l2l.vision.models.ConvBase(
            output_size=hidden_size,
            hidden=hidden_size,
            channels=3,
            max_pool=False,
            layers=layers,
            max_pool_factor=0.5,
        )
        self.features = torch.nn.Sequential(
            features,
            l2l.nn.Lambda(lambda x: x.mean(dim=[2, 3])),
            l2l.nn.Flatten(),
        )
        self.classifier = torch.nn.Linear(self.hidden_size, output_size, bias=True)
        l2l.vision.models.maml_init_(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(
        batch,
        model,
        update,
        diff_sgd,
        loss,
        adaptation_steps,
        shots,
        ways,
        device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model & learned update
    for step in range(adaptation_steps):
        adaptation_error = loss(model(adaptation_data), adaptation_labels)
        if step > 0:  # Update the learnable update function
            update_grad = torch.autograd.grad(adaptation_error,
                                              update.parameters(),
                                              create_graph=True,
                                              retain_graph=True)
            diff_sgd(update, update_grad)
        model_updates = update(adaptation_error,
                               model.parameters(),
                               create_graph=True,
                               retain_graph=True)
        diff_sgd(model, model_updates)

    # Evaluate the adapted model
    predictions = model(evaluation_data)
    eval_error = loss(predictions, evaluation_labels)
    eval_accuracy = accuracy(predictions, evaluation_labels)
    return eval_error, eval_accuracy


@app.command()
def main(
    fast_lr: float = 0.1,
    meta_lr: float = 0.003,
    num_iterations: int = 40000,
    meta_batch_size: int = 16,
    adaptation_steps: int = 5,
    dataset: str = 'cifarfs',
    layers: int = 4,
    shots: int = 5,
    ways: int = 5,
    cuda: int = 1,  # 0 or 1 only
    seed: int = 1234,
):
    args = dict(locals())
    wandb.init(
        project='kfo',
        group='MAML',
        name='maml-' + dataset,
        config=args,
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    # Create Tasksets using the benchmark interface
    tasksets = l2l.vision.benchmarks.get_tasksets(
        name=dataset,
        train_samples=2*shots,
        train_ways=ways,
        test_samples=2*shots,
        test_ways=ways,
        root='~/data',
    )

    # Create model and learnable update
    if dataset == 'cifarfs':
        model = CIFARCNN(output_size=ways, hidden_size=32, layers=layers)
    elif dataset == 'mini-imagenet':
        model = l2l.vision.models.CNN4(
            output_size=ways,
            hidden_size=32,
            layers=layers,
        )
    model.to(device)
    kfo_transform = l2l.optim.transforms.KroneckerTransform(
        kronecker_cls=l2l.nn.KroneckerLinear,
        psd=True,
    )
    fast_update = l2l.optim.ParameterUpdate(
        parameters=model.parameters(),
        transform=kfo_transform,
    )
    fast_update.to(device)
    diff_sgd = l2l.optim.DifferentiableSGD(lr=fast_lr)

    all_parameters = list(model.parameters()) + list(fast_update.parameters())
    opt = torch.optim.Adam(all_parameters, meta_lr)
    loss = torch.nn.CrossEntropyLoss(reduction='mean')

    for iteration in tqdm.trange(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            task_model = l2l.clone_module(model)
            task_update = l2l.clone_module(fast_update)
            batch = tasksets.train.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch,
                task_model,
                task_update,
                diff_sgd,
                loss,
                adaptation_steps,
                shots,
                ways,
                device,
            )
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            task_model = l2l.clone_module(model)
            task_update = l2l.clone_module(fast_update)
            batch = tasksets.validation.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch,
                task_model,
                task_update,
                diff_sgd,
                loss,
                adaptation_steps,
                shots,
                ways,
                device,
            )
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        # log some metrics
        wandb.log({
            'Train Error': meta_train_error / meta_batch_size,
            'Train Accuracy': meta_train_accuracy / meta_batch_size,
            'Validation Error': meta_valid_error / meta_batch_size,
            'Validation Accuracy': meta_valid_accuracy / meta_batch_size,
        }, step=iteration)

        # Average the accumulated gradients and optimize
        for p in model.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        for p in fast_update.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-testing loss
            task_model = l2l.clone_module(model)
            task_update = l2l.clone_module(fast_update)
            batch = tasksets.test.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch,
                task_model,
                task_update,
                diff_sgd,
                loss,
                adaptation_steps,
                shots,
                ways,
                device,
            )
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()
        wandb.log({
            'Test Error':  meta_test_error / meta_batch_size,
            'Test Accuracy': meta_test_accuracy / meta_batch_size,
        }, step=iteration)


if __name__ == '__main__':
    app()
