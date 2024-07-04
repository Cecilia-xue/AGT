import torch
import torch.nn.functional as F
from timm.utils import accuracy

def train_one_epoch(model, criterion, lr_schedule_values, start_steps, train_loader, optimizer, device, lr_scale=0.1):

    model.train(True)
    correct_num = 0
    sample_num = 0
    avg_loss = 0
    for iter_step, (samples, labels) in enumerate(train_loader):
        step=iter_step
        it = start_steps + step
        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                cur_lr_scale = 1.0 if i==0 else lr_scale
                param_group["lr"] = lr_schedule_values[it]*cur_lr_scale

        samples = samples.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(samples)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        correct_num += output.data.max(1)[1].eq(labels).cpu().numpy().sum()
        sample_num += samples.size()[0]
        avg_loss += loss

    acc = round(correct_num/sample_num*100, 2)

    return model, loss/(iter_step+1), acc



@torch.no_grad()
def evaluate(model, criterion, val_loader, device):
    model.eval()
    correct = 0
    sample_num = 0
    avg_loss = 0

    for iter_step, (samples, labels) in enumerate(val_loader):
        samples = samples.to(device)
        labels = labels.to(device)
        output = model(samples)
        correct += output.data.max(1)[1].eq(labels).cpu().numpy().sum()
        sample_num += samples.size()[0]
        avg_loss += criterion(output, labels)
    acc = round(correct/sample_num*100, 2)

    return sample_num, avg_loss/(iter_step+1), acc


# def evaluate(model, criterion, val_loader, device):
#     model.eval()
#     correct = 0
#     sample_num = 0
#     avg_loss = 0
#     with torch.no_grad():
#         for iter_step, (samples, labels) in enumerate(val_loader):
#             samples = samples.to(device)
#             labels = labels.to(device)
#             output = model(samples)
#             correct += output.data.max(1)[1].eq(labels).cpu().numpy().sum()
#             sample_num += samples.size()[0]
#             avg_loss += criterion(output, labels)
#     acc = round(correct/sample_num*100, 2)
#     return sample_num, avg_loss/(iter_step+1), acc

