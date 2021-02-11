import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
import numpy as np

from id_digital_tampering_model import id_digital_tampering_model

from DataSetTXT import Dataset

def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn

if __name__ == '__main__':

    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available. Training on CPU ...')
    else:
        print('CUDA is available! Training on GPU ...')

    num_workers = 0
    batch_size = 20

    train_transform = transforms.Compose([
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        ])

    val_transform = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        ])
    train_data = Dataset(csv_file='C:/Diploma/Train.csv', root_dir='', transform=train_transform)
    val_data = Dataset(csv_file='C:/Diploma/Validation.csv', root_dir='', transform=val_transform)

    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                            shuffle=True, num_workers=3) #Будет тормозить, поставить поменьше
    val_dataloader = DataLoader(val_data, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    #model = MobileNetV2(n_class=2)
    #model.load_state_dict(torch.load('model_cifar.pt'))
    #print(model)
    model = id_digital_tampering_model(dropout_rate=0.4, width_mult=1.)
    params_w_bn, params_wo_bn = separate_bn_paras(model)
    #=============================================================================#
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss()   #кросс-энтропия
    #criterion = nn.KLDivLoss() #Кульбака-Лейблера
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.SGD([{'params': params_wo_bn, 'weight_decay': 1e-3}, {'params': params_w_bn}], lr=0.01, momentum = 0.95)

    n_epochs = 1 ######?????????????????/
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for data, target in train_dataloader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        model.eval()
        for data, target in val_dataloader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(train_dataloader.dataset)
        valid_loss = valid_loss / len(val_dataloader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'model_cifar.pt') ####!!!!!!!TO DO: make good model name
            valid_loss_min = valid_loss
