import torch
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
# import torchvision
# from vit_pytorch import SimpleViT
from model.M2ORT import M2ORT
from utils import TestModel
import argparse

parser = argparse.ArgumentParser(description='abc')

parser.add_argument('--variant', default='base', type=str)
parser.add_argument('--dataset', default='HBC', type=str)
parser.add_argument('--dataset_path', default=None, type=str)
parser.add_argument('--EPOCH', default=200, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--dropout', default='0.1', type=float)
parser.add_argument('--lr', default=1e-4, type=float)
# parser.add_argument('--weight_decay', default=1e-4, type=float)
# parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--num_cls', default=250, type=int)
parser.add_argument('--selected_genes', default=None, type=str)
parser.add_argument('--data_parallel', default=True, type=bool)
parser.add_argument('--checkpoint', default=None, type=str)
parser.add_argument('--batch_size', default=96, type=int)
params = parser.parse_args()
print('Training M2ORT with the following configuration: \n',params,'\n===================')

selected_genes=[]
if params.selected_genes==None:
    if params.dataset=='HBC':
        selected_genes=np.load('./dataset/HBC_Selected_Genes.npy',allow_pickle=True).tolist()
    elif params.dataset=='HER2+':
        selected_genes=np.load('./dataset/HER2_Selected_Genes.npy',allow_pickle=True).tolist()
    elif params.dataset=='cSCC':
        selected_genes=np.load('./dataset/cSCC_Selected_Genes.npy',allow_pickle=True).tolist()
    else:
        raise NotImplementedError
else:
    selected_genes=np.load(params.selected_genes,allow_pickle=True).tolist()

writer = SummaryWriter(log_dir = './logs')
device=torch.device(params.device)

# model_gen = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
# model_gen.fc=torch.nn.Linear(2048,len(selected_genes))
model_gen=M2ORT(num_classes=len(selected_genes),depth=8,dim=256*3, mlp_dim=256*2, heads=12,dim_head=64)
if params.variant=='small':
    model_gen=M2ORT(num_classes=len(selected_genes),depth=6,dim=192*3, mlp_dim=192*2, heads=9,dim_head=64)
elif params.variant=='large':
    model_gen=M2ORT(num_classes=len(selected_genes),depth=12,dim=384*3, mlp_dim=384*2, heads=18,dim_head=64)
elif params.variant=='base':
    pass
else:
    print('Unrecognized M2ORT variant. Go with the M2ORT-Base model by default.')

model_gen.to(device)

if params.data_parallel:
    model_gen=torch.nn.DataParallel(model_gen)
if not params.checkpoint==None:
    model_gen.load_state_dict(torch.load(params.checkpoint,map_location='cpu'))

trainset=None
valset=None
testset=None

if params.dataset_path==None:
    print('Please specify the dataset path.')
    raise Exception

if params.dataset=='HBC':
    from dataset.HBCDataset import HBCDataset
    trainset=HBCDataset(params.dataset_path,mode='train', selected_genes=selected_genes)
    valset=HBCDataset(params.dataset_path,mode='val', selected_genes=selected_genes)
    testset=HBCDataset(params.dataset_path,mode='test', selected_genes=selected_genes)
elif params.dataset=='HER2+':
    from dataset.HER2Dataset import HER2Dataset
    trainset=HER2Dataset(params.dataset_path,mode='train', selected_genes=selected_genes)
    valset=HER2Dataset(params.dataset_path,mode='val', selected_genes=selected_genes)
    testset=HER2Dataset(params.dataset_path,mode='test', selected_genes=selected_genes)
elif params.dataset=='cSCC':
    from dataset.cSCCDataset import cSCCDataset
    trainset=cSCCDataset(params.dataset_path,mode='train', selected_genes=selected_genes)
    valset=cSCCDataset(params.dataset_path,mode='val', selected_genes=selected_genes)
    testset=cSCCDataset(params.dataset_path,mode='test', selected_genes=selected_genes)
else:
    raise NotImplementedError

trainloader=torch.utils.data.DataLoader(trainset, batch_size=params.batch_size, shuffle=True, drop_last=False)
valloader=torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True, drop_last=False)
testloader=torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, drop_last=False)

optimizer0=torch.optim.Adam(model_gen.parameters(),lr=params.lr)
lossfunc_gen=torch.nn.MSELoss()

best_pcc=0.1
# TestModel(model_gen,testloader,device)
# raise Exception

for x in range(params.EPOCH):
    model_gen.train()
    for i,data in enumerate(trainloader):
        _,img,img2,img3,label=data
        img=img.type(torch.FloatTensor).to(device)
        img2=img2.type(torch.FloatTensor).to(device)
        img3=img3.type(torch.FloatTensor).to(device)
        label=label.type(torch.FloatTensor).to(device)
        output=model_gen(img,img2,img3)

        optimizer0.zero_grad()

        loss_gen=lossfunc_gen(output,label)
        loss_gen.backward()
        optimizer0.step()

        # for p in model_dis.parameters():
        #     p.data.clamp_(-0.01, 0.01)

        if i%10==0:
            print('[',x,i,'] loss:',loss_gen.item())

    pcc=TestModel(model_gen,valloader,device)
    writer.add_scalar('loss', loss_gen.item(), global_step=x, walltime=None)
    writer.add_scalar('Pearson', pcc, global_step=x, walltime=None)
    if pcc>best_pcc:
        best_pcc=pcc
        print('New best Pearson:',pcc,'\n======Test=======>')
        TestModel(model_gen,testloader,device)
        torch.save(model_gen.state_dict(),'./weights/M2ORT_{}_{}_PCC_{}_best.pth'.format(str(params.variant), str(params.dataset),str(round(pcc,6))))
