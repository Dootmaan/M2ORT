import random
import math
from scipy.stats import pearsonr
import torch
import numpy as np

def TestModel(model_gen,valloader,device=torch.device('cuda')):
    model_gen.eval()
    pearson_list=[]
    rmse_list=[]
    for i,data in enumerate(valloader):
        _,img,img2,img3,label=data
        img=img.type(torch.FloatTensor).to(device)
        img2=img2.type(torch.FloatTensor).to(device)
        img3=img3.type(torch.FloatTensor).to(device)
        label=label.data.numpy()
        with torch.no_grad():
            output=model_gen(img,img2,img3)
        output=output.cpu().data.numpy()
        pearson_list.append(pearsonr(output.squeeze(0), label.squeeze(0))[0])
        rmse_list.append(math.sqrt(np.sum((output.squeeze(0)-label.squeeze(0))**2)/output.shape[-1]))
        # pvalue=pearsonr(output, label)[1]
        if i%100==0:
            print('[TEST ITER {}/{}] Pearson: {}, p-value: {}'.format(i, len(valloader), pearsonr(output.squeeze(0), label.squeeze(0))[0],pearsonr(output.squeeze(0), label.squeeze(0))[1]))

    print('Avg Pearson:',np.sum(pearson_list)/len(pearson_list))
    print('Avg RMSE:',np.sum(rmse_list)/len(rmse_list))
    return np.sum(pearson_list)/len(pearson_list)

def TestSingle(model_gen, img,img2,img3,device='cuda'):
    model_gen.eval()

    img=img.type(torch.FloatTensor).to(device)
    img2=img2.type(torch.FloatTensor).to(device)
    img3=img3.type(torch.FloatTensor).to(device)

    with torch.no_grad():
        output=model_gen(img,img2,img3)
    output=output.cpu().data.numpy()

    return output