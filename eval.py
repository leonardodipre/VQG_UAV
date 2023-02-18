import torch
from torch import autograd, nn, optim
from  modello  import CNNtoRNN
from dataloader import get_loader
import torchvision.transforms as transforms
from PIL import Image  # Load img
import os
from torchmetrics import BLEUScore

def blue_eval(preds_machine, target_human):
    
    weights_1 =(1.0/1.0, 0,0,0)
    weights_2 = (1.0/2.0, 1.0/2.0, 0,0)
    weights_3= (1.0/3.0, 1.0/3.0, 1.0/3.0,0 )
    weights_4 = (1.0/4.0, 1.0/4.0, 1.0/4.0 , 1.0/4.0)


    bleu_2 = BLEUScore(2, weights_2)
    bleu_3 = BLEUScore(3, weights_3)
    bleu_4 = BLEUScore(4, weights_4)

    return bleu_2(preds_machine, target_human), bleu_3(preds_machine, target_human), bleu_4(preds_machine, target_human)


#usa caption immage con mutinomial
def eval1(model, device, dataset, immage_url):

    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            #transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
        ]
    )
    
    #test_img1 = transform( Image.open(os.path.join(dir_loc, immage_url)).convert('RGB')).unsqueeze(0)
    test_img1 = transform(Image.open( immage_url).convert('RGB')).unsqueeze(0)
    
   
        
    return  " ".join(model.caption_image_multinomial(test_img1.to(device), dataset.vocab)[:-1])+"?"


def eval2(model, device, dataset, dir_loc, immage_url, questions):

    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            #transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
        ]
    )
    
    test_img1 = transform( Image.open(os.path.join(dir_loc, immage_url)).convert('RGB')).unsqueeze(0)
    
    prediction = " ".join(model.caption_image_multinomial(test_img1.to(device), dataset.vocab)[:-1])+"?"
    
   

    return prediction, questions
    

#Utilizza beam search
def beam_search(model, device, dataset, immage_url):

    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            #transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
        ]
    )
    
    #test_img1 = transform( Image.open(os.path.join(dir_loc, immage_url)).convert('RGB')).unsqueeze(0)
    test_img1 = transform(Image.open( immage_url).convert('RGB')).unsqueeze(0)
    
    
    return model.beam_search_multinomial(test_img1.to(device), dataset.vocab, beam_width= 5)
    