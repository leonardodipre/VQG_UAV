import pandas as pd
import torch
from torch import nn, optim
from dataloader import get_loader
import torchvision.transforms as transforms
import json
from torchmetrics import BLEUScore

from eval import eval1  , eval2 
from  modello  import CNNtoRNN

import torchvision.transforms as transforms


from torch.nn.utils.rnn import pack_padded_sequence
import json


def blue_eval(preds_machine, target_human):
    
    weights_1 =(1.0/1.0, 0,0,0)
    weights_2 = (1.0/2.0, 1.0/2.0, 0,0)
    weights_3= (1.0/3.0, 1.0/3.0, 1.0/3.0,0 )
    weights_4 = (1.0/4.0, 1.0/4.0, 1.0/4.0 , 1.0/4.0)

    bleu_1 = BLEUScore(1, weights_1)
    bleu_2 = BLEUScore(2, weights_2)
    bleu_3 = BLEUScore(3, weights_3)
    bleu_4 = BLEUScore(4, weights_4)

    return bleu_1(preds_machine, target_human).item(), bleu_2(preds_machine, target_human).item(), bleu_3(preds_machine, target_human).item(), bleu_4(preds_machine, target_human).item()



def evaluation():
    

    ###########################################################################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Declare transformations (later)
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            # transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
        ]
    )

    ###########################################################################################################
                            #Dataset per Vacab
                                                        #DIRECTORY#
   
    csv = r'train.csv'
    imm_dir =r'D:\Leonardo\Datasets\Coco\train2014\train2014'


    freq_threshold = 2 # 4019 vocab

    ############################################################################################################################################

    #Carico il dataset per i vocaboli presenti in train.csv

    _, dataset_vocab = get_loader(
        csv, imm_dir, freq_threshold, transform=transform , num_workers=1,
        )
    
##################################################################################################




    embed_size = 224
    hidden_size = 224
    vocab_size = len(dataset_vocab.vocab)
    num_layers = 1
    learning_rate = 1e-4

    


    ###########################################################################

        # Model declaration
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

    for j in range(16,17):
        
        PATH = r"save_Model_UAV" + str(j)
        print("Read ", PATH)
        model.load_state_dict(torch.load(PATH))

        model.eval()
        
        print("IMM 1")
        #a =[["What type of trees are in the landscape below the grass field?", "How many trees are in the landscape below the grass field?","Is there any other vegetation in the landscape below the grass field?", 
        #"Are there any buildings or structures in the landscape below the grass field?", "Is the grass field flat or hilly?", "Are there any roads or paths leading to the grass field?", "Are there any other fields in the landscape below the grass field?"
        #,"Are there any bodies of water in the landscape below the grass field?", "Are there any other features in the landscape below the grass field?",  "What is the size of the grass field?",  "Is the grass field surrounded by a fence or wall?",  "Are there any shadows in the landscape below the grass field?",  "Are there any animals in the landscape below the grass field?",  "Are there any people in the landscape below the grass field?",  "Are there any clouds in the sky above the grass field?",  "Is the grass field in direct sunlight?",  "Is the grass field in a valley or on a hilltop?",  "Are there any other colors in the landscape below the grass field?", "Are there any other shapes in the landscape below the grass field?",  "Are there any other patterns in the landscape below the grass field?"]]
        a = [['What type of trees are in the landscape below the grass field?', 'How many trees are in the landscape below the grass field?', 'Is there any other vegetation in the landscape below the grass field?', 'Are there any buildings or structures in the landscape below the grass field?', 'Is the grass field flat or hilly?', 'Are there any roads or paths leading to the grass field?', 'Are there any other fields in the landscape below the grass field?', 'Are there any bodies of water in the landscape below the grass field?', 'Are there any other features in the landscape below the grass field?', ' What is the size of the grass field?', ' Is the grass field surrounded by a fence or wall?', ' Are there any shadows in the landscape below the grass field?', ' Are there any animals in the landscape below the grass field?', ' Are there any people in the landscape below the grass field?', ' Are there any clouds in the sky above the grass field?', ' Is the grass field in direct sunlight?', ' Is the grass field in a valley or on a hilltop?', ' Are there any other colors in the landscape below the grass field?', ' Are there any other shapes in the landscape below the grass field?', ' Are there any other patterns in the landscape below the grass field?']]
        #pred =beam_search(model, device, dataset_vocab, "D:\Leonardo\VQG_final\immagini5\COCO_val2014_000000000136.jpg")
        pred = eval1(model, device, dataset_vocab, r"D:\Leonardo\Datasets\UAV\images\0110000001.png",)
        
        print(pred)
        preds_machine = [pred]
        print(blue_eval(preds_machine, a))


       
if __name__ == "__main__":
    evaluation()