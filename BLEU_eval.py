import pandas as pd
import torch
from torch import nn, optim
from dataloader import get_loader
import torchvision.transforms as transforms
import json
from torchmetrics import BLEUScore
from eval import eval2
from eval import eval1
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

import csv

def get_questions_by_id(file_path, target_id):
    questions = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['id'] == target_id:
                questions.append(row['question'])
    return questions

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
   
    csv = r'D:\Leonardo\UAV_VQG\VQG_UAV\csv\train.csv'
    
    imm_dir =r'D:\Leonardo\Datasets\UAV\images'



    freq_threshold = 2 # 4019 vocab

    ############################################################################################################################################

    #Carico il dataset per i vocaboli presenti in train.csv

    _, dataset_vocab = get_loader(
        csv, imm_dir, freq_threshold, transform=transform , num_workers=1,
        )
    
    #########################################################################################
                        #parte per validation_coco.csv
       #Dir cartelle
    csv = r"D:\Leonardo\UAV_VQG\VQG_UAV\csv\validation_BLEU.csv"              
    file_csv =pd.read_csv(csv, dtype={'id': 'str'})         # ID IMMAGINI
    dir_loc  =r"D:\Leonardo\Datasets\UAV\images"
       
    #Carico la lista
    id_imm_list = file_csv["id"]
    questions_list = file_csv["question"]
    
    
    
##################################################################################################




    embed_size = 224
    hidden_size = 224
    vocab_size = len(dataset_vocab.vocab)
    num_layers = 1
    learning_rate = 1e-4

    
    BLEU_tot1 = 0
    BLEU_tot2= 0
    BLEU_tot3 = 0
    BLEU_tot4 = 0

    ###########################################################################

        # Model declaration
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

    for j in range(0,20):
        
        PATH = r"D:\Leonardo\UAV_VQG\VQG_UAV\modelli\save_Model_UAV" + str(j)
        print("Read ", PATH)
        model.load_state_dict(torch.load(PATH))

        model.eval()

        BL1 = 0
        BL2 = 0
        BL3 = 0
        BL4 = 0

        for i in range(len(id_imm_list)):
            list_ = []
            for original_string in questions_list[i].split(','):
                
                characters_to_remove = "[]'"
                
                new_string = original_string
                
                for character in characters_to_remove:
                    new_string = new_string.replace(character, "")
                
                #print("Strig " , new_string) 
                list_ +=[new_string]
            
            
           

            id_png = str(id_imm_list[i]) + ".png"
        
            
            prediction, question = eval2(model,device, dataset_vocab, dir_loc,id_png, list_ )

           
            
            question = [question]
            pre_parantesist = [prediction]



            bl1, bl2, bl3, bl4 = blue_eval(pre_parantesist, question)
            
           
            BL1 += bl1
            BL2 += bl2
            BL3 += bl3
            BL4 += bl4

            if(i%100 == 0):
                print(i)

        l = len(id_imm_list)
        
        print("BLEU1 " , BL1/l) 
        print("BLEU2 ",  BL2/l) 
        print("BLEU3 " , BL3/l) 
        print("BLEU4 ",  BL4/l)   
        print("l", l) 
        
        BLEU_tot1 += BL1/l
        BLEU_tot2+= BL2/l
        BLEU_tot3 += BL3/l
        BLEU_tot4 += BL4/l

        with open(r"D:\Leonardo\UAV_VQG\VQG_UAV\txt_save\BLUE_UAV.txt", "a") as f:
            f.write("Epoch: " + str(j) + ", BLUE:1: " + str(BL1/l) + ", BLUE_2: " + str(BL2/l) + ", BLUE_3: " + str(BL3/l) + ", BLUE_4: " + str(BL4/l) + "\n")
        
if __name__ == "__main__":
    evaluation()