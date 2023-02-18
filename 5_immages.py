import pandas as pd
import torch
from torch import nn, optim
from dataloader import get_loader
import torchvision.transforms as transforms
import json
from torchmetrics import BLEUScore

from eval import eval1  , eval2 , beam_search
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

def compose_topK(vocabulary, pred, top_k):
    
       
        a3 = []
        for i in range(len(pred)):     
            pred_list = " ".join(([vocabulary.itos[idx] for idx in pred[i][1]])[:-1])+"?"
            cost_len = pred[i][2]/len(pred_list)

            a3.append([pred_list , cost_len])
       
        
        a3 = sorted(a3, key=lambda x: x[1], reverse=True)[:top_k]

        
        finale = []
        for i in range(len(a3)):
            finale.append(a3[i][0])  

        return finale

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
   
    csv = r'csv\train.csv'
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
        
        PATH = r"modelli\save_Model_UAV" + str(j)
        print("Read ", PATH)
        model.load_state_dict(torch.load(PATH))
        vocabulary = dataset_vocab.vocab
        model.eval()
        
        print("IMM 1")
        human = [["What type of building is the white building facade?"
                ,"How many stories tall is the white building facade?" ,"What type of material is the red roof made of?"
                ,"How large is the red roof?","What type of vegetation is located in the soil on the right?"
                ,"How much gravel is located in the center?","What type of soil is located on the right?","Are there any other buildings in the vicinity of the white building facade?"
                ,"Are there any trees or other vegetation near the red roof?"
                ,"Are there any other structures near the red roof?"," Are there any other colors present in the image besides red, white, and soil?"
                , "How far away is the red roof from the white building facade?"
                , "Is the gravel in the center of the image flat or uneven?", "Are there any other colors present in the soil on the right?"
                , "Are there any other colors present in the gravel in the center?", "Are there any other colors present in the red roof?"
                , "Are there any other colors present in the white building facade?", "Are there any other structures in the vicinity of the soil on the right?"
                , "Are there any other structures in the vicinity of the gravel in the center?"]]

        #pred =beam_search(model, device, dataset_vocab, r"D:\Leonardo\Datasets\UAV\images\017000108.png")
        pred = eval1(model, device, dataset_vocab, r"D:\Leonardo\Datasets\UAV\images\017000108.png")
    
        pred_beam = beam_search(model, device, dataset_vocab, r"D:\Leonardo\Datasets\UAV\images\017000108.png")

        pre_list_beam = compose_topK(vocabulary, pred_beam, 5)
    
        print("\nGredy search question machine")
        pred_gredy= [pred]
        print(pred_gredy)
        print("\nBeam search question machine")
        print(pre_list_beam)
    
        print("\nBeamSearch", blue_eval(pre_list_beam, human))
        print("\nGredy_search", blue_eval(pred_gredy, human))
        pred=[]
        print("\n\n")

        

        print("IMM 2")
        human = [["How many trees are in the image?","Are the trees evenly spaced?","What is the approximate size of the trees?"
                    ,"Are there any other objects in the image besides the trees?","Are the trees in a straight line or in a pattern?"
                    ,"Are the trees in a forest or in a park?" ,"Are there any other plants in the image?","Are there any buildings in the image?"
                    ,"Are there any roads in the image?", "Are there any bodies of water in the image?"
                    , "Are there any other trees in the image besides the large ones?"
                    , "Are the trees in the image deciduous or evergreen?", "Are the trees in the image healthy or diseased?"
                    , "Are there any animals in the image?"
                    , "Are there any signs of human activity in the image?", "Are there any shadows in the image?"
                    , "Are there any clouds in the image?", "Are there any other colors in the image besides green?"
                    , "Are there any other shapes in the image besides trees?", "Are there any other objects in the image that are not trees?"]]

       
        pred = eval1(model, device, dataset_vocab, r"D:\Leonardo\Datasets\UAV\images\017000111.png")
    
        pred_beam = beam_search(model, device, dataset_vocab, r"D:\Leonardo\Datasets\UAV\images\017000111.png")

        pre_list_beam = compose_topK(vocabulary, pred_beam, 5)
    
        print("\nGredy search question machine")
        pred_gredy= [pred]
        print(pred_gredy)
        print("\nBeam search question machine")
        print(pre_list_beam)
    
        print("\nBeamSearch", blue_eval(pre_list_beam, human))
        print("\nGredy_search", blue_eval(pred_gredy, human))
        pred=[]
        print("\n\n")


        print("IMM 3")
        human = [["What type of road is this?","What is the width of the road?"
                ,"Are there any other roads nearby?","Are there any buildings or structures near the road?"
                ,"Are there any other shadows in the image?","What type of vegetation is on the left and right of the road?"
                ,"Is the vegetation on the left and right of the road the same?"
                ,"Are there any other objects in the image?","Are there any other shadows in the image?"
                ,"Is the vegetation in the center of the road the same as the vegetation on the left and right?"
                , "Are there any other roads in the image?", "Are there any other objects in the image?"
                , "Are there any other shadows in the image?", "Are there any other roads in the image?"
                , "Are there any other objects in the image?", "Are there any other shadows in the image?"
                , "Are there any other roads in the image?", "Are there any other objects in the image?"
                , "Are there any other shadows in the image?", "Are there any other roads in the image?"]]

       
        pred = eval1(model, device, dataset_vocab, r"D:\Leonardo\Datasets\UAV\images\018000296.png")
    
        pred_beam = beam_search(model, device, dataset_vocab, r"D:\Leonardo\Datasets\UAV\images\018000296.png")

        pre_list_beam = compose_topK(vocabulary, pred_beam, 5)
    
        print("\nGredy search question machine")
        pred_gredy= [pred]
        print(pred_gredy)
        print("\nBeam search question machine")
        print(pre_list_beam)
    
        print("\nBeamSearch", blue_eval(pre_list_beam, human))
        print("\nGredy_search", blue_eval(pred_gredy, human))
        pred=[]
        print("\n\n")


        print("IMM 4")
        human = [["How many cars are parked in the field?"
                    ,"What color are the cars?"
                    ,"Is the road paved or unpaved?"
                    ,"Are there any other vehicles in the field?"
                    ,"Are there any people in the field?"
                    ,"Are there any buildings in the field?"
                    ,"Are there any trees in the field?"
                    ,"Are there any other objects in the field?"
                    ,"How wide is the road?"
                    , "How wide is the field?"
                    , "What is the terrain of the field?"
                    , "Are there any other roads in the vicinity?"
                    , "Are there any other fields in the vicinity?"
                    , "Are there any other cars parked on the road?"
                    , "Are there any other cars parked in the field?"
                    , "Are there any other objects on the road?"
                    , "Are there any other objects in the field?"
                    , "Are there any other people in the vicinity?"
                    , "Are there any other buildings in the vicinity?"
                    , "Are there any other trees in the vicinity?"]]

       
        pred = eval1(model, device, dataset_vocab, r"D:\Leonardo\Datasets\UAV\images\019000753.png")
    
        pred_beam = beam_search(model, device, dataset_vocab, r"D:\Leonardo\Datasets\UAV\images\019000753.png")

        pre_list_beam = compose_topK(vocabulary, pred_beam, 5)
    
        print("\nGredy search question machine")
        pred_gredy= [pred]
        print(pred_gredy)
        print("\nBeam search question machine")
        print(pre_list_beam)
    
        print("\nBeamSearch", blue_eval(pre_list_beam, human))
        print("\nGredy_search", blue_eval(pred_gredy, human))
        pred=[]
        print("\n\n")



        print("IMM 5")
        human = [["What color are the cars?"
                ,"How many doors does each car have?"
                ,"Are the cars parked in a line?"
                ,"Are the cars facing the same direction?"
                ,"Are there any other vehicles in the image?"
                ,"Are there any people in the image?"
                ,"Are there any buildings in the image?"
                ,"Are there any trees in the image?"
                ,"Are there any street signs in the image?"
                , "Are there any other objects in the image?"
                , "What type of surface is the cars parked on?"
                , "Are the cars parked close together?"
                , "Are the cars parked in a parking lot?"
                , "Are the cars parked on a street?"
                , "Are the cars parked in a driveway?"
                , "Are the cars parked in a garage?"
                , "Are the cars parked in a field?"
                , "Are the cars parked in a park?"
                , "Are the cars parked in a shopping center?"
                , "Are the cars parked in a residential area?"]]

       
        pred = eval1(model, device, dataset_vocab, r"D:\Leonardo\Datasets\UAV\images\018000572.png")
    
        pred_beam = beam_search(model, device, dataset_vocab, r"D:\Leonardo\Datasets\UAV\images\018000572.png")

        pre_list_beam = compose_topK(vocabulary, pred_beam, 5)
    
        print("\nGredy search question machine")
        pred_gredy= [pred]
        print(pred_gredy)
        print("\nBeam search question machine")
        print(pre_list_beam)
    
        print("\nBeamSearch", blue_eval(pre_list_beam, human))
        print("\nGredy_search", blue_eval(pred_gredy, human))
        pred=[]
        print("\n\n")

       
if __name__ == "__main__":
    evaluation()