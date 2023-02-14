import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CNNencoder(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(CNNencoder,self).__init__()
        self.train_CNN = train_CNN
        self.resNet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.preprocess = ResNet18_Weights.DEFAULT.transforms()
        self.resNet.fc = nn.Linear(self.resNet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resNet(self.preprocess(images))
        return features

class RNNdecoder(nn.Module):
    def __init__(self,embed_size, hidde_size, vocab_size, num_layers):
        super(RNNdecoder, self).__init__()

        # look up matrik, that map the indix for output words
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.lstm = nn.GRU(embed_size, hidde_size, num_layers, batch_first=True)
        self.linear= nn.Linear(hidde_size*2, vocab_size) # Append image features to all hidden states

        self.dropout = nn.Dropout(0)

    def forward(self, features, questions, lengths):
        embeddings = self.dropout(self.embed(questions))
        packed = pack_padded_sequence(embeddings, [l-1 for l in lengths], batch_first=True)
        hiddens, _ = self.lstm(packed,features.squeeze().unsqueeze(0))
        hiddens = pad_packed_sequence(hiddens, batch_first=True)
        new_hiddens = torch.cat((hiddens[0], features.unsqueeze(1).expand(-1,hiddens[0].shape[1],-1)), dim=2)
        packed = pack_padded_sequence(new_hiddens, [l-1 for l in lengths], batch_first=True)
        outputs = self.linear(packed[0])
        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = CNNencoder(embed_size)
        self.decoderRNN = RNNdecoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions, lengths):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions, lengths)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []
        
        self.eval()

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            
            states = x
            start_tok = self.decoderRNN.embed(torch.tensor([vocabulary.stoi["<SOS>"]]).cuda()).unsqueeze(0)
            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(start_tok, states)
                
                hiddens = torch.cat((hiddens, x),dim=2)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                start_tok = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

                result_caption.append(predicted.item())

        return [vocabulary.itos[idx] for idx in result_caption]