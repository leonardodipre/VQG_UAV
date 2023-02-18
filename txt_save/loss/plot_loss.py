import pandas as pd
import matplotlib.pyplot as plt
import numpy
# Carica il file CSV in un DataFrame di Pandas
df = pd.read_csv(r'D:\Leonardo\UAV_VQG\VQG_UAV\txt_save\loss_valuesVQG_2_UAV.csv')

# Estrae i valori numerici dalle stringhe delle colonne 'Epoch' e 'Loss'
df['Epoch'] = df['Epoch'].str.extract('(\d+)')
df['Loss'] = df['Loss'].str.extract('(\d+\.\d+)').astype(float)

lista_epochs = df['Epoch']
lista_loss = df['Loss']



plt.plot(lista_epochs.astype(str), lista_loss)


#plt.plot(lista_epochs.astype(str), lista_loss)

# Aggiunge un titolo e delle etichette agli assi
plt.title('Loss per Epoch')
plt.xlabel('epoch')
plt.ylabel('Loss')

# Mostra il grafico
plt.savefig('loss_UAV2.png')
