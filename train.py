import torch
from torchtext import data
from torchtext import datasets
import random
from nltk.tokenize import word_tokenize
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


def tokenizer(text): 
	return [tok for tok in word_tokenize(text)]

TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True, batch_first = True)
LABEL = data.Field(sequential=False, use_vocab=False, batch_first = True, tensor_type=torch.FloatTensor)

train,test = data.TabularDataset.splits(path='./', train='train.csv', test='test.csv', format='csv',fields=[('text', TEXT), ('label', LABEL)])

TEXT.build_vocab(train, vectors="glove.6B.100d")

train_iterator, test_iterator = data.BucketIterator.splits((train, test), batch_size=128, sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False)
vocab = TEXT.vocab

class sentiment(nn.Module):
	def __init__(self, vocab, emb_dim = 100, n_h = 128, batch_size = 64, n_layer = 2, bidirectional=True):
		super(sentiment, self).__init__()

		self.bidirectional= bidirectional
		self.emb_dim = emb_dim
		self.n_h = n_h
		self.batch_size = batch_size
		self.n_layer = n_layer

		self.embed = nn.Embedding(len(vocab), self.emb_dim)
		self.embed.weight.data.copy_(vocab.vectors)

		self.lstm = nn.LSTM( input_size=self.emb_dim, hidden_size=self.n_h, num_layers=self.n_layer, batch_first=True, bidirectional=bidirectional)

		self.hidden_to_tag = nn.Linear(self.n_h * 2,1) 

		self.dropout = nn.Dropout(0.3)
	
	def forward(self,X,X_lengths):
		X = self.dropout(self.embed(X))

		X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

		output, (hn,cn) = self.lstm(X)

		X = self.dropout(torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1))

		X = self.hidden_to_tag(X)

		return X.squeeze(1)


model = sentiment(vocab)
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
criterion = criterion.to(device)

def train(model, optimizer, train_iterator):
	epoch_loss = 0
	epoch_acc = 0 

	model.train()

	for batch in train_iterator:
		optimizer.zero_grad()

		(X, X_lengths), Y = batch.text, batch.label
		Y_hat = model(X, X_lengths) 
		loss = criterion(Y_hat, Y)
		loss.backward()

		optimizer.step()
		print(epoch_accuracy(Y_hat,Y))
		epoch_loss += loss.item()
		epoch_acc += epoch_accuracy(Y_hat, Y)
	return epoch_loss/len(train_iterator), epoch_acc/len(train_iterator)

def epoch_accuracy(Y_hat, Y):
	Y_hat = torch.round(F.sigmoid(Y_hat))
	correct = (Y_hat == Y).float() 
	acc = correct.sum()/len(correct)

	return acc

def evaluate(test_iterator):
	epoch_loss = 0
	epoch_acc = 0 

	model.eval()

	for batch in test_iterator:
		(X, X_lengths), Y = batch.text, batch.label
		Y_hat = model(X, X_lengths) 

		loss = criterion(Y_hat, Y)
		epoch_loss += loss.item()
		epoch_acc += epoch_accuracy(Y_hat, Y)

	return epoch_loss/len(test_iterator), epoch_acc/len(test_iterator)

epochs = 5
for epoch in range(epochs):

	train_epoch_loss, train_epoch_acc = train(model, optimizer, train_iterator)
	test_epoch_loss, test_epoch_acc = evaluate(test_iterator)

	print('Epoch ',epoch+1,' -- Train Loss: ',train_epoch_loss, ' Train accuracy: ', train_epoch_acc)
	print('Epoch ',epoch+1,' -- Testn Loss: ',train_epoch_loss, ' Test accuracy: ', train_epoch_acc)
	print('-'*25)

model.save_state_dict('mytraining.pt')





