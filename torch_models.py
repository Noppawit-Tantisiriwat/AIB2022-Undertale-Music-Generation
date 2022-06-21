import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cpu")

## single_rnn
class TobyFox(nn.Module):
    def __init__(self,hidden_size=256,possible_notes=1140,embedding_dim=50):
        super().__init__()
        self.hidden_size = hidden_size
        self.possible_notes = possible_notes
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.possible_notes,embedding_dim)
        self.rnn = nn.LSTM(input_size = embedding_dim, hidden_size=hidden_size ,batch_first=True)
        self.output = torch.nn.Linear(in_features=hidden_size, out_features=possible_notes, bias=True, device=None, dtype=None)

    def init_hidden(self, size):
        return (torch.zeros(1, self.batch, size).to(device), torch.zeros(1, self.batch, size).to(device))

    def forward(self,x):

        self.batch = x.shape[0]

        h0, c0 = self.init_hidden(self.hidden_size)

        x = self.embedding(x)

        x = torch.squeeze(x, dim=-2)

        o1, (h1, c1) = self.rnn(x, (h0, c0))

        x = self.output(h1.view(self.batch, self.hidden_size))


        return x, (h1, c1)


## bilstm

class MusicEmbeddingLSTM(nn.Module):

    def __init__(self, embedding_size, hidden_size, post_embedding, vocab_size=1140, bidirectional=True):

        super(MusicEmbeddingLSTM, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.post_embedding = post_embedding
        self.bidirectional = bidirectional

        self.hidden_lstm_input = hidden_size * 2 if bidirectional else hidden_size


        self.embedding_layer = nn.Embedding(1140, self.embedding_size)

        self.base_lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=bidirectional, batch_first=True)
        self.hidden_lstm = nn.LSTM(self.hidden_lstm_input, self.hidden_lstm_input, batch_first=True)
        self.output_lstm = nn.LSTM(self.hidden_lstm_input, self.post_embedding, batch_first=True)
        self.classifier = nn.Linear(self.post_embedding, self.vocab_size)

        #self.classifier_activation = torch.nn.Softmax(dim=-1)

        
    def init_hidden(self, size, bidirectional):
        return (torch.zeros(1+ (1*int(bidirectional)), self.batch, size).to(device), torch.zeros(1+ (1*int(bidirectional)), self.batch, size).to(device))

        
    def forward(self, x):
        
        self.batch = x.shape[0]

        h0, c0 = self.init_hidden(self.hidden_size, bidirectional=True)
        h1, c1 = self.init_hidden(self.hidden_lstm_input, bidirectional=False)
        h2, c2 = self.init_hidden(self.post_embedding, bidirectional=False)

        x = self.embedding_layer(x)
        #if len(x.shape) != 2:
        x = torch.squeeze(x, dim=-2)
    

        o1, (h0, c0) = self.base_lstm(x, (h0, c0))
        o2, (h1, c1) = self.hidden_lstm(o1, (h1, c1))
        o3, (h2, c2) = self.output_lstm(o2, (h2, c2))

        x = self.classifier(h2.view(self.batch, self.post_embedding))
        #x = self.classifier_activation(x)


        return x, (h0, h1, h2)



## teacher forcing

class TeacherForcingMusicLSTM(nn.Module):

    def __init__(self , Tx, embedding_size, n_hidden, vocab_size, test_size=5):

        super(TeacherForcingMusicLSTM, self).__init__()

        self.Tx = Tx
        self.embedding_size = embedding_size
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size
        self.test_size = test_size

        self.embedding = torch.nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size, padding_idx=0) # Embedding layer ที่กำหนดให้ผลลัพธ์ของ <S> เป็น vector 0
        # LSTM cell 2 ชั้น
        self.lstm_1 = torch.nn.LSTMCell(input_size=self.embedding_size, hidden_size=n_hidden)
        self.lstm_2 = torch.nn.LSTMCell(input_size=n_hidden, hidden_size=n_hidden)
        # classifier ที่มีขนาดเท่ากับจำนวนโน๊ตที่เป็นไปได้ทั้งหมด
        self.dense = torch.nn.Linear(n_hidden, vocab_size)

    # กำหนดค่า hidden state และ cell state เริ่มต้นของ lstm -> เบื้องต้นเป็น vector 0
    def _init_hidden(self, batch, size):
        
        return torch.zeros(batch, size).to(device), torch.zeros(batch, size).to(device)
    # กำหนดค่า start token สำหรับการ sampling generate ระหว่างการ train
    def _init_x0(self):

        x0 = torch.zeros(self.test_size).to(device)
    

        return x0

    def forward(self, x):

        if self.training: # ใน mode training ของ model
            
            self.batch = x.shape[0] # กำหนด batch
            # สร้าง hidden state และ cell state
            h0, c0 = self._init_hidden(self.batch, self.n_hidden)
            h1, c1 = self._init_hidden(self.batch, self.n_hidden)

            #นำ input ผ่าน embedding layer
            x = self.embedding(x)

            output = []

            for t in range(self.Tx): # loop ผ่าน sequence    

                x_t = x[:, t]

                x_t = x_t.view((self.batch, self.embedding_size))

                h0, c0 = self.lstm_1(x_t, (h0, c0))

                h1, c1 = self.lstm_2(h0, (h1, c1))

                out = self.dense(h1)

                #out = self.dense(h0)

                out = torch.unsqueeze(out, dim=0)

                output.append(out)


            return output
        
        else: # ใน model evaluation ของโมเดล

            x = self._init_x0().to(device).long() # กำหนด x0 (start token)
            # สร้าง hidden state และ cell state
            h0, c0 = self._init_hidden(self.test_size, self.n_hidden)
            h1, c1 = self._init_hidden(self.test_size, self.n_hidden)

            #นำ input ผ่าน embedding layer
            x = self.embedding(x)
            output = []

            for t in range(self.Tx): # loop ผ่าน sequence    


                ###x_t = x[:, t, :]

                ###x_t = x_t.view((self.batch, self.vocab_size))

                h0, c0 = self.lstm_1(x, (h0, c0))

                h1, c1 = self.lstm_2(h0, (h1, c1))

                out = self.dense(h1)

                ###out = self.dense(h0)

                out = torch.unsqueeze(out, dim=0)

                output.append(out)

                x = self.embedding(torch.multinomial(F.softmax(torch.squeeze(out), dim=-1), 1).long()) # ทำการสุ่มแบบ multinomial เพื่อทำนายโน๊ตตัวถัดไป

                x = x.view((self.test_size, self.embedding_size)) # นำโน๊ตตัวที่ได้จากการทำนายเป็น input ของ step ถัดไป


            return output







