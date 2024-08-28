import torch
import torch.nn as nn

# 读取数据
with open('poetryFromTang.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 创建字符集
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}

# 将字符转换为整数编码
encoded = torch.tensor([char2int[ch] for ch in text], dtype=torch.long)

def one_hot_encode(arr, n_labels):
    # 修改为 [batch_size * seq_length, n_labels] 形状
    one_hot = torch.zeros(arr.size(0) * arr.size(1), n_labels, dtype=torch.float32)
    # 使用 reshape 来展平 `arr` 的维度
    one_hot.scatter_(1, arr.reshape(-1, 1), 1.)
    # 恢复为 [batch_size, seq_length, n_labels] 形状
    one_hot = one_hot.view(arr.size(0), arr.size(1), n_labels)
    return one_hot


def get_batches(arr, batch_size, seq_length):
    batch_size_total = batch_size * seq_length
    n_batches = len(arr) // batch_size_total

    print(f"数据长度: {len(arr)}")
    print(f"总批次数: {n_batches}")

    if n_batches == 0:
        raise ValueError("数据不足以生成批次，请减少 batch_size 或 seq_length")

    arr = arr[:n_batches * batch_size_total]
    arr = arr.view(batch_size, -1)

    for n in range(0, arr.size(1), seq_length):
        x = arr[:, n:n+seq_length]
        y = torch.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

# 参数
batch_size = 12
seq_length = 32
n_chars = len(chars)

# 构建dataloader
train_data = list(get_batches(encoded, batch_size, seq_length))

class CharRNN(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001, model_type='LSTM'):
        super(CharRNN, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.model_type = model_type
        
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, dropout=drop_prob, batch_first=True) if model_type == 'LSTM' else nn.GRU(len(self.chars), n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(self.chars))
    
    def forward(self, x, hidden):
        r_output, hidden = self.lstm(x, hidden) if self.model_type == 'LSTM' else self.gru(x, hidden)
        out = self.dropout(r_output)
        out = out.reshape(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.model_type == 'LSTM':
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        else:
            hidden = weight.new(self.n_layers, batch_size, self.n_hidden).zero_()
        return hidden

def train(model, data, epochs=10, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    if not data:
        raise ValueError("训练数据为空，请检查 get_batches 的输入参数。")

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    val_idx = int(len(encoded) * (1 - val_frac))
    train_data, val_data = encoded[:val_idx], encoded[val_idx:]

    val_batches = list(get_batches(val_data, batch_size, seq_length))
    if len(val_batches) == 0:
        raise ValueError("验证数据不足以生成批次，请减少 batch_size 或 seq_length")

    train_batches = list(get_batches(train_data, batch_size, seq_length))
    if len(train_batches) == 0:
        raise ValueError("训练数据不足以生成批次，请减少 batch_size 或 seq_length")

    for e in range(epochs):
        h = model.init_hidden(batch_size)
        batch_loss = 0
        
        for x, y in train_batches:
            x = one_hot_encode(x, len(model.chars))
            inputs, targets = x, y

            h = tuple([each.data for each in h])
            
            model.zero_grad()
            output, h = model(inputs, h)
            loss = criterion(output, targets.view(batch_size * seq_length))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            
            batch_loss += loss.item()
        
        if len(train_batches) > 0:
            batch_loss /= len(train_batches)
        else:
            batch_loss = 0
        
        if e % print_every == 0:
            model.eval()
            val_h = model.init_hidden(batch_size)
            val_losses = []
            for x, y in val_batches:
                x = one_hot_encode(x, len(model.chars))
                inputs, targets = x, y
                
                val_h = tuple([each.data for each in val_h])
                output, val_h = model(inputs, val_h)
                val_loss = criterion(output, targets.view(batch_size * seq_length))
                val_losses.append(val_loss.item())

            model.train()
            print(f"Epoch: {e+1}/{epochs}... Loss: {batch_loss}... Val Loss: {sum(val_losses)/len(val_losses)}")
            perplexity = torch.exp(torch.tensor(sum(val_losses)/len(val_losses))).item()
            print(f"Perplexity: {perplexity}")

n_hidden = 512
n_layers = 2

model = CharRNN(chars, n_hidden=n_hidden, n_layers=n_layers, model_type='LSTM')

train_data = list(get_batches(encoded, batch_size, seq_length))
print(f"Number of training batches: {len(train_data)}")

train(model, train_data, epochs=20, lr=0.001)
