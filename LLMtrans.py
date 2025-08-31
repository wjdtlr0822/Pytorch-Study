import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================
# 1. 데이터 준비
# ======================
# 예시 문장
text = "hello world"

# 문자 단위 토큰화
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)  # [h,e,l,l,o, ,w,o,r,l,d]

# ======================
# 2. 작은 모델 정의 (Transformer 블록 1개)
# ======================
class TinyLLM(nn.Module):
    def __init__(self, vocab_size, emb_size=16, n_heads=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_size)
        self.attn = nn.MultiheadAttention(emb_size, n_heads, batch_first=True)
        self.fc = nn.Linear(emb_size, vocab_size)

    def forward(self, x):
        x = self.token_emb(x)                 # [batch, seq, emb]
        print(x)
        attn_out, _ = self.attn(x, x, x)      # self-attention
        logits = self.fc(attn_out)            # [batch, seq, vocab_size]
        return logits

model = TinyLLM(vocab_size)

# ======================
# 3. 학습 준비
# ======================
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# ======================
# 4. 학습 루프 (간단히)
# ======================
for epoch in range(200):
    inputs = data[:-1].unsqueeze(0)   # "hello worl"
    targets = data[1:].unsqueeze(0)   # "ello world"

    logits = model(inputs)
    loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ======================
# 5. 텍스트 생성
# ======================
context = torch.tensor([stoi['h']], dtype=torch.long).unsqueeze(0)  # "h"
model.eval()
with torch.no_grad():
    for _ in range(10):  # 10글자 생성
        logits = model(context)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_id], dim=1)

print("Generated text:", decode(context[0].tolist()))
