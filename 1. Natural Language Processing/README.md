

## 1. NLP Basic Models

### (1) Seq2Seq
> Encoder와 Decoder를 따로 만들어서, 입력으로 주어진 정보를 hidden & cell state에 잘 학습시킨 다음 Decoder에 전달해서 해석에 사용
 - Encoder & Decoder with LSTM
 - Neural Machine Translation
 - Embedding token to hidden state dim
 - Teacher Forcing
![Seq2Seq](../docs/seq2seq.png)



### (2) Seq2Seq with Attention
> Seq2Seq의 Encoder와 Decoder를 통해 학습을 할 때, 입력 Sequence의 길이가 길어지면 Deocder로 전달하는 Hidden & Cell state에서 병목현상이 일어날 수 밖에 없다. 또한 Gradient가 첫번째 Layer까지 도달하기 어려운 문제 (Long Term Dependency)를 해결하고 입력으로 주어진 정보들과의 유사도를 계산해서 필요한 정보를 얻기위해 Attention 구조를 사용
 - Encoder & Decoder with LSTM + Attention Layer
 - Use all hidden states in Encoder for caculate simialarity with Deocder's each hidden state (Dot Attention, Concat Attention)
 - Solve Long Term Dependency problem -> More shoter way for gradient to get in encoder
 - Neural Machine Translation
 - Teacher Forcing
![Seq2Seq with Attention](../docs/seq2seqwithAttention.png)



### (3) Transformer

>Self Attention 구조를 통해 오로지 Attention Block으로만 처리해서 Long Term Dependency 문제를 해결하고 입력값(query가 기준 -> key와 비교)들 사이에 관계를 파악한다.

- Encoder & Decoder with Multi-Head Attention Block
- Positional Encoding (with `nn.Embedding`)
- Tri-Mask for target source to prevent cheating in DECODER
- Make padding value -inf in energy so it has 0 attention score before getting attention
- Neural Machine Translation

![Transformer](../docs/transformer.png)

