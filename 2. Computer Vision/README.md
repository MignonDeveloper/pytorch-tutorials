

## 1. Object Detection

### (1) Intersection over Union
> 모델이 Bounding Box를 얼마나 정확하게 판별했는지를 평가하기 위한 Metric
 - Origin of coordinate is on top-left
 - Intersection 
    - top-left of Intersection: `X1` is max of (box1[0], box2[0]), `Y1` is max of (box1[1], box2[1])
    - bottom-right of Intersection: `X2` is min of (box1[2], box2[2]), `Y2` is min of (box1[3], box2[3])
 - Union
    - Area_A + Area_B - Intersection
 - `0.5`: "decent", `0.7`: "prettey good", `0.9`: "almost perfect"
![IoU](../docs/IOU.png)



### (2) Seq2Seq with Attention
> Seq2Seq의 Encoder와 Decoder를 통해 학습을 할 때, 입력 Sequence의 길이가 길어지면 Deocder로 전달하는 Hidden & Cell state에서 병목현상이 일어날 수 밖에 없다. 또한 Gradient가 첫번째 Layer까지 도달하기 어려운 문제 (Long Term Dependency)를 해결하고 입력으로 주어진 정보들과의 유사도를 계산해서 필요한 정보를 얻기위해 Attention 구조를 사용
 - Encoder & Decoder with LSTM + Attention Layer
 - Use all hidden states in Encoder for caculate simialarity with Deocder's each hidden state (Dot Attention, Concat Attention)
 - Solve Long Term Dependency problem -> More shoter way for gradient to get in encoder
 - Neural Machine Translation
 - Teacher Forcing
![Seq2Seq with Attention](../docs/seq2seqwithAttention.png)

