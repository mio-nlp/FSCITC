import torch
from transformers import BertModel, BertConfig
from fvcore.nn import FlopCountAnalysis, flop_count_table
from time import time

# 定义一个类来包装前6层或后6层的BERT模型
class PartialBertModel(torch.nn.Module):
    def __init__(self, embeddings, layers):
        super(PartialBertModel, self).__init__()
        self.embeddings = embeddings
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, input_ids):
        hidden_states = self.embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)[0]
        return hidden_states


# 初始化BERT base模型
config = BertConfig.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', config=config)

# 定义输入大小
input_ids = torch.ones((1, config.max_position_embeddings), dtype=torch.long)  # (batch_size, sequence_length)
input_ids2 = torch.ones((6, config.max_position_embeddings), dtype=torch.long)  # (batch_size, sequence_length)

# 获取前6层模型
front_layers = model.encoder.layer[:6]
front_model = PartialBertModel(model.embeddings, front_layers)

# 获取后6层模型
back_layers = model.encoder.layer[6:]
back_model = PartialBertModel(model.embeddings, back_layers)

with torch.no_grad():
    device = 'cuda'
    front_model = front_model.to(device)
    back_model = back_model.to(device)
    input_ids2 = input_ids2.to(device)
    input_ids = input_ids.to(device)
    start_time = time()
    for i in range(10):
        a = front_model(input_ids)
        b = front_model(input_ids2)
    end_time = time()
    print('MM时间：{}'.format(end_time-start_time))
    model = model.to(device)
    start_time = time()
    for i in range(10):
        a = model(input_ids)
    end_time = time()
    print('单个时间：{}'.format(end_time-start_time))


# # 计算前6层的FLOPs
# front_flops = FlopCountAnalysis(front_model, input_ids)
# print(f"FLOPs for the first 6 layers: {front_flops.total()}")
# print(flop_count_table(front_flops))
#
# # 计算后6层的FLOPs
# back_flops = FlopCountAnalysis(back_model, input_ids)
# print(f"FLOPs for the last 6 layers: {back_flops.total()}")
# print(flop_count_table(back_flops))
