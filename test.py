from model import GATMinGRU, encode_data, decode_event, batch_encode, decode_batch
import json
import torch

# load model 
model = GATMinGRU(input_size=166, hidden_size=512, event_embedding_size=16, gat_heads=4)
checkpoint = torch.load("./model.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# test
filename = "./data/"
with open("./data/test_data", "r") as f:
    data_batch = json.load(f)

# Encode the input data
batch_features, batch_edge_index, (batch_event_targets, batch_event_indices, batch_time_targets), _ = batch_encode(data_batch)

batch_size = batch_features.shape[0]
edge_index = batch_edge_index.coalesce().indices()

# Forward pass through the model
with torch.no_grad():
    candidate_embeddings, time_pred = model(batch_features, edge_index,h_prev1=torch.zeros(batch_size, 512), h_prev2=torch.zeros(batch_size, 512))

# Decode the output
decoded_event_index = decode_event(candidate_embeddings[0], model.event_embedding_layer)

print(decoded_event_index)