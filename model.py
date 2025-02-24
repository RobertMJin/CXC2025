
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_add
import numpy as np
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import math
import random

# ---- config ----
embedding_dim = 16      # Dimension for embeddings
secondary_embedding_dim = 8
user_table_size = 20  # Initial size of number of users
event_table_size = 50  # Initial size of number of event types
region_table_size = 1000
country_table_size = 150
max_event_history = 5

# --- Normalize Data ---
min_session_id, max_session_id = 1e12, 1e13
min_time, max_time = 1, 31556926

# ---- Initialize Embedding Layer ---
user_embedding_layer = nn.Embedding(user_table_size, embedding_dim)
event_embedding_layer = nn.Embedding(event_table_size, embedding_dim, padding_idx=0)  # Assume max 500 unique event IDs
region_embedding_layer = nn.Embedding(region_table_size, secondary_embedding_dim, padding_idx=0)
country_embedding_layer = nn.Embedding(country_table_size, secondary_embedding_dim, padding_idx=0)

# -- Initialize OneHotEncoders ---
# dim = 3
platform_categories = ["iOS", "Android", "Web"]
platform_encoder = OneHotEncoder(categories=[platform_categories], sparse_output=False)
platform_encoder.fit(np.array(platform_categories).reshape(-1, 1))

# dim = 10
device_type_categories = ["Unknown", "Android", "Apple iPad", "Apple iPhone", "Chromium OS", "Google Nexus 5", "Linux", "Mac", "Ubuntu", "Windows"]
dt_encoder = OneHotEncoder(categories=[device_type_categories], sparse_output=False)
dt_encoder.fit(np.array(device_type_categories).reshape(-1, 1))

# dim = 13
os_name_categories = ["Unknown", "android", "Chrome", "Chrome Headless", "Chrome Mobile", "Chrome Mobile iOS", "Edge", "Edge Mobile", "Firefox", "HeadlessChrome", "Mobile Safari", "Opera", "Safari"]
os_name_encoder = OneHotEncoder(categories=[os_name_categories], sparse_output=False)
os_name_encoder.fit(np.array(os_name_categories).reshape(-1, 1))

# dim = 4
language_categories = ["Unknown", "English", "Spanish", "Polish"]
language_encoder = OneHotEncoder(categories=[language_categories], sparse_output=False)
language_encoder.fit(np.array(language_categories).reshape(-1, 1))

# dim = 14
device_family_categories = ["Unknown","Android", "Apple iPad", "Apple iPhone", "Chrome OS", "Chromium OS", "Google Nexus Phone", "iOS", "K", "Linux", "Mac", "Mac OS X", "Ubuntu", "Windows"]
device_family_encoder = OneHotEncoder(categories=[device_family_categories], sparse_output=False)
device_family_encoder.fit(np.array(device_family_categories).reshape(-1, 1))

def pad_event_history(event_history, max_length):
    """Pad the event history to a fixed length."""
    padded_event_history = []
    for _ in range(max_length - len(event_history)):
        padded_event_history.append((0, 0))
    padded_event_history.extend(event_history)
    return padded_event_history

def hash_user_id(user_id, table_size):
    """Hash the user ID to an index within the embedding table size."""
    return abs(hash(user_id)) % table_size

def normalize_session_id(session_id):
  normalized_id = (session_id - min_session_id) / (max_session_id - min_session_id)
  return float(format(normalized_id, '.32f'))

def normalize_event_time(event_time):
  norm_event_time = event_time % max_time
  return float(format(norm_event_time, '.32f'))

def normalize_event_id(event_id):
  norm_event_id = event_id % event_table_size
  return norm_event_id

def log_tsl(tsl):
  if tsl >= 0 and tsl < 1:
    return 0
  return float(math.log10(tsl)/10)

def encode_data(data_point, mode=0):
    """Encodes a single data point according to the provided schema."""
    """
    Schema:
      {
        # These are feature variables
        "region": string,
        "country": string,
        "language": string, # one hot encoded
        "device_family": string,  # one hot encoded
        "device_type": string,  # one hot encoded
        "os_name": string,  # one hot encoded
        "platform": string, # one hot encoded
        "amplitude_id": int,
        "session_id": int,

        "event_type": string,
        "event_time": string in ISO,

        # Feature Dictionary:
        "dict_event_type": int,
        "dict_pet1": int, # optional
        "dict_pet2": int, # optional
        "dict_pet3": int, # optional
        "dict_pet4": int, # optional
        "dict_next_et": int,
        "dict_region": int,
        "dict_country": int,
        "dict_device_family": int,

        "time_since_last": datetime,  #optional

        "prev_1_event_type": string,    #optional
        "time_since_last_1": int,       #optional
        "prev_2_event_type": string,    #optional
        "time_since_last_2": int,       #optional
        "prev_3_event_type": string,    #optional
        "time_since_last_3": int,       #optional
        "prev_4_event_type": string,    #optional
        "time_since_last_4": int,       #optional

        "average_session_time": float, #optional
        "total_session_time": float,   #optional
        "user_retention_30": float,    #optional

        # Label Variables
        "next_event_type": string,
        "time_to_next_event": datetime,
      }
    """
    # --- One-Hot Encoding --- dim = 3 + 10 + 13 + 4 + 14 = 44
    platform = data_point["platform"]
    platform_reshaped = np.array([platform]).reshape(-1, 1)
    encoded_platform = torch.tensor(platform_encoder.transform(platform_reshaped), dtype=torch.float32)

    device_type = data_point["device_type"]
    if device_type not in device_type_categories:
      device_type = "Unknown"
    device_type_reshaped = np.array([device_type]).reshape(-1, 1)
    encoded_device_type = torch.tensor(dt_encoder.transform(device_type_reshaped), dtype=torch.float32)

    os_name = data_point["os_name"]
    if os_name not in os_name_categories:
      os_name = "Unknown"
    os_name_reshaped = np.array([os_name]).reshape(-1, 1)
    encoded_os_name = torch.tensor(os_name_encoder.transform(os_name_reshaped), dtype=torch.float32)

    language = data_point["language"]
    if language not in language_categories:
      language = "Unknown"
    language_reshaped = np.array([language]).reshape(-1, 1)
    encoded_language = torch.tensor(language_encoder.transform(language_reshaped), dtype=torch.float32)

    device_family = data_point["device_family"]
    if device_family not in device_family_categories:
      device_family = "Unknown"
    device_family_reshaped = np.array([device_family]).reshape(-1, 1)
    encoded_device_family = torch.tensor(device_family_encoder.transform(device_family_reshaped), dtype=torch.float32)

    # --- Hashing User ID & User ID Embed --- dim = 16
    user_id = data_point["amplitude_id"]
    hashed_user_id = hash_user_id(user_id, user_table_size)
    user_embed = user_embedding_layer(torch.tensor(hashed_user_id))

    # ----- Region and Country Embeds ---- dim = 8+8 = 16
    region = data_point["region"]
    country = data_point["country"]
    region_id = data_point["dict_region"]
    country_id = data_point["dict_country"]
    if region == None:
      region_id = 0
    if country == None:
      region_id = 0
    region_embed = region_embedding_layer(torch.tensor(region_id, dtype=torch.long))
    country_embed = country_embedding_layer(torch.tensor(country_id, dtype=torch.long))

    # --- Event History --- # dim = 16 * 5 = 80
    event_history = []
    for i in range(max_event_history - 1, 0, -1):
        if data_point[f"dict_pet{i}"]!=None:
          event_history.append((data_point[f"dict_pet{i}"], data_point[f"time_since_last_{i}"] if data_point.get(f"time_since_last_{i}") is not None else 0))
    event_history.append((data_point["dict_event_type"], data_point[f"time_since_last"] if data_point.get(f"time_since_last_{i}") is not None else 0))
    event_history = pad_event_history(event_history, max_event_history) # [5, 2] array

    # ---- Event_Type embeds ----
    event_type_embeddings = []
    for dict_et, _ in event_history:
      event_type_tensor = torch.tensor(normalize_event_id(dict_et), dtype=torch.long)
      event_embedding = event_embedding_layer(event_type_tensor)
      event_type_embeddings.append(event_embedding)

    # stacking events to maintain temporal structure
    event_type_embeddings = torch.stack(event_type_embeddings) # size (5, 16)
    time_feature = torch.tensor(([log_tsl(tsl) for _, tsl in event_history]), dtype=torch.float32) # dim 5

    # --- Numerical Features ---  # dim = 5
    # Normalizing numerical features
    event_time_str = data_point["event_time"].split(".")[0]
    event_time = datetime.strptime(event_time_str, "%Y-%m-%dT%H:%M:%S").timestamp()
    event_time = normalize_event_time(event_time) #normalized between (0, 1) as % from epoch to max_event_time

    session_id = data_point["session_id"] #normalized between (0, 1) as % from min_session to max_session
    session_id = normalize_session_id(session_id)

    numerical_features = torch.tensor([
        data_point.get("average_session_time", 0),
        data_point.get("total_session_time", 0),
        data_point.get("user_retention_30", 0),
        session_id,
        event_time
    ], dtype=torch.float32)


    # --- Combine All Features ---
    features = torch.cat([user_embed, region_embed, country_embed, event_type_embeddings.flatten(), time_feature.flatten(), numerical_features, encoded_platform.flatten(), encoded_device_family.flatten(), encoded_device_type.flatten(), encoded_os_name.flatten(), encoded_language.flatten()])
    if mode==1:
        return features
    
    # Labels
    # need to represent the next event type as a vector embedding that is in the same space as event_embedding
    target_event_index = data_point["dict_next_et"]
    if data_point["dict_next_et"] == None:
      target_event_index = 0
    target_event_embedding = event_embedding_layer(torch.tensor(normalize_event_id(target_event_index), dtype=torch.long))

    if data_point["time_to_next_event"] == None:
      target_time = torch.tensor(log_tsl(0), dtype=torch.float32)
    else:
      target_time = torch.tensor(log_tsl(data_point["time_to_next_event"]), dtype=torch.float32)
    return features, (target_event_embedding, target_event_index, target_time)

def create_batch_edges(batch_global_edge_list, lg_event_mapping, edge_counts, batch_size, min_edges_per_node=2):
    batch_edge_indices = []
    batch_edge_weights = []

    total_edge_combinations = sum(
        len(lg_event_mapping.get(globalsrc, [])) * len(lg_event_mapping.get(globaldst, []))
        for globaldst, globalsrc in batch_global_edge_list
    )

    max_edges = 10 * batch_size
    if total_edge_combinations < max_edges:
        # Keep all edges if below threshold
        for (globaldst, globalsrc) in batch_global_edge_list:
            weight = edge_counts[(globaldst, globalsrc)]
            for localdst in lg_event_mapping.get(globaldst, []):
                for localsrc in lg_event_mapping.get(globalsrc, []):
                    batch_edge_indices.append((localdst, localsrc))
                    batch_edge_weights.append(weight)
    else:
        # Use proportional sampling if exceeding max_edges
        edge_sampling_probs = {
            edge: (len(lg_event_mapping.get(edge[1], [])) * len(lg_event_mapping.get(edge[0], []))) / total_edge_combinations
            for edge in batch_global_edge_list
        }

        for (globaldst, globalsrc) in batch_global_edge_list:
            weight = edge_counts[(globaldst, globalsrc)]
            num_samples = max(int(edge_sampling_probs[(globaldst, globalsrc)] * max_edges), min_edges_per_node)

            local_srcs = lg_event_mapping.get(globalsrc, [])
            local_dsts = lg_event_mapping.get(globaldst, [])

            if local_srcs and local_dsts:
                sampled_srcs = random.choices(local_srcs, k=min(num_samples, len(local_srcs) * len(local_dsts)))
                sampled_dsts = random.choices(local_dsts, k=len(sampled_srcs))

                batch_edge_indices.extend(zip(sampled_dsts, sampled_srcs))
                batch_edge_weights.extend([weight] * len(sampled_srcs))

    # Convert edge indices to tensors
    batch_edge_index = torch.tensor(batch_edge_indices, dtype=torch.long).T

    # Normalize edge weights using softmax
    edge_weights_tensor = torch.tensor(batch_edge_weights, dtype=torch.float32)
    softmax_weights = torch.nn.functional.softmax(edge_weights_tensor, dim=0)

    return batch_edge_index, softmax_weights

def batch_encode(data_batch):
    batch_features = []
    batch_event_targets = []
    batch_event_indices = []
    batch_time_targets = []

    batch_global_edge_list = []
    edge_counts = {}
    lg_event_mapping = {}

    local_id = 0
    for data_point in data_batch:
        features, (target_event, target_event_index, target_time) = encode_data(data_point)
        batch_features.append(features)
        batch_event_targets.append(target_event)
        batch_event_indices.append(target_event_index)
        batch_time_targets.append(target_time)

        curr_event = data_point["dict_event_type"]
        for i in range(1, 5):
            if curr_event not in lg_event_mapping:
                lg_event_mapping[curr_event] = []
            lg_event_mapping[curr_event].append(local_id)

            prev_event = 0 if data_point[f"dict_pet{i}"] is None else data_point.get(f"dict_pet{i}", 0) # global id
            if prev_event >= 1 or curr_event >= 1:
                edge = (prev_event, curr_event)
                if edge not in edge_counts:
                    edge_counts[edge] = 0
                    batch_global_edge_list.append(edge)
                edge_counts[edge] += 1
            curr_event = prev_event
        local_id += 1

    # Use the optimized function to get batch edges
    batch_edge_index, batch_edge_weights = create_batch_edges(
        batch_global_edge_list, lg_event_mapping, edge_counts, batch_size=len(data_batch)
    )

    batch_features = torch.stack(batch_features) # local id of [0, 1, 2, 3 ...  batch_size]
    batch_event_targets = torch.stack(batch_event_targets)
    batch_time_targets = torch.stack(batch_time_targets)
    batch_event_indices = torch.tensor(batch_event_indices, dtype=torch.long)

    # Create sparse tensor for edge weights
    row, col = batch_edge_index
    indices = torch.stack([row, col])
    batch_edge_index_sparse = torch.sparse_coo_tensor(indices, batch_edge_weights, (batch_features.shape[0], batch_features.shape[0]))

    return batch_features, batch_edge_index_sparse, (batch_event_targets, batch_event_indices, batch_time_targets), lg_event_mapping



class MinGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MinGRU, self).__init__()
        self.hidden_size = hidden_size
        # Linear layers for update gate and candidate hidden state
        self.linear_z = nn.Linear(input_size, hidden_size)  # Update gate
        self.linear_h = nn.Linear(input_size, hidden_size)  # Candidate hidden state

    def forward(self, x, h_prev):
        # Compute update gate (z_t)
        z_t = torch.sigmoid(self.linear_z(x))
        # Compute candidate hidden state (h_tilde)
        h_tilde = torch.tanh(self.linear_h(x))

        # Compute new hidden state (h_t)
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        return h_t
    
class SparseGATConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1,
                 dropout=0.6, negative_slope=0.2, concat=True, bias=True):
        """
        in_channels: Dimensionality of input node features.
        out_channels: Dimensionality per attention head.
        heads: Number of attention heads.
        concat: If True, concatenate head outputs; if False, average them.
        dropout: Dropout rate applied to the attention coefficients.
        negative_slope: Slope for the LeakyReLU.
        """
        super(SparseGATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = nn.Dropout(dropout)
        self.negative_slope = negative_slope

        # Linear transformation to project input features into head-specific subspaces.
        self.linear = nn.Linear(in_channels, heads * out_channels, bias=False)
        # Two sets of learnable attention parameters for source and target.
        self.attn_l = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.attn_r = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if bias:
            if concat:
                self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
            else:
                self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier initialization as in the original DGL implementation.
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        """
        x: Node features of shape [N, in_channels].
        edge_index: Tensor of shape [2, E] with source and target node indices.
        """
        N = x.size(0)
        # 1. Linear projection and reshape:
        h = self.linear(x)  # [N, heads * out_channels]
        h = h.view(N, self.heads, self.out_channels)  # [N, heads, out_channels]

        # 2. Compute attention scores for each node:
        a1 = (h * self.attn_l).sum(dim=-1)  # [N, heads]
        a2 = (h * self.attn_r).sum(dim=-1)  # [N, heads]

        # 3. For every edge (i, j), compute:
        src, dst = edge_index  # edge_index shape: [2, E]
        e = a1[src] + a2[dst]  # [E, heads]
        e = F.leaky_relu(e, negative_slope=self.negative_slope)

        # 4. Normalize attention scores per destination node using scatter_softmax:
        alpha = scatter_softmax(e, dst, dim=0)  # [E, heads]
        alpha = self.dropout(alpha)

        # 5. Message passing: weight each source node feature with the attention coefficient.
        alpha = alpha.unsqueeze(-1)  # [E, heads, 1]
        message = h[src] * alpha      # [E, heads, out_channels]

        # 6. Aggregate messages for each destination node:
        out = scatter_add(message, dst, dim=0, dim_size=N)  # [N, heads, out_channels]

        if self.concat:
            # Concatenate heads to get [N, heads*out_channels]
            out = out.view(N, self.heads * self.out_channels)
        else:
            # Or average over the heads.
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out
    
class GATMinGRU(nn.Module):
    def __init__(self, input_size, hidden_size, event_embedding_size, gat_heads=1):
        super(GATMinGRU, self).__init__()
        self.hidden_size = hidden_size
        # GAT layer
        self.event_embedding_layer = nn.Embedding(event_table_size, embedding_dim, padding_idx=0)

        self.gat = SparseGATConv(input_size, hidden_size, heads=gat_heads,
                                 dropout=0.6, negative_slope=0.2, concat=True)
        # When using multiple heads with concatenation, the output dimension becomes hidden_size * gat_heads.
        if gat_heads > 1:
            self.proj = nn.Linear(hidden_size * gat_heads, hidden_size)
        else:
            self.proj = nn.Identity()

        # MinGRU layers (2 hidden layers)
        self.min_gru1 = MinGRU(hidden_size, hidden_size)
        self.min_gru2 = MinGRU(hidden_size, hidden_size)

        #Triple candidate event predictions
        self.event_fc1 = nn.Linear(hidden_size, event_embedding_size)
        self.event_fc2 = nn.Linear(hidden_size, event_embedding_size)
        self.event_fc3 = nn.Linear(hidden_size, event_embedding_size)

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, edge_index, h_prev1, h_prev2):
        # Process graph data with GAT
        x = self.gat(x, edge_index)
        x = self.proj(x)
        # Pass through first MinGRU layer
        h1 = self.min_gru1(x, h_prev1)
        # Pass through second MinGRU layer
        h2 = self.min_gru2(h1, h_prev2)

        # Compute candidate event embeddings from three different heads.
        candidate_embedding1 = self.event_fc1(h2)  # [batch_size, event_embedding_size]
        candidate_embedding2 = self.event_fc2(h2)  # [batch_size, event_embedding_size]
        candidate_embedding3 = self.event_fc3(h2)  # [batch_size, event_embedding_size]

        # Stack candidate embeddings: shape [batch_size, 3, event_embedding_size]
        candidate_events = torch.stack([candidate_embedding1, candidate_embedding2, candidate_embedding3], dim=1)

        # Time prediction: shape [batch_size]
        time_prediction = self.fc(h2).squeeze(1)

        return candidate_events, time_prediction
    

def decode_event(candidate_embedding, event_embedding_layer):
    """
    Given a candidate embedding of shape [E] and an event embedding layer,
    this function returns the index of the event embedding that is closest.
    
    Args:
      candidate_embedding: Tensor of shape [E].
      event_embedding_layer: nn.Embedding whose weight matrix has shape [num_events, E].
      
    Returns:
      event_idx: int, the index of the closest event embedding.
    """
    # Get the event embedding table (detach so gradients are not tracked)
    event_table = event_embedding_layer.weight.detach()  # [num_events, E]
    
    # Compute the Euclidean distances between candidate_embedding and all event embeddings
    distances = torch.norm(event_table - candidate_embedding.unsqueeze(0), p=2, dim=1)  # [num_events]
    
    # Return the index with the minimum distance
    event_idx = torch.argmin(distances).item()
    return event_idx

def decode_batch(candidate_embeddings, event_embedding_layer):
    """
    Decodes a batch of candidate embeddings.
    
    Args:
      candidate_embeddings: Tensor of shape [B, 3, E] (e.g., three candidates per sample)
      event_embedding_layer: The embedding layer for events.
    
    Returns:
      decoded_events: A list (or tensor) of length B containing the predicted event indices.
    """
    batch_size = candidate_embeddings.shape[0]
    decoded_events = []
    for i in range(batch_size):
        # For each sample, select the candidate with the lowest distance to any event embedding
        candidates = candidate_embeddings[i]  # shape [3, E]
        # Compute distances for each candidate
        distances = torch.cdist(candidates.unsqueeze(0), event_embedding_layer.weight.detach(), p=2)
        # distances shape: [1, 3, num_events]
        # For each candidate, get the min distance over all event embeddings
        candidate_min, _ = distances.squeeze(0).min(dim=1)  # shape: [3]
        # Choose the candidate with the minimum min-distance overall
        best_candidate_idx = candidate_min.argmin().item()
        # Now decode that candidate to an event index:
        event_idx = decode_event(candidates[best_candidate_idx], event_embedding_layer)
        decoded_events.append(event_idx)
    return decoded_events