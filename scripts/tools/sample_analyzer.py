'''
dict_keys(['agent_history_positions', 'agent_history_positions_mask', 'agent_history_speed', 
'agent_future_positions', 'agent_future_positions_mask', 
'agent_type', 'agent_is_target', 'agent_heading',
'map_lane_positions', 'map_lane_positions_mask', 
'map_is_intersection', 'map_turn_direction', 'map_traffic_control', 
'origin', 'theta']) 
'''

import torch

# 加载样本
sample = torch.load('/root/vc/data/train/processed_dense/114908.pt')

# 查看所有 keys
#print(sample.keys())

# 查看形状
for key, val in sample.items():
    if isinstance(val, torch.Tensor):
        print(f"{key}: {val.shape}")

# 查看有效数量
has_history = sample['agent_history_positions_mask'].any(dim=1)  # [64]
num_history_agents = has_history.sum().item()
has_future = sample['agent_future_positions_mask'].any(dim=1)  # [64]
num_future_agents = has_future.sum().item()
print("历史 Agent:", num_history_agents)
print("未来 Agent:", num_future_agents)
print("T=19 Agent:", sample['agent_history_positions_mask'][:, -1].sum())
print("Lane:", sample['map_lane_positions_mask'][:, 0].sum())

agent_history_positions = sample['agent_history_positions'] # shape [64, 20, 2]
#print(f"agent_history_positions:{agent_history_positions[1]}")

agent_history_positions_mask = sample['agent_history_positions_mask']   # shape [64, 20]
#print(f"agent_history_positions_mask:{agent_history_positions_mask[1]}")

agent_history_speed = sample['agent_history_speed'] # shape [64, 20, 2]
#print(f"agent_history_speed:{agent_history_speed[0]}")

agent_future_positions = sample['agent_future_positions']   # shape [64, 30, 2]
#print(f"agent_future_positions:{agent_future_positions[0]}")

agent_future_positions_mask = sample['agent_future_positions_mask'] # shape [64, 30]
#print(f"agent_future_positions_mask:{agent_future_positions_mask[0]}")

agent_type = sample['agent_type']   # shape [64]
#print(f"agent_type:{agent_type}")

agent_is_target = sample['agent_is_target'] # shape [64]
#print(f"agent_is_target:{agent_is_target}")

agent_heading = sample['agent_heading'] # shape [64]
print(f"agent_heading:{agent_heading}")

map_lane_positions = sample['map_lane_positions']   # shap [256, 10, 2]e
#print(f"map_lane_positions:{map_lane_positions}")

map_lane_positions_mask = sample['map_lane_positions_mask'] # shape [256, 10]
#print(f"map_lane_positions_mask:{map_lane_positions_mask}")

map_is_intersection = sample['map_is_intersection'] # shape [256]
#print(f"map_is_intersection:{map_is_intersection}")

map_turn_direction = sample['map_turn_direction']   # shape [256]
#print(f"map_turn_direction:{map_turn_direction}")

map_traffic_control = sample['map_traffic_control'] # shape [256]
#print(f"map_traffic_control:{map_traffic_control}")


origin = sample['origin']
print(f"origin:{origin}")

theta = sample['theta']
print(f"theta:{theta}")