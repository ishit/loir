import torch

def resolve_collisions(pos, collision_threshold=10, canvas_width=1):
    scaled_pos = pos * (canvas_width / 4)
    max_iterations = 50
    for _ in range(max_iterations):
        # Compute pairwise distances between all points
        distances = torch.cdist(scaled_pos, scaled_pos)
        
        # Create a mask for colliding pairs (excluding self-distances)
        collision_mask = (distances < collision_threshold) & (distances > 0)
        
        if not collision_mask.any():
            break  # No collisions, exit the loop
        
        # Calculate repulsion vectors
        repulsion = scaled_pos.unsqueeze(1) - scaled_pos.unsqueeze(0)
        repulsion_distances = distances.unsqueeze(2).repeat(1, 1, 2)
        repulsion = torch.where(collision_mask.unsqueeze(2), repulsion / repulsion_distances, torch.zeros_like(repulsion))
        
        # Sum repulsion vectors for each point
        total_repulsion = repulsion.sum(dim=1)
        
        # Apply repulsion
        scaled_pos += total_repulsion * 0.5  # Adjust the factor to control repulsion strength
        
        # Clip positions to stay within canvas bounds
        scaled_pos.clamp_(-canvas_width / 2, canvas_width / 2)
    
    # Update pos based on the resolved scaled_pos
    pos = scaled_pos / (canvas_width / 4)
    return pos
