import os
import sys

# Get the base directory
base_path = os.path.join(os.getcwd(), "Grounded_SAM_2")
repo_path = os.path.join(base_path, "grounding_dino")

# 1. Add the repo root so 'import groundingdino' works for YOU
if repo_path not in sys.path:
    sys.path.append(repo_path)

# 2. Add the folder ABOVE the repo so 'import grounding_dino' works for the INTERNAL scripts
if base_path not in sys.path:
    sys.path.append(base_path)

# Now the imports will stop screaming
from groundingdino.util.inference import load_model

def load_grounding_dino(device="cuda"):
    model = load_model(
        model_config_path="Grounded_SAM_2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_checkpoint_path="Grounded_SAM_2/gdino_checkpoints/groundingdino_swint_ogc.pth",
        device=device
    )
    return model