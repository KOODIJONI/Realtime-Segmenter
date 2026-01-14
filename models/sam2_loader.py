import os
from hydra import initialize_config_dir, compose
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

def load_sam2_models():
    # 1. Define absolute paths
    # Note: No leading slash at the start of Grounded_SAM_2
    base_path = os.path.join(os.getcwd(), "Grounded_SAM_2")
    config_path = os.path.join(base_path, "sam2/configs") # Points to the folder containing 'sam2.1'
    checkpoint = os.path.join(base_path, "checkpoints/sam2.1_hiera_large.pt")
    
    # 2. Hydra Logic: Initialize the config directory
    # 'config_name' must be the relative path from the 'configs' folder
    # so: "sam2.1/sam2.1_hiera_l" (usually without the .yaml)
    
    # Clean up Hydra if it was already initialized
    from hydra.core.global_hydra import GlobalHydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_path):
        # SAM2 build functions internally call 'compose'
        # They expect the config name relative to the config_dir
        model_cfg = "sam2.1/sam2.1_hiera_l.yaml"
        
        sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")
        image_predictor = SAM2ImagePredictor(sam2_model)
        video_predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda")

    return sam2_model, image_predictor, video_predictor