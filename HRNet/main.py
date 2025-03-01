import os
import torch
from yacs.config import CfgNode as CN

# Assuming HR_config.py and HR_net.py are in the same directory as main.py
from HR_config import MODEL_CONFIGS
from HRnet import get_hrnet_model

def load_config(config_name):
    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.EXTRA = MODEL_CONFIGS[config_name]
    cfg.MODEL.ALIGN_CORNERS = True
    cfg.MODEL.INIT_WEIGHTS = True
    cfg.MODEL.PRETRAINED = ''  # Provide the correct path or leave empty if not available
    cfg.MODEL.OCR = CN()
    cfg.MODEL.OCR.MID_CHANNELS = 512
    cfg.MODEL.OCR.KEY_CHANNELS = 256
    cfg.DATASET = CN()
    cfg.DATASET.NUM_CLASSES = 9  # Update this number according to your dataset
    return cfg

def main():
    config_name = 'hrnet48'  # Change to 'hrnet18' or 'hrnet32' if needed
    input_channels = 16  # Update this to 8, 16, or 24 based on your dataset
    
    # Load configuration
    cfg = load_config(config_name)
    
    # Initialize model
    model = get_hrnet_model(cfg, input_channels)
    
    # Print model summary
    print(model)
    
    # Example: Test the model with random input
    dummy_input = torch.randn(128, input_channels, 64, 64)
    output = model(dummy_input)
    
    print("Output Tensor Shape:", output.shape)

if __name__ == "__main__":
    main()
