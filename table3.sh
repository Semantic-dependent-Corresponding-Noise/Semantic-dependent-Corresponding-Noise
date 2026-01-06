#!/bin/bash

# CLIP - Object Omission
python main_CLIP.py --batch_size 256 --epochs 5 --lr 5e-7 --vision_model ViT-B/32 --noise_ratio 0.2 --num_anns 5 --dataset_root dataset/Object_Omission_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/clip/Object_Omission_noise_5error_models
python main_CLIP.py --batch_size 256 --epochs 5 --lr 5e-7 --vision_model ViT-B/32 --noise_ratio 0.4 --num_anns 5 --dataset_root dataset/Object_Omission_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/clip/Object_Omission_noise_5error_models
python main_CLIP.py --batch_size 256 --epochs 5 --lr 5e-7 --vision_model ViT-B/32 --noise_ratio 0.6 --num_anns 5 --dataset_root dataset/Object_Omission_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/clip/Object_Omission_noise_5error_models
python main_CLIP.py --batch_size 256 --epochs 5 --lr 5e-7 --vision_model ViT-B/32 --noise_ratio 1.0 --num_anns 5 --dataset_root dataset/Object_Omission_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/clip/Object_Omission_noise_5error_models

# CLIP - Entity Referential Error
python main_CLIP.py --batch_size 256 --epochs 5 --lr 5e-7 --vision_model ViT-B/32 --noise_ratio 0.2 --num_anns 5 --dataset_root dataset/Entity_Referential_Error_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/clip/Entity_Referential_Error_5error_models
python main_CLIP.py --batch_size 256 --epochs 5 --lr 5e-7 --vision_model ViT-B/32 --noise_ratio 0.4 --num_anns 5 --dataset_root dataset/Entity_Referential_Error_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/clip/Entity_Referential_Error_5error_models
python main_CLIP.py --batch_size 256 --epochs 5 --lr 5e-7 --vision_model ViT-B/32 --noise_ratio 0.6 --num_anns 5 --dataset_root dataset/Entity_Referential_Error_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/clip/Entity_Referential_Error_5error_models
python main_CLIP.py --batch_size 256 --epochs 5 --lr 5e-7 --vision_model ViT-B/32 --noise_ratio 1.0 --num_anns 5 --dataset_root dataset/Entity_Referential_Error_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/clip/Entity_Referential_Error_5error_models

# CLIP - Short Description
python main_CLIP.py --batch_size 256 --epochs 5 --lr 5e-7 --vision_model ViT-B/32 --noise_ratio 0.2 --num_anns 5 --dataset_root dataset/Short_Description_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/clip/Short_Description_noise_5error_models
python main_CLIP.py --batch_size 256 --epochs 5 --lr 5e-7 --vision_model ViT-B/32 --noise_ratio 0.4 --num_anns 5 --dataset_root dataset/Short_Description_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/clip/Short_Description_noise_5error_models
python main_CLIP.py --batch_size 256 --epochs 5 --lr 5e-7 --vision_model ViT-B/32 --noise_ratio 0.6 --num_anns 5 --dataset_root dataset/Short_Description_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/clip/Short_Description_noise_5error_models
python main_CLIP.py --batch_size 256 --epochs 5 --lr 5e-7 --vision_model ViT-B/32 --noise_ratio 1.0 --num_anns 5 --dataset_root dataset/Short_Description_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/clip/Short_Description_noise_5error_models

# CLIP - High level Semantic Confusion
python main_CLIP.py --batch_size 256 --epochs 5 --lr 5e-7 --vision_model ViT-B/32 --noise_ratio 0.2 --num_anns 5 --dataset_root dataset/High_level_Semantic_Confusion_5error_f30k --dataset f30k --checkpoint_path f30k_models/clip/High_level_Semantic_Confusion_5error_models
python main_CLIP.py --batch_size 256 --epochs 5 --lr 5e-7 --vision_model ViT-B/32 --noise_ratio 0.4 --num_anns 5 --dataset_root dataset/High_level_Semantic_Confusion_5error_f30k --dataset f30k --checkpoint_path f30k_models/clip/High_level_Semantic_Confusion_5error_models
python main_CLIP.py --batch_size 256 --epochs 5 --lr 5e-7 --vision_model ViT-B/32 --noise_ratio 0.6 --num_anns 5 --dataset_root dataset/High_level_Semantic_Confusion_5error_f30k --dataset f30k --checkpoint_path f30k_models/clip/High_level_Semantic_Confusion_5error_models
python main_CLIP.py --batch_size 256 --epochs 5 --lr 5e-7 --vision_model ViT-B/32 --noise_ratio 1.0 --num_anns 5 --dataset_root dataset/High_level_Semantic_Confusion_5error_f30k --dataset f30k --checkpoint_path f30k_models/clip/High_level_Semantic_Confusion_5error_models

# NPC - Entity Referential Error
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.2 --num_anns 5 --dataset_root dataset/Entity_Referential_Error_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/NPC/Entity_Referential_Error_5error_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.4 --num_anns 5 --dataset_root dataset/Entity_Referential_Error_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/NPC/Entity_Referential_Error_5error_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.6 --num_anns 5 --dataset_root dataset/Entity_Referential_Error_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/NPC/Entity_Referential_Error_5error_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 1.0 --num_anns 5 --dataset_root dataset/Entity_Referential_Error_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/NPC/Entity_Referential_Error_5error_models

# NPC - Object Omission
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.2 --num_anns 5 --dataset_root dataset/Object_Omission_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/NPC/Object_Omission_noise_5error_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.4 --num_anns 5 --dataset_root dataset/Object_Omission_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/NPC/Object_Omission_noise_5error_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.6 --num_anns 5 --dataset_root dataset/Object_Omission_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/NPC/Object_Omission_noise_5error_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 1.0 --num_anns 5 --dataset_root dataset/Object_Omission_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/NPC/Object_Omission_noise_5error_models

# NPC - Short Description
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.2 --num_anns 5 --dataset_root dataset/Short_Description_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/NPC/Short_Description_noise_5error_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.4 --num_anns 5 --dataset_root dataset/Short_Description_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/NPC/Short_Description_noise_5error_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.6 --num_anns 5 --dataset_root dataset/Short_Description_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/NPC/Short_Description_noise_5error_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 1.0 --num_anns 5 --dataset_root dataset/Short_Description_noise_5error_f30k --dataset f30k --checkpoint_path f30k_models/NPC/Short_Description_noise_5error_models

# NPC - High level Semantic Confusion
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.2 --num_anns 5 --dataset_root dataset/High_level_Semantic_Confusion_5error_f30k  --dataset f30k --checkpoint_path f30k_models/NPC/High_level_Semantic_Confusion_5error_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.4 --num_anns 5 --dataset_root dataset/High_level_Semantic_Confusion_5error_f30k  --dataset f30k --checkpoint_path f30k_models/NPC/High_level_Semantic_Confusion_5error_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.6 --num_anns 5 --dataset_root dataset/High_level_Semantic_Confusion_5error_f30k  --dataset f30k --checkpoint_path f30k_models/NPC/High_level_Semantic_Confusion_5error_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 1.0 --num_anns 5 --dataset_root dataset/High_level_Semantic_Confusion_5error_f30k  --dataset f30k --checkpoint_path f30k_models/NPC/High_level_Semantic_Confusion_5error_models