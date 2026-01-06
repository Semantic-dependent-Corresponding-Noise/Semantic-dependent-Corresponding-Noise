#!/bin/bash

# NPC - Entity Referential Error
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.2 --num_anns 5 --dataset_root dataset/Entity_Referential_Error_noise_MSCOCO --dataset coco --checkpoint_path MSCOCO_models/NPC/Entity_Referential_Error_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.4 --num_anns 5 --dataset_root dataset/Entity_Referential_Error_noise_MSCOCO --dataset coco --checkpoint_path MSCOCO_models/NPC/Entity_Referential_Error_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.6 --num_anns 5 --dataset_root dataset/Entity_Referential_Error_noise_MSCOCO --dataset coco --checkpoint_path MSCOCO_models/NPC/Entity_Referential_Error_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 1.0 --num_anns 5 --dataset_root dataset/Entity_Referential_Error_noise_MSCOCO --dataset coco --checkpoint_path MSCOCO_models/NPC/Entity_Referential_Error_models

# NPC - Object Omission
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.2 --num_anns 5 --dataset_root dataset/Object_Omission_noise_MSCOCO --dataset coco --checkpoint_path MSCOCO_models/NPC/Object_Omission_noise_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.4 --num_anns 5 --dataset_root dataset/Object_Omission_noise_MSCOCO --dataset coco --checkpoint_path MSCOCO_models/NPC/Object_Omission_noise_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.6 --num_anns 5 --dataset_root dataset/Object_Omission_noise_MSCOCO --dataset coco --checkpoint_path MSCOCO_models/NPC/Object_Omission_noise_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 1.0 --num_anns 5 --dataset_root dataset/Object_Omission_noise_MSCOCO --dataset coco --checkpoint_path MSCOCO_models/NPC/Object_Omission_noise_models

# NPC - Short Description
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.2 --num_anns 5 --dataset_root dataset/Short_Description_noise_MSCOCO --dataset coco --checkpoint_path MSCOCO_models/NPC/Short_Description_noise_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.4 --num_anns 5 --dataset_root dataset/Short_Description_noise_MSCOCO --dataset coco --checkpoint_path MSCOCO_models/NPC/Short_Description_noise_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.6 --num_anns 5 --dataset_root dataset/Short_Description_noise_MSCOCO --dataset coco --checkpoint_path MSCOCO_models/NPC/Short_Description_noise_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 1.0 --num_anns 5 --dataset_root dataset/Short_Description_noise_MSCOCO --dataset coco --checkpoint_path MSCOCO_models/NPC/Short_Description_noise_models

# NPC - High level Semantic Confusion
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.2 --num_anns 5 --dataset_root dataset/High_level_Semantic_Confusion_MSCOCO  --dataset coco --checkpoint_path MSCOCO_models/NPC/High_level_Semantic_Confusion_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.4 --num_anns 5 --dataset_root dataset/High_level_Semantic_Confusion_MSCOCO  --dataset coco --checkpoint_path MSCOCO_models/NPC/High_level_Semantic_Confusion_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 0.6 --num_anns 5 --dataset_root dataset/High_level_Semantic_Confusion_MSCOCO  --dataset coco --checkpoint_path MSCOCO_models/NPC/High_level_Semantic_Confusion_models
python main_NPC.py --batch_size 196 --epochs 5 --lr 2e-7 --vision_model ViT-B/32 --noise_ratio 1.0 --num_anns 5 --dataset_root dataset/High_level_Semantic_Confusion_MSCOCO  --dataset coco --checkpoint_path MSCOCO_models/NPC/High_level_Semantic_Confusion_models