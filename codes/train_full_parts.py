


    print("Here only the normal images from train dataset are used for memory bank construction")
    print(len(train_dataset.images))
    # ============================================
    # MEMORY BANK CONSTRUCTION (Normal Images Only)
    # ============================================
    
    print("Building memory bank from normal training images...")
    
    # Get indices of normal images (label == 0)
    normal_indices = [i for i, (_, label, _) in enumerate(train_dataset) if label == 0]
    
    print(f"Found {len(normal_indices)} normal images out of {len(train_dataset)} total")
    
    # Create subset of normal images
    from torch.utils.data import Subset
    
    support_dataset = Subset(train_dataset, normal_indices)
    support_loader = torch.utils.data.DataLoader(
        support_dataset, 
        batch_size=1, 
        shuffle=False,  # ✅ Don't shuffle memory bank
        **kwargs
    )
    
    print(f"Memory bank loader created with {len(support_dataset)} normal samples")



        # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()


    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_biomedclip_prompt_ensemble1(clip_model, REAL_NAME[args.obj], device)
    print("Text features shape:", text_features.shape)  
