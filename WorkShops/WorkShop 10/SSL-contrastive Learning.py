# ======================================================================
# == Main Execution Block ==============================================
# ======================================================================
if __name__ == "__main__":

    # --- Define file paths for saving models and plots ---
    ssl_encoder_save_path = os.path.join(config["save_dir"], "ssl_encoder_final.pth")
    ssl_full_model_save_path = os.path.join(config["save_dir"], "ssl_full_model_final.pth")
    linear_classifier_save_path = os.path.join(config["save_dir"], "linear_classifier_final.pth")
    augment_vis_save_path = os.path.join(config["save_dir"], "augmentations_visualization.png")
    ssl_loss_save_path = os.path.join(config["save_dir"], "ssl_training_loss.png")
    tsne_save_path = os.path.join(config["save_dir"], "tsne_visualization_final.png")

    # ======================================================
    # == 1. SELF-SUPERVISED PRE-TRAINING (SimCLR) ========
    # ======================================================
    print("\n" + "="*70)
    print(" STEP 1: Self-Supervised Pre-training (SimCLR) ".center(70, "="))
    print("="*70 + "\n")

    start_ssl_time = time.time()

    # --- SSL Data Loading ---
    print("Loading CIFAR-10 dataset for SSL pre-training...")
    # Define the contrastive transformations pipeline
    ssl_transforms_func = get_cifar10_ssl_transforms(img_size=config["image_size"])
    contrastive_transforms_wrapper = ContrastiveTransformations(ssl_transforms_func, n_views=config["n_views"])

    # Load the base training dataset *without* transforms first to allow subsetting and visualization
    base_train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=None # Get PIL images initially
    )

    # Handle optional subset usage for faster demo runs
    if config["use_subset_ssl"] and config["use_subset_ssl"] < len(base_train_dataset):
        print(f"Using a subset of {config['use_subset_ssl']} images for SSL training.")
        subset_indices = list(range(len(base_train_dataset)))
        random.shuffle(subset_indices) # Shuffle indices before selecting subset
        # Create a Subset object pointing to the base dataset with selected indices
        ssl_train_dataset_for_loader = Subset(base_train_dataset, subset_indices[:config['use_subset_ssl']])
        # IMPORTANT: Apply the contrastive transform *to the underlying base dataset*
        # This way, the Subset accesses already transformed data when indexed.
        # We modify the transform attribute of the dataset wrapped by the Subset.
        ssl_train_dataset_for_loader.dataset.transform = contrastive_transforms_wrapper

        # Dataset for visualization (needs PIL access, so use the subset of the non-transformed data)
        vis_dataset_ssl = Subset(base_train_dataset, subset_indices[:config['use_subset_ssl']])
    else:
        print("Using the full CIFAR-10 training set for SSL.")
        # Apply the contrastive transform directly to the dataset if using the full set
        base_train_dataset.transform = contrastive_transforms_wrapper
        ssl_train_dataset_for_loader = base_train_dataset

        # Use the non-transformed base dataset for visualization base
        vis_dataset_ssl = datasets.CIFAR10(root='./data', train=True, download=False, transform=None)


    # Create the DataLoader for SSL training
    ssl_train_loader = DataLoader(
        ssl_train_dataset_for_loader,
        batch_size=config["ssl_batch_size"],
        shuffle=True, # Shuffle data each epoch
        num_workers=config["num_workers"],
        pin_memory=True, # Speeds up data transfer to GPU
        drop_last=True   # Drop the last incomplete batch, important for NTXentLoss assumptions
    )

    # --- Visualize Augmentations ---
    print("\nVisualizing sample SSL augmentations...")
    visualize_augmentations(vis_dataset_ssl, config["vis_num_augmentations"], augment_vis_save_path)

    # --- Initialize SSL Model, Loss, Optimizer ---
    print("\nInitializing SimCLR model...")
    # Load the base encoder network (e.g., ResNet18) without pre-trained weights
    base_encoder = get_resnet_encoder(name=config["ssl_model_name"], use_pretrained=False)
    # Create the full SimCLR model (encoder + projection head)
    ssl_model = SimCLRModel(base_encoder, config["projection_dim"]).to(config["device"])

    # Initialize the NT-Xent loss function
    ssl_criterion = NTXentLoss(
        temperature=config["temperature"],
        # Provide the expected batch size (loss function might adjust internally for last batch if drop_last=False)
        batch_size=config["ssl_batch_size"],
        n_views=config["n_views"],
        device=config["device"]
    )

    # Initialize the optimizer (AdamW is common for transformers and often works well here too)
    ssl_optimizer = optim.AdamW(
        ssl_model.parameters(), # Optimize all parameters in the SimCLR model (encoder + projector)
        lr=config["ssl_learning_rate"],
        weight_decay=config["ssl_weight_decay"]
    )
    # Initialize a learning rate scheduler (Cosine Annealing is common for SSL)
    ssl_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        ssl_optimizer,
        T_max=len(ssl_train_loader) * config["ssl_epochs"], # Total number of training steps
        eta_min=0 # Minimum learning rate
    )

    # --- SSL Training Loop ---
    print(f"\nStarting SimCLR pre-training for {config['ssl_epochs']} epochs...")
    ssl_train_losses = []
    for epoch in range(config["ssl_epochs"]):
        ssl_model.train() # Set model to training mode
        epoch_loss = 0.0
        num_samples = 0

        # Use tqdm for progress bar
        progress_bar = tqdm(ssl_train_loader, desc=f"SSL Epoch {epoch+1}/{config['ssl_epochs']}", leave=True)

        for batch_idx, (images, _) in enumerate(progress_bar): # Labels (_) are ignored in SSL
            # 'images' is a list of tensors [view1_batch, view2_batch, ...] from ContrastiveTransformations
            # Concatenate the views along the batch dimension:
            # e.g., [B, C, H, W], [B, C, H, W] -> [2*B, C, H, W]
            images_cat = torch.cat(images, dim=0).to(config["device"])
            current_batch_size = images[0].size(0) # Batch size of one view
            num_samples += current_batch_size

            ssl_optimizer.zero_grad() # Reset gradients

            # Forward pass: Get features and projections from the SimCLR model
            _, projections = ssl_model(images_cat)

            # Calculate the contrastive loss using the projections
            loss = ssl_criterion(projections)

            # Check for NaN loss (can happen with unstable training/large LRs)
            if torch.isnan(loss):
                 print(f"\nWarning: NaN loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping update.")
                 # Consider stopping training or reducing LR if this happens frequently
                 continue # Skip backward pass and optimizer step

            # Backward pass: Compute gradients
            loss.backward()
            # Optimizer step: Update model parameters
            ssl_optimizer.step()
            # Scheduler step: Update learning rate (after optimizer step)
            ssl_scheduler.step()

            # Accumulate loss for the epoch. Multiply by batch size since loss is averaged per sample.
            epoch_loss += loss.item() * current_batch_size
            # Update progress bar description with current loss and learning rate
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{ssl_optimizer.param_groups[0]['lr']:.1e}")

        # Calculate average loss for the epoch (per sample)
        avg_epoch_loss = epoch_loss / num_samples if num_samples > 0 else 0
        ssl_train_losses.append(avg_epoch_loss)
        print(f"SSL Epoch {epoch+1}/{config['ssl_epochs']} - Average Loss: {avg_epoch_loss:.5f}")


    end_ssl_time = time.time()
    print(f"\nSSL pre-training finished in {(end_ssl_time - start_ssl_time)/60:.2f} minutes.")

    # --- Save Final SSL Model Weights ---
    print("Saving final SSL model weights...")
    # Save only the ENCODER weights - this is typically what's used for downstream tasks
    torch.save(ssl_model.encoder.state_dict(), ssl_encoder_save_path)
    print(f" --> Final Encoder weights saved to: {ssl_encoder_save_path}")
    # Save the full SimCLR model (encoder + projector) as well, might be useful
    torch.save(ssl_model.state_dict(), ssl_full_model_save_path)
    print(f" --> Final Full SSL model saved to: {ssl_full_model_save_path}")

    # --- Plot SSL Loss Curve ---
    plot_loss_curve(ssl_train_losses, "SimCLR Pre-training Loss per Sample", ssl_loss_save_path)


    # ======================================================
    # == 2. DOWNSTREAM TASK: LINEAR PROBING ==============
    # ======================================================
    print("\n" + "="*70)
    print(" STEP 2: Downstream Task - Linear Probing Evaluation ".center(70, "="))
    print("="*70 + "\n")

    start_linear_time = time.time()

    # Linear Probing Data Loading 
    normalize = transforms.Normalize(config["cifar_mean"], config["cifar_std"])
    linear_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config["image_size"], scale=(0.8, 1.0)), # Less aggressive crop
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    # For testing, only normalization is needed
    linear_test_transform = transforms.Compose([
        transforms.Resize(config["image_size"]), # Ensure consistent size
        transforms.ToTensor(),
        normalize,
    ])

    print("Loading CIFAR-10 dataset for linear probing (with labels)...")
    # Load the training set with standard augmentations
    linear_train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=False, # Already downloaded
        transform=linear_train_transform
    )
    # Load the test set with only test-time transformations
    linear_test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=False,
        transform=linear_test_transform
    )

    # Create DataLoaders for the linear probing phase
    linear_train_loader = DataLoader(
        linear_train_dataset,
        batch_size=config["linear_batch_size"],
        shuffle=True, # Shuffle training data
        num_workers=config["num_workers"],
        pin_memory=True
    )
    linear_test_loader = DataLoader(
        linear_test_dataset,
        batch_size=config["linear_batch_size"],
        shuffle=False, # No shuffling for test set
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # --- Model for Linear Probing ---
    print("\nPreparing model for linear probing...")
    # 1. Load the base encoder structure (must match the architecture saved during SSL)
    encoder = get_resnet_encoder(name=config["ssl_model_name"], use_pretrained=False)

    # 2. Load the saved SSL-trained weights into the encoder
    print(f"Loading pre-trained encoder weights from: {ssl_encoder_save_path}")
    encoder.load_state_dict(torch.load(ssl_encoder_save_path, map_location=config["device"]))
    encoder = encoder.to(config["device"])

    # 3. Freeze the encoder parameters
    # We only want to train the linear classifier head, not the pre-trained encoder.
    print("Freezing encoder parameters...")
    for param in encoder.parameters():
        param.requires_grad = False

    # 4. Create a new linear classifier head
    num_features = encoder.n_features # Get feature dimension from the loaded encoder
    linear_classifier = nn.Linear(num_features, config["num_classes"]).to(config["device"])
    print(f"Created linear classifier head ({num_features} features -> {config['num_classes']} classes).")

    # --- Optimizer and Loss for Linear Probing ---
    linear_optimizer = optim.AdamW(
        linear_classifier.parameters(), # <-- Only pass the classifier's parameters
        lr=config["linear_learning_rate"],
        weight_decay=config["linear_weight_decay"]
    )
    # Standard Cross-Entropy Loss for classification task
    linear_criterion = nn.CrossEntropyLoss().to(config["device"])

    # --- Linear Probing Training Loop ---
    print(f"\nStarting linear classifier training for {config['linear_epochs']} epochs...")
    linear_train_accuracies = []
    best_test_acc = 0.0

    for epoch in range(config["linear_epochs"]):
        encoder.eval()
        linear_classifier.train()

        epoch_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(linear_train_loader, desc=f"Linear Epoch {epoch+1}/{config['linear_epochs']}", leave=False)

        for images, labels in progress_bar:
            images, labels = images.to(config["device"]), labels.to(config["device"])
            linear_optimizer.zero_grad()
            with torch.no_grad():
                features = encoder(images)
            outputs = linear_classifier(features)

            loss = linear_criterion(outputs, labels)
            loss.backward()
            linear_optimizer.step()
            epoch_loss += loss.item() * images.size(0) # Accumulate loss weighted by batch size
            _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item() # Count correct predictions
            batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.2f}%")

        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss / total
        epoch_acc = 100 * correct / total
        linear_train_accuracies.append(epoch_acc)
        print(f"Linear Epoch {epoch+1}/{config['linear_epochs']} -> Avg Loss: {avg_epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%")

        # --- Optional: Evaluate on test set periodically during training ---
        if (epoch + 1) % 5 == 0 or epoch == config["linear_epochs"] - 1: # Eval every 5 epochs and at the end
             encoder.eval()
             linear_classifier.eval()
             test_correct = 0
             test_total = 0
             with torch.no_grad():
                 for test_images, test_labels in linear_test_loader:
                     test_images, test_labels = test_images.to(config["device"]), test_labels.to(config["device"])
                     features = encoder(test_images)
                     outputs = linear_classifier(features)
                     _, predicted = torch.max(outputs.data, 1)
                     test_total += test_labels.size(0)
                     test_correct += (predicted == test_labels).sum().item()
             current_test_acc = 100 * test_correct / test_total
             print(f"  -> Test Accuracy after Epoch {epoch+1}: {current_test_acc:.2f}%")
             if current_test_acc > best_test_acc:
                 best_test_acc = current_test_acc
                 # Save the best performing linear classifier head
                 torch.save(linear_classifier.state_dict(), linear_classifier_save_path)
                 print(f"   -> New best test accuracy! Saved linear classifier head to: {linear_classifier_save_path}")


    end_linear_time = time.time()
    print(f"\nLinear classifier training finished in {(end_linear_time - start_linear_time)/60:.2f} minutes.")

    # --- Final Evaluation on Test Set (using the best saved head) ---
    print("\nEvaluating final linear classifier performance on the test set...")
    # Load the best performing linear head
    if os.path.exists(linear_classifier_save_path):
        print(f"Loading best linear classifier head from: {linear_classifier_save_path}")
        linear_classifier.load_state_dict(torch.load(linear_classifier_save_path, map_location=config["device"]))
    else:
        print("Warning: Best linear classifier head not found. Using the head from the final epoch.")

    encoder.eval()          # Ensure encoder is in eval mode
    linear_classifier.eval() # Ensure classifier is in eval mode

    final_test_correct = 0
    final_test_total = 0
    with torch.no_grad(): # No gradients needed for final evaluation
        for images, labels in tqdm(linear_test_loader, desc="Final Testing"):
            images, labels = images.to(config["device"]), labels.to(config["device"])
            features = encoder(images)           # Extract features
            outputs = linear_classifier(features) # Classify features
            _, predicted = torch.max(outputs.data, 1)
            final_test_total += labels.size(0)
            final_test_correct += (predicted == labels).sum().item()

    final_test_accuracy = 100 * final_test_correct / final_test_total
    print("\n" + "*"*70)
    print(f"| Final Downstream Linear Probing Test Accuracy: {final_test_accuracy:.2f}% |".center(70))
    print("*"*70 + "\n")


    # ======================================================
    # == 3. VISUALIZATION of Learned Features (t-SNE) =====
    # ======================================================
    print("\n" + "="*70)
    print(" STEP 3: Visualize Learned SSL Features using t-SNE ".center(70, "="))
    print("="*70 + "\n")

    # Use the test data loader (with simple transforms) for visualization
    visualize_embeddings(
        encoder, # Pass the SSL-pretrained, frozen encoder
        linear_test_loader, # Use the test loader (contains labels for coloring)
        config["device"],
        config["vis_tsne_subset_size"], # Number of points to visualize
        title="t-SNE of SSL Encoder Features",
        save_path=tsne_save_path
    )

    print("\n" + "#"*70)
    print(" Execution Finished Successfully! ".center(70, "#"))
    print(f"Models and plots saved in directory: {config['save_dir']}".center(70))
    print("#"*70 + "\n")