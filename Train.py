import os
import torch
import torch.optim as optim
import time

from models.FracFormer import Fracformer, parse_arg,ReconLoss
from FractureLoader import DataLoader
from pointnet2_ops import pointnet2_utils

# DataSets and Model path
boneName = 'Femur'  # Hipbone , Sacrum , Femur
model_name = 'FracFormer' + '_' + boneName
checkpoint_path = os.path.join('checkpoint', model_name+'.pth')

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_recon = checkpoint['best_recon']
        Best_epoch = checkpoint['best_epoch']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        return start_epoch, best_recon,Best_epoch
    else:
        print("No checkpoint found, starting from scratch.")
        return 0, float('inf'), 0

def save_checkpoint(model, optimizer, scheduler, epoch, best_recon,best_epoch, checkpoint_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_recon': best_recon,
        'best_epoch': best_epoch,
    }, checkpoint_path)

def fps_cat(fps_idx_0,fps_sets):
    batch_indices = torch.arange(fps_idx_0.size(0)).unsqueeze(1)
    actual_fps_idx_1 = fps_idx_0[batch_indices,fps_sets[0]]
    actual_fps_idx_2 = actual_fps_idx_1[batch_indices,fps_sets[1]]
    return actual_fps_idx_2

###########################
#  Main Training Loop
###########################
if __name__ == "__main__":
    # device
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # model
    args = parse_arg()
    model = Fracformer(args).to(device)

    # prepare training and testing dataset
    DataGenerator = DataLoader(boneName)
    DataLoader = torch.utils.data.DataLoader(DataGenerator, batch_size=20, shuffle=True, num_workers=5,
                                             pin_memory=True, persistent_workers=True)

    # loss function
    criterion_PointLoss = ReconLoss().to(device)

    # optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-6)

    # Resume from checkpoint if available
    start_epoch, best_recon,Best_epoch = load_checkpoint(model, optimizer, scheduler, checkpoint_path)

    niter = 300  # Total number of epochs
    start_time_epoch = time.time()
    for epoch in range(start_epoch, niter):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for i, FracturesDict in enumerate(DataLoader):

                optimizer.zero_grad()

                Fractures, Target, Label,fps_idx_0 = FracturesDict['Fractures'].to(device), FracturesDict['Target'].to(device), \
                FracturesDict['Label'].to(device),FracturesDict['fps_0'].to(device)

                encoded_label = torch.nn.functional.one_hot(Label, num_classes=7)
                Fractures_label = torch.cat((Fractures, encoded_label), 2).contiguous()

                # Model forward pass
                Recon = model(Fractures_label)
                final_fps_idx = fps_cat(fps_idx_0, Recon[-2])
                Target_fps = pointnet2_utils.gather_operation(Target.transpose(1, 2).contiguous(),
                                                              final_fps_idx).transpose(1, 2).contiguous()
                # Compute loss
                loss_coarse, loss_fine = criterion_PointLoss(Recon, [Target,Target_fps,encoded_label])
                loss_recon = loss_coarse + loss_fine

                # Backpropagation
                loss_recon.backward()
                optimizer.step()

                epoch_loss += loss_recon.item()
                # Logging for current batch
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print( f'Epoch [{epoch}/{niter}], Batch [{i + 1}/{len(DataLoader)}], Loss: {loss_recon.item():.4f}, Time: {elapsed_time:.2f}s')

        # Scheduler step after epoch
        scheduler.step()
        save_checkpoint(model, optimizer, scheduler, epoch, best_recon, Best_epoch, checkpoint_path)
