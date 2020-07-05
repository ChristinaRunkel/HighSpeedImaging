import torch
from torch.utils.data import DataLoader
import datetime
from solver import Solver
from dataloader import VideoVolume
from loss import CharbonnierPenalty
from network import MultiLossNet, SingleLossNet
from scheduler import OneCycleScheduler


'''
Define parameters
'''
GPU = torch.device("cuda", 0)
LOSS_TYPE = 'single'  # one of: each_full, each_once, fade_early, single
INITIALIZATION = "initialize_default"  # any method in solver.py
MULTI_OPTIMIZE = False  # set to true to optimize after each residual block
STITCHING = False # set to true to use stitching during testing
NORMALIZE = True  # true changes range of inputs and gt to be [0,1] instead of [0,255]
SNAPS = 10  # save snapshot every n-epochs
TRAIN_SET_SIZE = 2300 # 2300 max 
#SIZE = (64, 160, 160)  # batch 1 
SIZE = (32, 154, 154)  # batch 2, reducing spatial dimension at least a little to prevent overfitting
#SIZE=(24, 130, 130)  # batch 4
#SIZE=(20, 100, 100)  # batch 8
BATCH_SIZE = 2
TAIL = 5
EPOCHS = 15 + TAIL
CONTINUE_EPOCH = 0  # if this is not 0, model is initialized from the snapshot
CONTINUE_SNAPSHOT = "snapshots/snapshot_epoch40_2019-07-11T081108Z.pth"
EXTRA_BLOCK = False
ONE_CYCLE = True
DROPOUT = False


'''
Load data
'''
if TRAIN_SET_SIZE > 1500:
    v1 = VideoVolume(gpu=GPU, normalize=NORMALIZE, size=SIZE, test=False, start=0, end=1500)
    second = TRAIN_SET_SIZE-1500
    assert second <= 800, "train set overlaps with test set"
    v2 = VideoVolume(gpu=GPU, normalize=NORMALIZE, size=SIZE, test=False, start=1600, end=1600+second)
    train_set = v1 + v2
else:
    train_set = VideoVolume(gpu=GPU, normalize=NORMALIZE, size=SIZE, test=False, start=0, end=TRAIN_SET_SIZE)  # start small for faster results

val_set   = VideoVolume(gpu=GPU, normalize=NORMALIZE, size=SIZE, test=True, start=1500, end=1600, debug=False)  # validation now without augmentation
test_set  = VideoVolume(gpu=GPU, normalize=NORMALIZE, size=SIZE, test=True, start=2400, end=2500, debug=False)

train_loader = DataLoader(train_set, BATCH_SIZE=BATCH_SIZE, shuffle=True, num_workers=0) # workers=0 since data is on gpu
val_loader   = DataLoader(val_set, BATCH_SIZE=BATCH_SIZE, num_workers=0)
test_loader  = DataLoader(test_set, BATCH_SIZE=1, num_workers=0)

date_str = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")

'''
Load network
'''
if not DROPOUT and CONTINUE_EPOCH > 0:  # convert weights from a network trained with dropout to training without it
    snap = torch.load(CONTINUE_SNAPSHOT, map_location="cpu")
    if "model" in snap:
        snap = snap["model"]
    dummy_layers = "layers.0.layers.12.weight" in snap
    network = MultiLossNet(final_block=EXTRA_BLOCK, dropout=DROPOUT, dummy_dropout_layers=dummy_layers).to(GPU)
else:
    network = MultiLossNet(final_block=EXTRA_BLOCK, dropout=DROPOUT).to(GPU)

'''
Set loss function, optimizer, scheduler
'''
loss = CharbonnierPenalty(10, total_variation=False, per_pixel=False)  # highest possible charbonnier n is 1023

hi_lr = 3e-3  # only used for one_cycle 
low_lr = 1e-3
#optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=2e-3)
#optimizer = torch.optim.SGD(network.parameters(), lr=1e-10, momentum=0.9, weight_decay=2e-3)
optimizer = torch.optim.AdamW(network.parameters(), lr=low_lr, weight_decay=8e-9, betas=(0.9, 0.99))
scheduler = None
if ONE_CYCLE:
    ONE_CYCLE = OneCycleScheduler(epochs=EPOCHS, hi_lr=hi_lr, low_lr=low_lr, tail_epochs=TAIL)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, ONE_CYCLE)

solver = Solver(train_loader=train_loader, validation_loader=val_loader, test_loader=test_loader, 
                loss=loss, device=GPU, date_str=date_str, continue_epoch=CONTINUE_EPOCH, continue_snapshot=CONTINUE_SNAPSHOT)

'''
Save parameters to log file
'''
param_str = "Parameters: device={}, extra_block={}, dropout={}, loss_type='{}', loss={}, multi_optimize={}, epochs={}, batch_size={}, train_videos={}, optimizer={}, initialization='{}', scheduler={}".format(
    GPU, EXTRA_BLOCK, DROPOUT, LOSS_TYPE, loss, MULTI_OPTIMIZE, EPOCHS, BATCH_SIZE, TRAIN_SET_SIZE, optimizer, INITIALIZATION if CONTINUE_EPOCH <= 0 else CONTINUE_SNAPSHOT, ONE_CYCLE)
print(param_str)
fw = open(solver.logfile, 'a+')
if CONTINUE_EPOCH > 0:
    fw.write("\n\n")
fw.write(param_str)
fw.write("\n")
fw.close()

'''
Train and evaluate network
''' 
solver.train_network(network, optimizer, epochs=EPOCHS, scheduler=scheduler, loss_type=LOSS_TYPE, multi_optimize=MULTI_OPTIMIZE, snaps=SNAPS, init=INITIALIZATION, print_initial_losses=10, snapshot_metadata=param_str)
print("Start testing")
solver.evaluate_network(network, validation=False, loss_type=LOSS_TYPE)
