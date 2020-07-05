import torch
import datetime
from torch_stitching import ChunkedProcessor, simulate_net_output
from loss import calculate_ssim, calculate_psnr
import math


class Solver():

    def __init__(self, train_loader, validation_loader, test_loader, loss, device, date_str, continue_epoch=0, continue_snapshot=None):
        super().__init__()
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.loss = loss
        self.device = device
        self.date_str = date_str
        self.continue_epoch = continue_epoch
        self.continue_snapshot = continue_snapshot
        if continue_epoch > 0:
            #override date_str and logfile to continue logging into same file and snapshot
            fromstr = "epoch"+str(continue_epoch)+"_"
            from_index = continue_snapshot.find(fromstr)
            if from_index < 0:
                raise ValueError("provided snapshot does not match provided epoch")
            self.date_str = continue_snapshot[from_index+len(fromstr):-4]
        self.logfile = "train_logs/loss_"+self.date_str+".txt"

    def find_lr(self, model, optimizer, start_lr=1e-10, max_lr=1, beta=0.98, loss_type='single', init='initialize_default'):
        model.apply(globals()[init])
        
        num = len(self.train_loader)-1
        mult = (max_lr / start_lr) ** (1/num)
        lr = start_lr
        optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        lrs = []
        for data, gt in self.train_loader:
            data, gt = data.to(self.device), gt.to(self.device)
            data = data.unsqueeze(1)
            gt = gt.unsqueeze(1)
            batch_num += 1
            #As before, get the loss for this mini-batch of inputs/outputs
            optimizer.zero_grad()
            out_array = model(data)
            batch_loss, loss_array = self.__calc_loss__(out_array, gt, loss_type, 0, 1)
            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1-beta) *batch_loss.item()
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                return lrs, losses
            #Record the best loss
            if smoothed_loss < best_loss or batch_num==1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            lrs.append(lr)
            print("LR: {:<.20f}, loss: {:>20,.4f}, {}/{}".format(lr, smoothed_loss, batch_num, num))
            #Do the SGD step
            batch_loss.backward()
            optimizer.step()
            #Update the lr for the next step
            lr *= mult
            optimizer.param_groups[0]['lr'] = lr
        return lrs, losses
    
    def train_network(self, model, optimizer, epochs, scheduler=None, loss_type='single', multi_optimize=False, snaps=10, init='initialize_weights_kaiming_out_leaky', print_initial_losses=0, snapshot_metadata=None):
        '''
        Args:
            loss_type: 'each_full'  = (L1 + L1+L2 + L1+L2+L3) (previously multi_loss=true)
                       'each_once'  = (L1+L2+L3)
                       'fade_early' = ((1-epoch/epochs)(L1+L2)+L3), less impact of intermediate layers over time
                       'single'     = (L3) (previously multi_loss=false)
            multi_optimize: call loss.backward() on the loss result of every block, instead of just at the end
            init: any method name that is defined in this file here
            continue_epoch: e.g. if you completed 20 epochs of training, set this to 20 and provide the snapshot of 20
            snapshot_metadata: included under "metadata" in the snapshots, e.g. a description of parameters
        '''
        
        if model.__class__.__name__ == "SingleLossNet" and loss_type != 'single':
            raise ValueError("For SingleLossNet please set loss_type='single'.")
        
            
        print("Start training -", self.date_str)
        if self.continue_epoch <= 0:
            model.apply(globals()[init])
        else:
            checkpoint = torch.load(self.continue_snapshot, map_location=self.device)
            if "model" in checkpoint:
                print("loading model and optimizer")
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
            else:
                # support for old snapshots
                print("loading model. warning: optimizer state was not loaded")
                model.load_state_dict(checkpoint)
            epochs += self.continue_epoch  # adjust max epochs here!
        model.train()

        number_elements = len(self.train_loader) #* self.train_loader.batch_size
        
        for i in range(self.continue_epoch, epochs):
            print("Epoch {}:".format(i+1))
            model.train()
            train_loss = 0.0

            for k, (data, gt) in enumerate(self.train_loader):
                #needed to match the expected 5D format even videos don't have multiple channels
                data = data.unsqueeze(1)
                gt = gt.unsqueeze(1)

                optimizer.zero_grad()
                data, gt = data.to(self.device), gt.to(self.device)
                
                out_array = model(data)
                batch_loss, loss_array = self.__calc_loss__(out_array, gt, loss_type, i, epochs)
                
                if multi_optimize:
                    for j in range(len(loss_array)):
                        not_last = j != len(loss_array)-1
                        loss_array[j].backward(retain_graph=not_last)  # do not retain on last iteration
                else:
                    batch_loss.backward()  # note that this now also works for single_layer network
                
                output = out_array[-1] if isinstance(out_array, list) else out_array
                train_loss += batch_loss.item()
                if i == self.continue_epoch and k < print_initial_losses:
                    print("Initial loss:    {:>20,.4f}".format(train_loss/(k+1)))
                optimizer.step()
            train_loss /= number_elements
            
            print("Training loss:   {:>20,.4f}".format(train_loss))
            self.save_loss_to_file(eval_type="training", loss=train_loss)
            
            print("Start validation")
            validation_loss = self.evaluate_network(model, loss_type=loss_type, epoch=i, maxepochs=epochs)
            
            if scheduler is not None:
                scheduler.step()
            
            if i > 0 and (i+1)%snaps==0 or i == epochs-1:
                path = "snapshots/snapshot_epoch"+str(i+1)+"_"+self.date_str+".pth"
                print("Saving:", path)
                torch.save({
                    "model":     model.state_dict(), 
                    "optimizer": optimizer.state_dict(),
                    "scheduler": None if scheduler is None else scheduler.state_dict(),  # will probably remain unused
                    "metadata":  snapshot_metadata
                }, path)
    
    def evaluate_network(self, model, validation=True, stitching=False, loss_type='single', epoch=0, maxepochs=1):
        model.eval()

        evaluation_loss = 0.0
        ssim_total = 0.0
        psnr_total = 0.0

        with torch.no_grad():
            if validation:
                data_loader = self.validation_loader
                eval_type = "validation"  # this is no longer called loss_type
            else:
                data_loader = self.test_loader
                eval_type = "test"

            for (data, gt) in data_loader:  # should be (batch x frames x height x width)
                
                for video, video_gt in zip(data, gt): # for each in batch: (frames x height x width)
                    video, video_gt = video.to(self.device), video_gt.to(self.device) 
                    
                    out = []
                    gt_array = []
                    
                    if stitching:
                        p = ChunkedProcessor(video, chunk_size=(30, 160, 160), debug=False)  # split large image into chunks
                        p_gt = ChunkedProcessor(video_gt, chunk_size=(30, 160, 160), debug=False)
                        
                        for chunk_gt, chunk in zip(p_gt, p):
                            chunk_gt = chunk_gt.unsqueeze(0).unsqueeze(0)
                            chunk = chunk.unsqueeze(0).unsqueeze(0) # needed to fit 5D input dimensions of network

                            out_array = model(chunk)  # might not actually be an array, but it's fine
                            output = out_array[-1].squeeze()
                            p.store_back(output)
                            
                            # used to calculate loss for each chunk seperately; neeeded for all "multi" loss tpyes
                            out += [out_array] 
                            gt_array += [chunk_gt]

                        full = p.recombine()
                        full = full.to(self.device)
                        
                    else:
                        video = video[None, None, ...]
                        video_gt = video_gt[None, None, ...]
                        out_array = model(video)
                        out += [out_array]
                        gt_array += [video_gt]
                    
                    
                    for i in range(len(out)):
                        video_loss, loss_array = self.__calc_loss__(out[i], gt_array[i], loss_type, epoch, maxepochs)
                        evaluation_loss += video_loss.item()
                        
                        if not validation:
                            output = out[i][-1].squeeze().cpu().numpy()
                            groundtruth = gt_array[i].squeeze().cpu().numpy()
                            ssim_total += calculate_ssim(output, groundtruth)
                            psnr_total += calculate_psnr(output, groundtruth)
                        
            number_elements = len(data_loader) * len(out) * data_loader.batch_size 

            evaluation_loss /= number_elements
            if not validation:
                ssim = ssim_total/number_elements
                psnr = psnr_total/number_elements
            
        print("Evaluation loss: {:>20,.4f}".format(evaluation_loss))
        if not validation:
            print("SSIM: {:>20,.3f}".format(ssim))
            print("PSNR: {:>20,.2f}".format(psnr))
            self.save_loss_to_file(eval_type=eval_type, loss=evaluation_loss, ssim=ssim, psnr=psnr)
        else:
            self.save_loss_to_file(eval_type=eval_type, loss=evaluation_loss)
        
    def __calc_loss__(self, out_array, gt, loss_type, epoch, maxepochs):
        '''
        return batch_loss, loss_array to avoid duplicate code in train and eval
        '''
        
        loss_array = [0] * len(out_array)

        if loss_type is 'each_full':
            # split loss into loss after 1st, 2nd, ..., n-st residual block
            for i in range(len(out_array)):
                if i==0:
                    loss_array[i] = self.loss(out_array[i], gt)
                else:
                    loss_array[i] = self.loss(out_array[i], gt) + loss_array[i-1]
        elif loss_type is 'each_once':
            for i in range(len(out_array)):
                loss_array[i] = self.loss(out_array[i], gt)
        elif loss_type is 'fade_early':
            for i in range(len(out_array)):
                # first epoch L1 and L2 have full weight
                # middle epoch they have half weight
                # last epoch they have no weight 
                mult = 1 if i==len(out_array)-1 else 1.-(epoch/maxepochs)
                loss_array[i] = mult * self.loss(out_array[i], gt)
        elif loss_type is 'single':
            # this should now work with Multi_loss and Single_loss networks
            if isinstance(out_array, list):
                out_array = out_array[-1]
            loss_array[-1] = self.loss(out_array, gt)
        else:
            raise NotImplementedError(loss_type)

        batch_loss = sum(loss_array)
        return batch_loss, loss_array
    
    def save_loss_to_file(self, eval_type, loss, ssim=None, psnr=None):
        fw = open(self.logfile, 'a+')
        if ssim is not None and psnr is not None:
            line = "\n{} - {:<12} - {:>20,.4f} - ssim={:>4,.3f} - psnr={:>4,.2f}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), eval_type, loss, ssim, psnr)
        else:
            line = "\n{} - {:<12} - {:>20,.4f}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         eval_type,
                         loss)
        fw.write(line)
        fw.close()
        
def initialize_weights_kaiming_out_leaky(m):
    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
            
def initialize_weights_kaiming_out_leaky_uniform(m):
    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_out')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
            
def initialize_weights_kaiming_in_leaky(m):
    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
            
def initialize_weights_normal(m):
    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.normal_(m.weight, mean=0, std=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
            
def initialize_weights_kaiming3(m):
    if isinstance(m, torch.nn.Conv3d):
        shp = m.weight.data.shape
        var = 3.0/torch.Tensor(list(shp))[1:].prod()
        torch.nn.init.normal_(m.weight, mean=0, std=math.sqrt(var))
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def initialize_weights_kaiming2_5(m):
    if isinstance(m, torch.nn.Conv3d):
        shp = m.weight.data.shape
        var = 2.5/torch.Tensor(list(shp))[1:].prod()
        torch.nn.init.normal_(m.weight, mean=0, std=math.sqrt(var))
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
            
def initialize_default(m):
    pass