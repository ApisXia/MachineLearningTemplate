from __future__ import division
import torch
import torch.optim as optim
from torch.nn import functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
from tqdm import tqdm
import time


# Detect nan input in run time
def nan_hook(self, inp, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print("In", self.__class__.__name__)
            raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:",
                               out[nan_mask.nonzero()[:, 0].unique(sorted=True)])


class Trainer(object):
    def __init__(self, model, device, train_dataset, test_dataset, exp_name,
                 optimizer='Adam', lr=0.1, lr_step=800, lr_gamma=0.5,
                 val_period=5, val_regular=True,
                 silent_progress=False, nan_picker=False):
        self.model = model.to(device)
        self.device = device
        self.val_period = val_period
        self.silent = silent_progress
        self.val_regular = val_regular
        self.nan_picker = nan_picker
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=lr)
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_step, gamma=lr_gamma)

        self.train_dataset = train_dataset
        self.val_dataset = test_dataset
        self.exp_path = os.path.dirname(__file__) + '/experiments/{}/'.format( exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format(exp_name)
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary'.format(exp_name))
        self.val_min_loss = 10000
        self.val_min_epoch = 0
        self.val_min_ckpt_format = 'Best_{}.tar'

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_loss(self, batch):
        raise ValueError('Need to modify.')
        pass

    def train_model(self, epochs):
        loss = 0
        train_data_loader = self.train_dataset.get_loader()
        start = self.load_checkpoint()

        for epoch in range(start, epochs):
            sum_loss = 0
            desc = tqdm(
                train_data_loader,
                leave=True,
                unit="batch",
                postfix={'loss': 0.0},
                disable=self.silent
            )
            desc.set_description_str("Training... Epoch: {}, LR: {}".format(epoch, self.scheduler.get_last_lr()[0]))
            for batch in desc:
                loss = self.train_step(batch)
                if self.nan_picker:
                    for submodule in self.model.modules():
                        submodule.register_forward_hook(nan_hook)
                # print("Current loss: {}".format(loss / self.train_dataset.num_sample_points))
                desc.set_postfix(loss="%.6f" % loss)
                sum_loss += loss

            self.scheduler.step()
            self.writer.add_scalar('training loss last batch', loss, epoch)
            self.writer.add_scalar('training loss batch avg', sum_loss / len(train_data_loader), epoch)

            # validate model
            if (epoch + 1) % self.val_period == 0:  # eval model every X min and at start
                if self.val_regular:
                    self.save_checkpoint(epoch)  # per checkpoint

                val_loss = self.compute_val_loss()

                if val_loss < self.val_min_loss:
                    self.val_min_loss = val_loss
                    for path in glob(self.checkpoint_path + self.val_min_ckpt_format.format('*')):
                        os.remove(path)
                    self.save_checkpoint(epoch, self.val_min_ckpt_format.format('Ep_%d_vls_%.6f' % (epoch, val_loss)))
                    print('**** best model updated with Loss={:.6f} ****'.format(val_loss))
                    time.sleep(0.001)
                    # np.save(self.exp_path + 'val_min={}'.format(epoch), [epoch, val_loss])

                self.writer.add_scalar('val loss batch avg', val_loss, epoch)
                print('Evaluating... Epoch: {}, vls={}'.format(epoch, val_loss))
                time.sleep(0.001)

    def save_checkpoint(self, epoch, define_name=None):
        if define_name is None:
            path = self.checkpoint_path + 'checkpoint_{}.tar'.format(epoch)
        else:
            path = self.checkpoint_path + define_name
        if not os.path.exists(path):
            torch.save({ #'state': torch.cuda.get_rng_state_all(),
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=float)
        checkpoints = np.sort(checkpoints)
        # path = self.checkpoint_path + 'checkpoint_{}h:{}m:{}s_{}.tar'.format(*[*convertSecs(checkpoints[-1]),checkpoints[-1]])
        path = self.checkpoint_path + 'checkpoint_{}.tar'.format(int(checkpoints[-1]))

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        # training_time = checkpoint['training_time']
        # torch.cuda.set_rng_state_all(checkpoint['state']) # batch order is restored. unfortunately doesn't work like that.

        # ! load best info # tododdd

        return epoch

    def compute_val_loss(self):
        self.model.eval()

        sum_val_loss = 0
        num_batches = 120
        for _ in range(num_batches):
            try:
                val_batch = self.val_data_iterator.next()
            except:
                self.val_data_iterator = self.val_dataset.get_loader().__iter__()
                val_batch = self.val_data_iterator.next()

            sum_val_loss += self.compute_loss(val_batch).item()

        return sum_val_loss / num_batches