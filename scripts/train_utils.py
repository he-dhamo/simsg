from tensorboardX import SummaryWriter
import torch
from torch.functional import F
from collections import defaultdict


def check_args(args):
  H, W = args.image_size
  for _ in args.decoder_network_dims[1:]:
    H = H // 2

  if H == 0:
    raise ValueError("Too many layers in decoder network")


def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
  curr_loss = curr_loss * weight
  loss_dict[loss_name] = curr_loss.item()
  if total_loss is not None:
    total_loss += curr_loss
  else:
    total_loss = curr_loss
  return total_loss


def calculate_model_losses(args, skip_pixel_loss, img, img_pred, bbox, bbox_pred):

  total_loss = torch.zeros(1).to(img)
  losses = {}

  l1_pixel_weight = args.l1_pixel_loss_weight

  if skip_pixel_loss:
    l1_pixel_weight = 0

  l1_pixel_loss = F.l1_loss(img_pred, img)

  total_loss = add_loss(total_loss, l1_pixel_loss, losses, 'L1_pixel_loss',
                        l1_pixel_weight)

  loss_bbox = F.mse_loss(bbox_pred, bbox)
  total_loss = add_loss(total_loss, loss_bbox, losses, 'bbox_pred',
                        args.bbox_pred_loss_weight)

  return total_loss, losses


def init_checkpoint_dict(args, vocab, model_kwargs, d_obj_kwargs, d_img_kwargs):

  ckpt = {
        'args': args.__dict__, 'vocab': vocab, 'model_kwargs': model_kwargs,
        'd_obj_kwargs': d_obj_kwargs, 'd_img_kwargs': d_img_kwargs,
        'losses_ts': [], 'losses': defaultdict(list), 'd_losses': defaultdict(list),
        'checkpoint_ts': [], 'train_iou': [], 'val_losses': defaultdict(list),
        'val_iou': [], 'counters': {'t': None, 'epoch': None},
        'model_state': None, 'model_best_state': None, 'optim_state': None,
        'd_obj_state': None, 'd_obj_best_state': None, 'd_obj_optim_state': None,
        'd_img_state': None, 'd_img_best_state': None, 'd_img_optim_state': None,
        'best_t': [],
      }
  return ckpt


def print_G_state(args, t, losses, writer, checkpoint):
  # print generator losses on terminal and save on tensorboard

  print('t = %d / %d' % (t, args.num_iterations))
  for name, val in losses.items():
    print('G [%s]: %.4f' % (name, val))
    writer.add_scalar('G {}'.format(name), val, global_step=t)
    checkpoint['losses'][name].append(val)
  checkpoint['losses_ts'].append(t)


def print_D_obj_state(args, t, writer, checkpoint, d_obj_losses):
  # print D_obj losses on terminal and save on tensorboard

  for name, val in d_obj_losses.items():
    print('D_obj [%s]: %.4f' % (name, val))
    writer.add_scalar('D_obj {}'.format(name), val, global_step=t)
    checkpoint['d_losses'][name].append(val)


def print_D_img_state(args, t, writer, checkpoint, d_img_losses):
  # print D_img losses on terminal and save on tensorboard

  for name, val in d_img_losses.items():
    print('D_img [%s]: %.4f' % (name, val))
    writer.add_scalar('D_img {}'.format(name), val, global_step=t)
    checkpoint['d_losses'][name].append(val)
