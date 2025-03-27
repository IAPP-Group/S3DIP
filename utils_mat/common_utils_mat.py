""" common_utils.py utilizzando direttamente le matrici delle immagini (senza passare da Pillow) """

import torch
import torchvision
import numpy as np

import matplotlib.pyplot as plt
import cv2


# crop_image
def crop_image(img, d=32):
    """Make dimensions divisible by `d`"""

    # img.shape[0] è la height
    # img.shape[1] è la width
    #  ---> x
    # |
    # |
    # V # y

    new_size = (img.shape[0] - img.shape[0] % d,
                img.shape[1] - img.shape[1] % d)
    print("new size: ", new_size)

    x0 = int((img.shape[1] - new_size[1]) / 2)  # top left corner x
    y0 = int((img.shape[0] - new_size[0]) / 2)  # top left corner y

    x1 = int((img.shape[1] + new_size[1]) / 2)  # bottom right corner x
    y1 = int((img.shape[0] + new_size[0]) / 2)  # bottom right corner y
    print("x0: ", x0)
    print("y0: ", y0)
    print("x1: ", x1)
    print("y1: ", y1)

    crop_img = img[y0:y1, x0:x1]

    return crop_img


def get_params(opt_over, net, net_input, noise_tensor, downsampler=None):
    """Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
        noise_tensor: torch.tensor in which input speckle is stored
    """
    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters()]
            print("net parameters")
        elif opt == 'down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
            print("input parameters")
        elif opt == 'noise':   # ottimizza su noise
            # params += [x for x in sp_net.parameters()]
            noise_tensor.requires_grad = True
            params += [noise_tensor]

            print("speckle parameters")
        else:
            assert False, 'what is it?'

    return params


def get_image_grid(images_np, nrow=8):
    """Creates a grid from a list of images by concatenating them."""
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()


def plot_image_grid(images_np, writer, step, nrow=8, factor=1, interpolation='lanczos'):
    """Draws images in a grid

    Args:
        images_np: list of images, each image is np.array of size 3xHxW or 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)

    # fig = plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    fig = plt.figure(figsize=(len(images_np) + factor, factor))

    if images_np[0].shape[0] == 1:
        # plt.imshow clippa naturalmente tra 0,1 se valori float, tra 0,255 se int
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)

    # if step == 0:
    #     writer.log_figure(figure_name='Start clean and noisy images', figure=fig, step=step)  # name, plt, l
    #     # writer.add_figure('Start clean and noisy images', fig, step)
    # else:
    #     name = 'out_im_' + str(step)
    #     writer.log_figure(figure_name=name, figure=fig, step=step)  # name, plt, l
    #     # writer.add_figure('Training/out_images', fig, step)

    if writer is None:
        pass
        # plt.show()
    else:
        name = 'out_im_' + str(step)
        writer.log_figure(figure_name=name, figure=fig, step=step)  # name, plt, l

    fig.clear()
    plt.close(fig)

    # return grid


# def load(path):
#     """Load PIL image."""
#     img = Image.open(path)
#     return img


# get_image
def resize(img, imsize=-1):  # old get_image
    """Resize an image to a specific size.

    Args:
        img: img as a matrix
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """

    # Resize con opencv

    if isinstance(imsize, int):
        print("1")
        imsize = (imsize, imsize)

    if imsize[0] != -1 and img.shape != imsize:
        print("2")
        if imsize[0] > img.shape[0]:
            print("3")
            img = cv2.resize(img, dsize=imsize, interpolation=cv2.INTER_CUBIC)
        else:
            print("Antialias interpolation")
            # TODO: controlla implementazione
            img = cv2.resize(img, dsize=imsize, interpolation=cv2.INTER_CUBIC)
            # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    return img


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    import pickle

    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var

    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input


# pil_to_np
def normalization(img, factor):  # sarebbbe circa pil_to_np # TODO: migliorare
    # normalization
    # if len(img.shape) == 3:
    #     img = img.transpose(2, 0, 1)  # da (W, H, C) a (C, W, H)
    # else:
    #     img = img[None, ...]  # da (W, H) a (1, W, H)

    if not factor:
        # calcolo il massimo e normalizzo per quel valore a
        max_val = np.max(img)
    else:
        max_val = factor
    # print("Max value of intensity: ", max_val)
    norm_im = img.astype(np.float32) / max_val

    return norm_im, max_val


# np_to_pil
def np_to_visual(img_np):  # max_val solo per salvare e visualizzare
    """
    From C x W x H [0..1] to  W x H x C [0...255]
    """
    # ar = np.clip(img_np * max_val, 0, max_val).astype(np.uint8)  # 255
    #
    # if img_np.shape[0] == 1:
    #     ar = ar[0]
    # else:
    #     ar = ar.transpose(1, 2, 0)
    if img_np.shape[0] == 1:
        img_np = img_np[0]
    else:
        img_np = img_np.transpose(1, 2, 0)

    return img_np


def np_to_torch(img_np):
    """Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    """
    # print("shape img np: ", img_np.shape)
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    """
    return img_var.detach().cpu().numpy()[0]


def optimize(optimizer_type, parameters, closure, LR, num_iter, speckle_tensor):  # writer
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')

        def closure2():
            optimizer.zero_grad()
            return closure()

        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)

        # total_norm = 0
        for j in range(num_iter):

            # TODO: dopo un certo numero di epoche inserire parametri rumore da ottimizzare
            # if j == 100:
            #     print("ADD SPECKLE PARAMETERS")
            #     # p = get_params(opt_over='net,noise', net=net, net_input=net_input, noise_tensor=speckle_tensor,
            #     #                downsampler=None)
            #     speckle_tensor.requires_grad = True
            #     #  parameters += [speckle_tensor]
            #     # optimizer.param_groups.append({'params': speckle_tensor})
            #     s = sum([np.prod(list(g.size())) for g in parameters])
            #     print('Number of TOTAL params: %d' % s)
            #     # print("Params group len: ", len(optimizer.param_groups))
            #     # print("NUOVO OTTIMIZZATORE")
            #     # optimizer = torch.optim.Adam(parameters, lr=LR)

            optimizer.zero_grad()
            closure()

            ### Monitoraggio norma gradiente
            # for p in parameters:
            #     param_norm = p.grad.detach().data.norm(2)
            #     total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** 0.5
            # writer.add_scalar('Training/total_norm', total_norm, j)
            # print("total norm shape: ", total_norm)

            # clip gradienti TODO: ricontrollare
            # total_norm = torch.nn.utils.clip_grad_norm(parameters, max_norm=1)
            # # print("total norm shape: ", total_norm)
            # writer.add_scalar('Training/total_norm', total_norm, j)

            ###
            optimizer.step()
    else:
        assert False


# added
def padding(img, d=32):
    import math
    # print("PADDING FUNCTION")

    # controllo shape
    if len(img.shape) == 2:
        # print("H, W")
        hei, wid = img.shape  # abbiamo un nd_array
        # print("hei and wid: ", hei, wid)
        w_pad = int(math.floor((int(math.ceil(wid / d)) * d - wid) / 2))
        h_pad = int(math.floor((int(math.ceil(hei / d)) * d - hei) / 2))
        print("pad: ", w_pad, h_pad)
        padded_img = np.pad(img, ((h_pad, h_pad), (w_pad, w_pad)), mode='constant')  # pad 0
        # TODO: aggiungi 1 per il canale ...se lo aggiungi in testa ricorda di trasporre
        #  quando hai tre canali per avere (C, H, W) invece di (H, W, C)
        padded_img = padded_img[None, ...]  # (1, H, W)

    else:
        # print("H, W, C")
        hei, wid, _ = img.shape  # abbiamo un nd_array
        # print("hei and wid: ", hei, wid)
        w_pad = int(math.floor((int(math.ceil(wid / d)) * d - wid) / 2))
        h_pad = int(math.floor((int(math.ceil(hei / d)) * d - hei) / 2))
        # print("pad: ", w_pad, h_pad)
        padded_img = np.pad(img, ((h_pad, h_pad), (w_pad, w_pad), (0, 0)), mode='constant')  # pad 0
        padded_img = np.transpose(padded_img, (2, 0, 1))

    # print("padded_img shape: ", padded_img.shape)
    return padded_img, h_pad, w_pad


def de_padding(img_to_depad, h_pad, w_pad):  # in input ho un tensore del tipo CxHxW
    # print("DE PADDING FUNCTION")
    # 1, C, H, W
    if len(img_to_depad.shape) == 3:
        # print("3 ...")
        # (C, H, W)
        _, hei, wid = img_to_depad.shape
        # print("hei and wid: ", hei, wid)
        cropped = img_to_depad[:, h_pad:(hei - h_pad), w_pad:(wid - w_pad)]

    else:
        # print("4 ...")
        # (N, C, H, W)
        _, _, hei, wid = img_to_depad.shape
        # print("hei and wid: ", hei, wid)
        cropped = img_to_depad[:, :, h_pad:(hei - h_pad), w_pad:(wid - w_pad)]  # è un tensore

    # print("crop shape: ", cropped.shape)
    return cropped
