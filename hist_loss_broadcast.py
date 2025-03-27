""" Histogram loss  Implementation """

import torch
import torch.nn

# h =  vettore dei bin dell'istogramma del rumore empirico che vogliamo ottimizzare
# (N che inizializzo in qualche modo (matrice di tutti 1 / rumore gaussiano))
# h_rif = vettore dei bin dell'istogramma del rumore di riferimento Nrif...


def compute_h_broadcast(noise_tensor, resh_expanded, step):

    # (C, H, W)
    delta = torch.clip((step - (abs(noise_tensor - resh_expanded))) / step, 0, 1)
    # print("delta shape: ", delta.shape)
    sum = torch.sum(delta, dim=(1, 2))
    # print("sum shape: ", sum.shape)

    # out of range pixels
    # diff = noise_tensor - resh_expanded
    # under_min = torch.sum(diff[0] < 0).item()  # count the negative values of noise_tensor - t0
    # over_max = torch.sum(diff[-1] > 0).item()  # count the positive values of noise_tensor - tR
    # print("under min: ", under_min)
    # print("over max: ", over_max)

    h = sum / (noise_tensor.shape[1] * noise_tensor.shape[2])
    # h_sum = torch.sum(h)
    # print("h sum: ", h_sum.item())

    return h  # , under_min, over_max



# istogramma h empirico (calcolato con compute_h) e istogramma del rumore di riferimento
# per adesso usiamo U generato da matlab e 'campionato' con lo stesso istogramma fissato qui ...
def histogram_loss(noise_bins, ref_bins):
    diff = (noise_bins - ref_bins) ** 2
    # diff = noise_bins - ref_bins  # prova l1
    # print("diff shape: ", diff.shape)
    hl = torch.sum(diff)
    # hl = torch.mean(diff)

    return hl


# def cum_histogram_loss(noise_bins, ref_bins):
#     # compute cdf of noise bins and ref bins
#     cum_noise_bins = torch.cumsum(noise_bins, dim=0)  # è un tensore R dimensionale (come noise_bins)
#     cum_ref_bins = torch.cumsum(ref_bins, dim=0)
#     diff = (cum_noise_bins - cum_ref_bins) ** 2
#     # print("diff shape: ", diff.shape)
#     hl = torch.sum(diff)

#     return hl