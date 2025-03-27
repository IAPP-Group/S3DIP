""" Histogram loss  Implementation (no broadcasting and pytorch tensors)"""

import numpy as np
import pickle
import matplotlib.pyplot as plt


# h =  vettore dei bin dell'istogramma del rumore empirico che vogliamo ottimizzare
# (N che inizializzo in qualche modo (matrice di tutti 1 / rumore gaussiano))
# h_rif = vettore dei bin dell'istogramma del rumore di riferimento Nrif...


def compute_h(noise, t):
    height, width = noise.shape
    step = t[1] - t[0]

    # versione margini
    R = len(t)  # bins == margini
    bins = []
    for r in range(R):
        # calcolo singolo hr
        delta_sum = 0
        print("current r: ", r)
        for i in range(height):
            for j in range(width):
                # calcolo delta i,j,r
                delta = 0
                print("Pixel: ", i, j, noise[i, j])
                if r - 1 != -1 and t[r - 1] <= noise[i, j] <= t[r]:
                    delta = noise[i, j] - t[r - 1]
                elif r + 1 < R and t[r] <= noise[i, j] <= t[r + 1]:  # r + 1 < len(t) and
                    delta = t[r + 1] - noise[i, j]
                # deltas.append(delta)  # per controlli
                delta_sum += (delta / step)
        # hi.append(delta_sum)
        hr = delta_sum / (height * width)
        bins.append(hr)

    return bins  # , deltas, hi


def compute_h_new(noise, t):
    height, width = noise.shape
    step = t[1] - t[0]

    R = len(t)
    # inizializzo un vettore di tanti zeri quanti sono i margini
    bins = np.zeros(R)

    for i in range(height):
        for j in range(width):
            # print("Pixel: ", i, j, noise[i, j])
            # find bins for the pixel
            b = np.abs(noise[i, j] - edges)
            ind_bins = np.argwhere(b <= step)
            # calcolo i contributi ai bins
            red = np.abs((b[ind_bins] - step)) / step
            bins[ind_bins] += red

    bins = bins / (height * width)
    return bins


# istogramma h empirico (calcolato con compute_h) e istogramma del rumore di riferimento
# per adesso usiamo U generato da matlab e 'campionato' con lo stesso istogramma fissato qui ...
def histogram_loss(noise_bins, ref_bins):
    hl = 0
    for r in range(len(noise_bins)):  # R
        diff = (noise_bins[r] - ref_bins[r]) ** 2  # (hr^ - hr)^2
        hl += diff

    return hl


if __name__ == '__main__':
    import torch
    import numpy as np
    # from hist_loss_broadcast import compute_h_broadcast
    from utils_mat.common_utils_mat import np_to_torch
    import matplotlib.pyplot as plt
    import scipy.io
    import argparse
    import time

    # # mat = np.array([[1, 2, 4.5], [3, 1.6, 9]])
    # mat = scipy.io.loadmat('C:/Users/chiar/Desktop/sar_data/new_data/u_sqrt_50_50.mat')
    # mat = mat['u_sqrt']
    # _, edges = np.histogram(mat, bins=20)
    # step = edges[1] - edges[0]
    # start = time.time()
    # h_ref = compute_h(mat, edges)
    # end = time.time()
    #
    # start1 = time.time()
    # h_ref1 = compute_h_new(mat, edges)
    # end1 = time.time()
    # print("no cycle solution time: ", end1 - start1)
    # print("cycle solution time: ", end - start)

    parser = argparse.ArgumentParser(description="Histogram loss without pytorch broadcasting")

    parser.add_argument("--mat", dest="mat", default='./data/SAR/pickle/new_data/u_sqrt_10000_10000.mat',
                        help="path to pickle file")
    parser.add_argument("--base_save_path", dest="base_save_path",
                        default='./data/SAR/pickle/new_data/hists_no_broadcast',
                        help="path to the base folder for histograms pckl")
    parser.add_argument("--n_bins", dest="n_bins", default=None, help="number of bins for the histogram")

    args = parser.parse_args()

    # mat = scipy.io.loadmat('./data/SAR/pickle/new_data/u_sqrt_10000_10000.mat')  # u sqrt
    mat = scipy.io.loadmat(str(args.mat))
    # u_or = mat['u_sqrt']
    # u_or = mat['U_sqrt']
    # per test gaussian noise
    u_or = mat['u_ref']
    u = np_to_torch(u_or)  # 1, 10000, 10000

    # edges
    n_bins = int(args.n_bins)
    _, edges = np.histogram(u_or, bins=n_bins)
    edges_t = np_to_torch(edges)
    step = (edges_t[0, 1] - edges_t[0, 0]).item()

    # h_ref = compute_h(u_or, edges)

    start = time.time()
    h_ref = compute_h_new(u_or, edges)
    end = time.time()

    # h_ref1 = compute_h_broadcast(u, edges_t, step)  # confronto con l'istogramma calcolato con la versione broadcast

    print("sum h_ref: ", sum(h_ref))
    print("time of execution: ", end - start)

    # save
    # 'C:/Users/chiar/PycharmProjects/LAB/DIP_SAR_images/data/SAR/pickle/new_data/hists_no_broadcast
    f = open(str(args.base_save_path) + '/h_ref_' + str(n_bins) + '.pckl', 'wb')
    # per salvarli come se fossere prodotti da compute_h_broadcast
    h_ref_tens = np.array(h_ref)
    h_ref_tens = torch.from_numpy(h_ref_tens)
    pickle.dump([h_ref_tens, edges_t], f)
    f.close()

    # plot
    # margins = edges.tolist()
    # h_ref = h_ref.tolist()
    #
    # plt.vlines(x=margins, ymin=0, ymax=max(h_ref))
    # plt.plot(margins, h_ref, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=2,
    #          label='h_ref')
    #
    # plt.xlabel('Margins')
    # plt.ylabel('Bin heights')
    # plt.grid()
    # plt.legend()
    # plt.show()

