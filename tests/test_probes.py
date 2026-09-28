import numpy as np
import torch
import torch.nn.functional as F

from jepa_lptem.eval.probes import AttentiveProbe, CentroidProbe, decode_centroids


def test_attentive_probe_overfits_tiny_batch():
    torch.manual_seed(0)
    tokens = torch.randn(8, 256, 128)  # (B, n_tokens, dim) one latent frame
    target = torch.randn(8, 3)  # [count, mean_r, total_area]
    probe = AttentiveProbe(dim=128, out_dim=3)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    loss = None
    for _ in range(400):
        loss = F.mse_loss(probe(tokens), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 1e-2


def test_centroid_probe_output_shapes():
    probe = CentroidProbe(dim=128, grid=16)
    tokens = torch.randn(4, 256, 128)  # 256 = 16*16 tokens
    occ, off = probe(tokens)
    assert occ.shape == (4, 16, 16)
    assert off.shape == (4, 16, 16, 2)


def test_decode_centroids_recovers_offcenter():
    # pure-math check of the sub-cell offset decode (no training)
    occ = np.zeros((16, 16))
    off = np.zeros((16, 16, 2))
    occ[5, 7] = 1.0
    off[5, 7] = (0.3, -0.2)  # (dy, dx) in cell-fraction units
    cents = decode_centroids(occ, off, threshold=0.5, cell_size=16.0)
    assert cents.shape == (1, 2)
    expected = np.array([(5 + 0.5 + 0.3) * 16, (7 + 0.5 - 0.2) * 16])  # (row, col) px
    assert np.linalg.norm(cents[0] - expected) < 1.0


def test_centroid_probe_learns_a_target_field():
    # can the probe represent + learn a specific occupancy+offset field, and does
    # the decoded centroid then land within 1 px of the intended location?
    torch.manual_seed(0)
    B, dim, grid, cell = 4, 128, 16, 16.0
    tokens = torch.randn(B, grid * grid, dim)
    cell_r, cell_c, dy, dx = 5, 7, 0.3, -0.2

    occ_t = torch.zeros(B, grid, grid)
    occ_t[:, cell_r, cell_c] = 1.0
    off_t = torch.zeros(B, grid, grid, 2)
    off_t[:, cell_r, cell_c, 0] = dy
    off_t[:, cell_r, cell_c, 1] = dx

    probe = CentroidProbe(dim=dim, grid=grid)
    opt = torch.optim.Adam(probe.parameters(), lr=5e-3)
    for _ in range(500):
        occ_logits, off = probe(tokens)
        loss = F.binary_cross_entropy_with_logits(occ_logits, occ_t)
        loss = loss + F.mse_loss(off[occ_t.bool()], off_t[occ_t.bool()])
        opt.zero_grad()
        loss.backward()
        opt.step()

    occ_logits, off = probe(tokens)
    prob = torch.sigmoid(occ_logits)[0].detach().numpy()
    off0 = off[0].detach().numpy()
    cents = decode_centroids(prob, off0, threshold=0.5, cell_size=cell)
    expected = np.array([(cell_r + 0.5 + dy) * cell, (cell_c + 0.5 + dx) * cell])
    assert cents.shape[0] == 1
    assert np.linalg.norm(cents[0] - expected) < 1.0


def test_decode_centroids_keeps_one_detection_per_blob():
    # A trained probe lights up every cell a particle covers, not just its centre
    # cell. Only the local maximum may decode, or one particle becomes several.
    occ = np.zeros((16, 16))
    occ[4:7, 4:7] = 0.7
    occ[5, 5] = 0.95          # blob: centre plus its 8 above-threshold neighbours
    occ[12, 12] = 0.8         # a second, separate particle
    cents = decode_centroids(occ, np.zeros((16, 16, 2)), threshold=0.5, cell_size=16.0)
    assert sorted(map(tuple, cents)) == [(88.0, 88.0), (200.0, 200.0)]
