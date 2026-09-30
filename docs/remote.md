# Remote training environment (work PC)

Set up 2026-09-29. Code is edited on the Mac only; the PC pulls, runs, and results come
back by `rsync`. Nothing is ever committed on the PC.

## Machine

| | |
|---|---|
| GPUs | 2× RTX 3090 (24 GB), driver 595.84, CUDA 13.2 |
| CPU / RAM | 24 cores, 60 GiB RAM, 8 GiB swap |
| Account | `mlmonster` — **shared** with other people's projects |
| Repo | `/mnt/drive2/niki-masters/jepa-lptem` |
| Upstream code | `code/LPTEMsimulator` (gitignored; the generator needs its `MTFmatrix512.mat`) |
| Encoder weights | `~/.cache/huggingface/` (downloaded on first use, ~1.4 GB) |

**Shared machine.** At setup, other users' jobs occupied GPU 0 (a long-running output-generation
job, ~8 GB, 80–86 % utilisation) and a vLLM server started by `root` held ~1.9 GB on *each*
GPU, idle on GPU 1. Swap was nearly full. Every run so far went on GPU 1. Check before each
launch; see [Etiquette](#etiquette).

## Access

**Mac → PC:** plain SSH as `mlmonster`. An alias in the Mac's `~/.ssh/config` saves typing
the address and port (fill in the real values):

```
Host trainbox
  HostName <pc-address>
  User mlmonster
  Port <port>          # only if not 22
```

**PC → GitHub: a read-only deploy key**, not the personal key. The account is shared, so a
copied key must be worth as little as possible: this one can read this one repo and nothing
else, and deleting it on GitHub (repo → Settings → Deploy keys) revokes the PC with no
access to the PC needed. Key `~/.ssh/id_ed25519_niki_jepa_thesis`, alias in the PC's
`~/.ssh/config`:

```
Host github-niki-jepa
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_niki_jepa_thesis
  IdentitiesOnly yes
```

`IdentitiesOnly` stops ssh offering another key first and authenticating as the wrong
account ("Repository not found"). `ssh -T git@github-niki-jepa` must greet
`NikolaJankovikj/jepa-lptem` — the repo name, which is how a deploy key answers.

## Setup (once)

```bash
sudo mkdir -p /mnt/drive2/niki-masters && sudo chown $USER:$USER /mnt/drive2/niki-masters
cd /mnt/drive2/niki-masters
git clone git@github-niki-jepa:NikolaJankovikj/jepa-lptem.git && cd jepa-lptem
git clone https://github.com/JamaliLab/LPTEMsimulator.git code/LPTEMsimulator
uv sync
```

`sudo` only for the directory; never `sudo git`/`sudo uv`: root reads `/root/.ssh` (the
alias does not resolve) and leaves root-owned files every later command trips over.

## Running a job

Always from the repo root, always detached, always unbuffered and logged:

```bash
git pull
export CUDA_DEVICE_ORDER=PCI_BUS_ID          # CUDA's numbering = nvidia-smi's
tmux new -d -s <name> -c "$PWD" \
  'CUDA_VISIBLE_DEVICES=1 uv run python -u scripts/<script>.py ... 2>&1 | tee logs/<name>.log'
tail -f logs/<name>.log                      # Ctrl-c stops watching, not the job
```

- **`-u` is not optional.** Piped into `tee`, Python buffers stdout and the log stays empty for
  hours — indistinguishable from a hang (this happened on the first Gate-1 launch).
- **`CUDA_VISIBLE_DEVICES`** picks the card; the scripts just ask for `cuda`.
- **tmux prefix.** The Mac runs herdr, which also takes `Ctrl-b`, so the PC's tmux never sees
  it. Launch with `-d` and there is nothing to detach from; to work inside tmux, set
  `set -g prefix C-a` in the PC's `~/.tmux.conf`. Closing the laptop or dropping SSH does not
  stop a tmux job.
- The `VIRTUAL_ENV=... does not match` warning is another project's venv auto-activated on
  the shared account. uv ignores it and uses `.venv`; harmless. Never pass `--active`.

## Monitoring

```bash
tmux ls                                   # our sessions, and others' — kill only ours by name
nvidia-smi                                # which card, memory, utilisation
nvidia-smi pmon -i 1 -c 10                # per-process activity on GPU 1
ps -o user,pid,etime,cmd -p <pid>         # who owns a process nvidia-smi shows
```

Measured so far: dataset generation ~3 s/clip on one core (~0.8 GB RAM); feature extraction
100 % GPU at ~2 GB; probe training ~10 % GPU because it is disk/decompression-bound (~4 h
for Gate 1).

## Results back to the Mac

On the Mac, **from inside the local `jepa-lptem` folder** (a relative destination lands
wherever you run it — the first transfer landed one level too high):

```bash
rsync -avz trainbox:/mnt/drive2/niki-masters/jepa-lptem/experiments/<name>/ experiments/<name>/
```

Without the alias: `-e "ssh -p <port>" mlmonster@<pc-address>:...`. Keep the trailing
slashes. Checkpoints (`*.pt`) stay out of git; `metrics.json` and the write-up are committed
from the Mac.

## Etiquette

- Before launching: `nvidia-smi` — use a card with no other active process; leave the busy
  one alone (two jobs on one 24 GB card can both die of OOM). Check `free -h` too.
- `tmux kill-session -t <ours>` only. **Never `tmux kill-server`**: the account shares one
  tmux server, so it ends everyone's sessions.
- `rm -rf` only inside our repo. Tell whoever runs the vLLM server when GPU 1 will be busy
  for hours: sharing a card roughly halves its throughput while both are active.
