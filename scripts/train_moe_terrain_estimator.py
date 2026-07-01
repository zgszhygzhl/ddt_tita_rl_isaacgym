import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from modules.moe_terrain_estimator import MoeTerrainEstimator, TERRAIN_LABEL_NAMES


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/moe_terrain")
    parser.add_argument("--output", type=str, default="logs/moe_terrain_estimator/model_latest.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--vel_loss_weight", type=float, default=1.0)
    parser.add_argument("--gray_loss_weight", type=float, default=2.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def find_shards(path):
    if os.path.isfile(path):
        return [path]
    shards = sorted(glob.glob(os.path.join(path, "*.pt")))
    if len(shards) == 0:
        raise FileNotFoundError(f"No .pt shards found in: {path}")
    return shards


def load_dataset(path):
    shards = find_shards(path)

    obs_hist_list = []
    vel_label_list = []
    gray_label_list = []
    scenario_list = []

    print("[train] loading shards:")
    for shard in shards:
        print("  ", shard)
        data = torch.load(shard, map_location="cpu")

        obs_hist = data["obs_hist"].float()
        vel_label = data["vel_label"].float()
        gray_label = data["gray_label"].float()

        scenario_id = data.get(
            "scenario_id",
            torch.zeros(obs_hist.shape[0], dtype=torch.long),
        ).long()

        obs_hist_list.append(obs_hist)
        vel_label_list.append(vel_label)
        gray_label_list.append(gray_label)
        scenario_list.append(scenario_id)

    obs_hist = torch.cat(obs_hist_list, dim=0)
    vel_label = torch.cat(vel_label_list, dim=0)
    gray_label = torch.cat(gray_label_list, dim=0)
    scenario_id = torch.cat(scenario_list, dim=0)

    print("[train] dataset:")
    print("  obs_hist:", tuple(obs_hist.shape))
    print("  vel_label:", tuple(vel_label.shape))
    print("  gray_label:", tuple(gray_label.shape))

    for sid in sorted(torch.unique(scenario_id).tolist()):
        count = int((scenario_id == sid).sum().item())
        print(f"  scenario {sid}: {count}")

    return obs_hist, vel_label, gray_label


def weighted_gray_mse(pred, target):
    # order:
    # 0 step_up_score
    # 1 slope_score
    # 2 traction_loss_score
    # 3 instability_score
    # 4 stall_score
    weights = torch.tensor(
        [2.0, 1.0, 2.0, 2.0, 1.5],
        device=pred.device,
        dtype=pred.dtype,
    ).view(1, -1)

    return torch.mean(weights * torch.square(pred - target))


def evaluate(model, loader, device, args):
    model.eval()

    total_loss = 0.0
    total_vel = 0.0
    total_gray = 0.0
    total_n = 0

    gray_sse = None
    gray_n = 0

    with torch.no_grad():
        for obs_hist, vel_label, gray_label in loader:
            obs_hist = obs_hist.to(device)
            vel_label = vel_label.to(device)
            gray_label = gray_label.to(device)

            v_hat, gray_hat = model(obs_hist)

            loss_vel = F.mse_loss(v_hat, vel_label)
            loss_gray = weighted_gray_mse(gray_hat, gray_label)
            loss = args.vel_loss_weight * loss_vel + args.gray_loss_weight * loss_gray

            n = obs_hist.shape[0]
            total_loss += float(loss.item()) * n
            total_vel += float(loss_vel.item()) * n
            total_gray += float(loss_gray.item()) * n
            total_n += n

            err = torch.square(gray_hat - gray_label).sum(dim=0).detach().cpu()
            if gray_sse is None:
                gray_sse = err
            else:
                gray_sse += err
            gray_n += n

    avg_loss = total_loss / max(total_n, 1)
    avg_vel = total_vel / max(total_n, 1)
    avg_gray = total_gray / max(total_n, 1)
    gray_mse_each = gray_sse / max(gray_n, 1)

    return avg_loss, avg_vel, avg_gray, gray_mse_each


def main():
    args = get_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_hist, vel_label, gray_label = load_dataset(args.data)

    if obs_hist.dim() != 3:
        raise RuntimeError(f"Expected obs_hist [N,T,D], got {tuple(obs_hist.shape)}")

    n_proprio = obs_hist.shape[-1]
    history_len = obs_hist.shape[1]

    dataset = TensorDataset(obs_hist, vel_label, gray_label)

    n_total = len(dataset)
    n_val = max(1, int(0.10 * n_total))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = MoeTerrainEstimator(
        n_proprio=n_proprio,
        history_len=history_len,
        hidden_dims=(256, 128, 64),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_vel = 0.0
        total_gray = 0.0
        total_n = 0

        for obs_hist_b, vel_label_b, gray_label_b in train_loader:
            obs_hist_b = obs_hist_b.to(device)
            vel_label_b = vel_label_b.to(device)
            gray_label_b = gray_label_b.to(device)

            v_hat, gray_hat = model(obs_hist_b)

            loss_vel = F.mse_loss(v_hat, vel_label_b)
            loss_gray = weighted_gray_mse(gray_hat, gray_label_b)

            loss = args.vel_loss_weight * loss_vel + args.gray_loss_weight * loss_gray

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n = obs_hist_b.shape[0]
            total_loss += float(loss.item()) * n
            total_vel += float(loss_vel.item()) * n
            total_gray += float(loss_gray.item()) * n
            total_n += n

        train_loss = total_loss / max(total_n, 1)
        train_vel = total_vel / max(total_n, 1)
        train_gray = total_gray / max(total_n, 1)

        val_loss, val_vel, val_gray, gray_each = evaluate(model, val_loader, device, args)

        print(
            f"[epoch {epoch:03d}] "
            f"train={train_loss:.6f} vel={train_vel:.6f} gray={train_gray:.6f} | "
            f"val={val_loss:.6f} vel={val_vel:.6f} gray={val_gray:.6f}"
        )

        msg = "  gray_mse:"
        for name, mse in zip(TERRAIN_LABEL_NAMES, gray_each.tolist()):
            msg += f" {name}={mse:.5f}"
        print(msg)

        ckpt = {
            "model_state_dict": model.state_dict(),
            "n_proprio": n_proprio,
            "history_len": history_len,
            "label_names": TERRAIN_LABEL_NAMES,
            "epoch": epoch,
            "val_loss": val_loss,
        }

        torch.save(ckpt, args.output)

        if val_loss < best_val:
            best_val = val_loss
            best_path = args.output.replace(".pt", "_best.pt")
            torch.save(ckpt, best_path)
            print("[train] saved best:", best_path)

    print("[train] saved latest:", args.output)


if __name__ == "__main__":
    main()
