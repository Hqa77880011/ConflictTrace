import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from .data import PacketStore, prepare, stress
from .model import PAIRS, SOURCES, ConflictTrace, jsd, objective


def construct(schema, args):
    return ConflictTrace(schema, args.width, args.variant, args.beta, args.kappa, args.tau).to(args.device)


def summarize(probability, target):
    p, y = np.asarray(probability), np.asarray(target)
    prediction, confidence = p.argmax(1), p.max(1)
    correct = prediction == y
    order = np.argsort(confidence)
    bins = [indices for indices in np.array_split(order, min(15, len(y))) if len(indices)]
    ece = sum(len(indices) / len(y) * abs(confidence[indices].mean() - correct[indices].mean()) for indices in bins)
    return dict(macro_f1=float(f1_score(y, prediction, average="macro", zero_division=0)),
                nll=float(-np.log(p[np.arange(len(y)), y].clip(1e-8)).mean()),
                brier=float(np.mean(np.sum((p - np.eye(p.shape[1])[y]) ** 2, 1))),
                ece_percent=float(ece * 100), high_confidence_error_percent=float(100 * np.mean((~correct) & (confidence >= 0.9))))


@torch.no_grad()
def validation(model, store, rows, device):
    model.eval()
    probability = []
    for decision in rows:
        snapshot = store.snapshot(decision, decision["time"])
        output = model.fuse(*store.arrays(model, snapshot, decision["time"], device))
        probability.append(output["probability"].cpu().numpy())
    return summarize(probability, [row["label"] for row in rows])


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    store = PacketStore(args.data)
    model = construct(store.manifest["schema"], args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    training = [row for row in store.manifest["decisions"] if row["split"] == "train"]
    development = [row for row in store.manifest["decisions"] if row["split"] == "val"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history, best_f1, best_ece, waiting = [], -1, np.inf, 0
    for epoch in range(args.epochs):
        model.train()
        minibatch, losses = [], []
        for index in rng.permutation(len(training)):
            decision = training[index]
            snapshot = store.snapshot(decision, decision["time"])
            arrays = store.arrays(model, snapshot, decision["time"], args.device)
            full = model.fuse(*arrays)
            available = np.flatnonzero(arrays[3].cpu().numpy())
            removed = int(rng.choice(available)) if len(available) else 0
            mask = arrays[3].clone()
            mask[removed] = 0
            partial = model.fuse(*arrays[:3], mask)
            minibatch.append(objective(full, partial, decision["label"], removed, epoch, args))
            if len(minibatch) >= args.batch_size:
                value = torch.stack(minibatch).mean()
                optimizer.zero_grad()
                value.backward()
                optimizer.step()
                losses.append(float(value.detach()))
                minibatch = []
        if minibatch:
            value = torch.stack(minibatch).mean()
            optimizer.zero_grad()
            value.backward()
            optimizer.step()
            losses.append(float(value.detach()))
        metrics = validation(model, store, development, args.device)
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), **metrics))
        improvement = metrics["macro_f1"] > best_f1 + 0.001 or (abs(metrics["macro_f1"] - best_f1) <= 0.001 and metrics["ece_percent"] < best_ece)
        if improvement:
            best_f1, best_ece, waiting = metrics["macro_f1"], metrics["ece_percent"], 0
            torch.save(dict(state=model.state_dict(), schema=store.manifest["schema"], config=vars(args)), output / "model.pt")
        else:
            waiting += 1
            if waiting >= args.patience:
                break
    (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")


@torch.no_grad()
def replay(args):
    store = PacketStore(args.data)
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=True)
    options = argparse.Namespace(**checkpoint["config"])
    options.device = args.device
    model = construct(checkpoint["schema"], options)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    rows = sorted([row for row in store.manifest["decisions"] if row["split"] == args.split], key=lambda row: (row["run"], row["time"]))
    issued, targets, traces, latencies = [], [], [], []
    conflicts, conflict_truth, quality_scores, quality_truth = [], [], [], []
    regret, variations, flips, action_changes, changes = [], [], [], [], []
    encoder_calls, late_packets = 0, 0
    for decision in rows:
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        cache = {}
        initial_snapshot = store.snapshot(decision, decision["time"])
        initial = model.fuse(*store.arrays(model, initial_snapshot, decision["time"], args.device, cache))
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
        issued.append(initial["probability"].cpu().numpy())
        targets.append(decision["label"])
        relevant = [packet for packet in store.by_run.get(decision["run"], [])
                    if decision["time"] - store.manifest["context"] < packet["acquisition"] <= decision["time"]]
        deliveries = sorted(set(packet["delivery"] for packet in relevant if decision["time"] < packet["delivery"] <= decision["time"] + args.history))
        late_packets += sum(packet["delivery"] > decision["time"] + args.history for packet in relevant)
        snapshots = [(decision["time"], initial_snapshot, initial)]
        previous_ids = [packet["id"] if packet else None for packet in initial_snapshot]
        for arrival in deliveries:
            snapshot = store.snapshot(decision, arrival)
            identities = [packet["id"] if packet else None for packet in snapshot]
            if identities == previous_ids:
                continue
            updated = model.fuse(*store.arrays(model, snapshot, decision["time"], args.device, cache))
            snapshots.append((arrival, snapshot, updated))
            previous_ids = identities
        encoder_calls += len(cache)
        full_snapshot = store.snapshot(decision, np.inf)
        full = model.fuse(*store.arrays(model, full_snapshot, decision["time"], args.device, cache))
        truth = torch.eye(model.classes, device=args.device)[decision["label"]]
        full_error = (full["probability"] - truth).square().sum()
        sequence = []
        variation, flip_count, alert_count = 0.0, 0, 0
        previous = None
        for arrival, snapshot, result in snapshots:
            p = result["probability"]
            regret.append(float(torch.relu((p - truth).square().sum() - full_error)))
            if previous is not None:
                variation += float(jsd(previous, p))
                flip_count += int(previous.argmax() != p.argmax())
                old_action = int(previous.argmax()) if previous.max() >= 0.9 and previous.argmax() != 0 else 0
                new_action = int(p.argmax()) if p.max() >= 0.9 and p.argmax() != 0 else 0
                alert_count += int(old_action != new_action)
                changes.append(int(old_action != new_action))
            for source, packet in enumerate(snapshot):
                if packet is not None:
                    quality_scores.append(1 - float(result["reliability"][source]))
                    quality_truth.append(int(packet["damaged"]))
            for pair_index, (first, second) in enumerate(PAIRS):
                a, b = snapshot[first], snapshot[second]
                if a is None or b is None or a["damaged"] or b["damaged"]:
                    continue
                positive = (a["contradictory"] or b["contradictory"]) and a["content_label"] != b["content_label"]
                if positive or not (a["contradictory"] or b["contradictory"]):
                    conflict_truth.append(int(positive))
                    distance = float(jsd(result["discounted_probability"][first], result["discounted_probability"][second]) / np.log(2))
                    conflicts.append(float(result["relation"][pair_index, 2]) * distance)
            sequence.append(dict(delivery=arrival, probability=p.cpu().tolist(), uncertainty=float(result["uncertainty"]),
                                 reliability=result["reliability"].cpu().tolist(), conflict=float(result["conflict"]),
                                 redundancy=float(result["redundancy"]), relations=result["relation"].cpu().tolist(),
                                 packets=[packet["id"] if packet else None for packet in snapshot]))
            previous = p
        variations.append(variation)
        flips.append(flip_count)
        action_changes.append(alert_count)
        traces.append(dict(run=decision["run"], time=decision["time"], label=decision["label"], revisions=sequence,
                           full_probability=full["probability"].cpu().tolist()))
    if not issued:
        raise ValueError("No decisions in requested split")
    result = summarize(issued, targets)
    result.update(config=vars(args), variant=options.variant,
                  conflict_auroc=float(roc_auc_score(conflict_truth, conflicts)) if len(set(conflict_truth)) == 2 else None,
                  reliability_auroc=float(roc_auc_score(quality_truth, quality_scores)) if len(set(quality_truth)) == 2 else None,
                  revision_regret=float(np.mean(regret)), cumulative_prediction_variation=float(np.mean(variations)),
                  mean_class_flips=float(np.mean(flips)), action_change_rate=float(np.mean(changes)) if changes else 0,
                  mean_action_changes=float(np.mean(action_changes)), latency_ms=float(np.mean(latencies) * 1000),
                  parameters=sum(parameter.numel() for parameter in model.parameters()), available_encoder_calls=encoder_calls,
                  late_packet_decision_pairs=late_packets,
                  risk_coverage=[dict(coverage=float(fraction), risk=float(1 - np.mean(np.asarray(issued).argmax(1)[indices] == np.asarray(targets)[indices])))
                                 for fraction in np.arange(0.1, 1.01, 0.1)
                                 for indices in [np.argsort(np.asarray(issued).max(1))[::-1][:max(1, round(fraction * len(targets)))]]])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("metrics", result), ("revisions", traces)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("prepare")
    data.add_argument("--packets", required=True)
    data.add_argument("--decisions", required=True)
    data.add_argument("--output", required=True)
    data.add_argument("--context", type=float, default=60)
    corruption = sub.add_parser("stress")
    corruption.add_argument("--data", required=True)
    corruption.add_argument("--output", required=True)
    corruption.add_argument("--split", choices=["train", "val", "test"], default="test")
    corruption.add_argument("--kind", choices=["missing", "outage", "delay", "quality", "contradiction", "mixed"], required=True)
    corruption.add_argument("--rate", type=float, default=0.3)
    corruption.add_argument("--delay", type=float, default=30)
    corruption.add_argument("--source", choices=SOURCES, default="video")
    corruption.add_argument("--start", type=float, default=0)
    corruption.add_argument("--duration", type=float, default=60)
    corruption.add_argument("--seed", type=int, default=13)
    training = sub.add_parser("train")
    training.add_argument("--data", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--variant", choices=["full", "mean", "evidence_sum", "no_reliability", "no_conflict", "no_redundancy", "no_revision"], default="full")
    for name, default in (("width", 128), ("epochs", 100), ("batch-size", 64), ("patience", 15), ("seed", 13)):
        training.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("lr", 0.0003), ("weight-decay", 0.0001), ("beta", 2), ("kappa", 1), ("tau", 30),
                          ("revision-weight", 0.1), ("brier-weight", 0.1), ("revision-eta", 1), ("conflict-margin", 0.1)):
        training.add_argument(f"--{name}", type=float, default=default)
    training.add_argument("--device", default="cpu")
    evaluation = sub.add_parser("replay")
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--split", choices=["val", "test"], default="test")
    evaluation.add_argument("--history", type=float, default=60)
    evaluation.add_argument("--device", default="cpu")
    plotting = sub.add_parser("plot")
    plotting.add_argument("--metrics", nargs="+", required=True)
    plotting.add_argument("--clean")
    plotting.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "stress":
        if not 0 <= args.rate <= 1 or args.delay < 0 or args.duration < 0:
            parser.error("Invalid missingness, delay or outage parameters")
        stress(args)
    elif args.command == "train":
        if min(args.width, args.epochs, args.batch_size, args.patience, args.tau) <= 0:
            parser.error("Positive training and age-scale parameters required")
        train(args)
    elif args.command == "replay":
        if args.history < 0:
            parser.error("Nonnegative revision history required")
        replay(args)
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.metrics]
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        degradation = []
        clean = json.loads(Path(args.clean).read_text(encoding="utf-8")) if args.clean else None
        for metric in ("macro_f1", "ece_percent", "high_confidence_error_percent", "conflict_auroc", "revision_regret", "cumulative_prediction_variation"):
            selected = [row for row in records if row[metric] is not None]
            figure, axis = plt.subplots(figsize=(8, 4))
            axis.bar([f"{row['variant']}:{Path(row['config']['data']).name}" for row in selected], [row[metric] for row in selected])
            axis.tick_params(axis="x", rotation=30)
            axis.set_ylabel(metric)
            figure.tight_layout()
            figure.savefig(output / f"{metric}.png", dpi=180)
            plt.close(figure)
        if clean:
            for row in records:
                if row["variant"] != clean["variant"]:
                    raise ValueError("Clean-minus-stressed degradation requires the same variant")
                degradation.append(dict(data=row["config"]["data"], macro_f1_degradation=clean["macro_f1"] - row["macro_f1"]))
            (output / "degradation.json").write_text(json.dumps(degradation, indent=2), encoding="utf-8")
