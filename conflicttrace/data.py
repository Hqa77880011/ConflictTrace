import copy
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .model import SOURCES


def quality(value, source, statistics):
    finite = np.isfinite(value)
    missing = 1 - float(finite.mean())
    if source == "video":
        exposure = float(((value <= 2) | (value >= 253)).mean())
        sharpness = np.mean([cv2.Laplacian(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for frame in value])
        return [exposure, float(1 / (1 + sharpness)), 1 - missing]
    if source == "logs":
        return [float(np.mean(value > 0)), min(len(value) / 128, 1), 1 - missing]
    lower, upper = np.asarray(statistics[source]["lower"]), np.asarray(statistics[source]["upper"])
    saturation = float(np.mean(((value <= lower) | (value >= upper)) & finite))
    return [missing, saturation, 1 - missing]


def prepare(args):
    source = Path(args.packets)
    with source.open(encoding="utf-8-sig", newline="") as stream:
        packets = list(csv.DictReader(stream))
    with Path(args.decisions).open(encoding="utf-8-sig", newline="") as stream:
        decisions = list(csv.DictReader(stream))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    values, groups = [], {}
    for row in packets:
        if row["source"] not in SOURCES or row["split"] not in ("train", "val", "test"):
            raise ValueError("Unknown source or split")
        if row["run"] in groups and groups[row["run"]] != row["split"]:
            raise ValueError("Physical runs cannot cross splits")
        groups[row["run"]] = row["split"]
        acquisition, delivery = float(row["acquisition"]), float(row["delivery"])
        if delivery < acquisition:
            raise ValueError("Delivery cannot precede acquisition")
        value = np.load(source.parent / row["path"], allow_pickle=False)
        expected_dimensions = dict(process=2, video=4, transient=1, logs=1)
        if value.ndim != expected_dimensions[row["source"]] or value.size == 0:
            raise ValueError(f"Incorrect packet shape for {row['source']}")
        if row["source"] == "video" and value.shape[-1] != 3:
            raise ValueError("Video packets use RGB channel-last frames")
        values.append(value)
    statistics = {}
    for name in ("process", "transient"):
        selected = [value.reshape(-1, value.shape[-1]) for value, row in zip(values, packets) if row["source"] == name and row["split"] == "train"]
        if not selected:
            raise ValueError(f"Training {name} packets required")
        joined = np.concatenate(selected)
        mean, std = np.nanmean(joined, 0), np.nanstd(joined, 0).clip(1e-6)
        lower, upper = np.nanquantile(joined, [0.01, 0.99], axis=0)
        if not np.isfinite(mean).all():
            raise ValueError(f"Every {name} channel needs training observations")
        statistics[name] = dict(mean=mean.tolist(), scale=std.tolist(), lower=((lower - mean) / std).tolist(), upper=((upper - mean) / std).tolist())
    vocabulary = max((int(value.max()) for value, row in zip(values, packets) if row["source"] == "logs" and row["split"] == "train"), default=0) + 1
    if vocabulary < 2:
        raise ValueError("Training logs need positive token IDs; zero is the unknown/padding token")
    schema = dict(classes=max(int(row["label"]) for row in decisions) + 1, vocabulary=vocabulary,
                  process_channels=len(statistics["process"]["mean"]), transient_features=len(statistics["transient"]["mean"]))
    records = []
    for index, (row, value) in enumerate(zip(packets, values)):
        name = row["source"]
        if name in statistics:
            stat = statistics[name]
            value = (value - np.asarray(stat["mean"])) / np.asarray(stat["scale"])
        elif name == "logs":
            value = np.where((value >= 0) & (value < vocabulary), value, 0).astype(np.int64)
        path = f"packet_{index:07d}.npy"
        q = quality(value, name, statistics)
        np.save(output / path, value.astype(np.float32))
        records.append(dict(id=index, file=path, source=name, acquisition=float(row["acquisition"]),
                            delivery=float(row["delivery"]), run=row["run"], split=row["split"],
                            label=int(row["label"]), content_label=int(row["label"]), quality=q, damaged=False, contradictory=False))
    decision_rows = []
    for row in decisions:
        run, split, end = row["run"], row["split"], float(row["time"])
        if groups.get(run) != split:
            raise ValueError("Decision and packet run splits must agree")
        label = int(row["label"])
        context = [packet for packet in records if packet["run"] == run and end - args.context < packet["acquisition"] <= end]
        if any(packet["label"] != label for packet in context):
            continue
        decision_rows.append(dict(run=run, split=split, time=end, label=label))
    if set(row["split"] for row in decision_rows) != {"train", "val", "test"}:
        raise ValueError("Nonempty train/validation/test decision splits required")
    labels = sorted(set(row["label"] for row in decision_rows))
    if labels != list(range(schema["classes"])):
        raise ValueError("Consecutive class IDs starting at zero required")
    manifest = dict(packets=records, decisions=decision_rows, schema=schema, statistics=statistics, context=args.context)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


class PacketStore:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.packets = self.manifest["packets"]
        self.by_run = {}
        for packet in self.packets:
            self.by_run.setdefault(packet["run"], []).append(packet)

    def snapshot(self, decision, cutoff):
        eligible = [packet for packet in self.by_run.get(decision["run"], [])
                    if decision["time"] - self.manifest["context"] < packet["acquisition"] <= decision["time"]
                    and packet["delivery"] <= cutoff]
        return [max((packet for packet in eligible if packet["source"] == name), key=lambda packet: (packet["acquisition"], packet["delivery"], packet["id"]), default=None)
                for name in SOURCES]

    def arrays(self, model, snapshot, decision_time, device, cache=None):
        cache = {} if cache is None else cache
        latents, qualities, ages, masks = [], [], [], []
        for source, packet in enumerate(snapshot):
            if packet is None:
                latents.append(torch.zeros(model.width, device=device))
                qualities.append([0, 0, 0])
                ages.append(0)
                masks.append(0)
            else:
                if packet["id"] not in cache:
                    value = torch.as_tensor(np.nan_to_num(np.load(self.root / packet["file"], allow_pickle=False), nan=0, posinf=0, neginf=0), device=device)
                    cache[packet["id"]] = model.encode(value, source)
                latents.append(cache[packet["id"]])
                qualities.append(packet["quality"])
                ages.append(max(0, decision_time - packet["acquisition"]))
                masks.append(1)
        return (torch.stack(latents), torch.tensor(qualities, device=device, dtype=torch.float32),
                torch.tensor(ages, device=device, dtype=torch.float32), torch.tensor(masks, device=device, dtype=torch.float32))


def stress(args):
    store = PacketStore(args.data)
    manifest = copy.deepcopy(store.manifest)
    rng = np.random.default_rng(args.seed)
    packets = manifest["packets"]
    eligible = [index for index, packet in enumerate(packets) if packet["split"] == args.split]
    removed, degraded, contradictory = set(), set(), set()
    if args.kind in ("missing", "mixed"):
        removed = set(rng.choice(eligible, round((0.3 if args.kind == "mixed" else args.rate) * len(eligible)), replace=False).tolist())
    if args.kind == "outage":
        removed = {index for index in eligible if packets[index]["source"] == args.source and args.start <= packets[index]["acquisition"] < args.start + args.duration}
    eligible = [index for index in eligible if index not in removed]
    if args.kind in ("delay", "mixed"):
        for index in eligible:
            packets[index]["delivery"] += 30 if args.kind == "mixed" else args.delay
    if args.kind in ("quality", "mixed"):
        degraded = set(rng.choice(eligible, round((0.3 if args.kind == "mixed" else args.rate) * len(eligible)), replace=False).tolist())
    if args.kind in ("contradiction", "mixed"):
        contradictory = set(rng.choice(eligible, round((0.3 if args.kind == "mixed" else args.rate) * len(eligible)), replace=False).tolist())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    updated = []
    for index, packet in enumerate(packets):
        if index in removed:
            continue
        value = np.load(store.root / packet["file"], allow_pickle=False)
        if index in contradictory:
            donors = [candidate for candidate in store.packets if candidate["split"] == args.split and candidate["source"] == packet["source"]
                      and candidate["label"] != packet["label"] and candidate["run"] != packet["run"]]
            if donors:
                donor = min(donors, key=lambda candidate: np.linalg.norm(np.asarray(candidate["quality"]) - packet["quality"]))
                value = np.load(store.root / donor["file"], allow_pickle=False)
                packet["content_label"], packet["contradictory"] = donor["label"], True
        if index in degraded:
            name = packet["source"]
            if name == "video":
                value = np.stack([cv2.GaussianBlur(frame, (9, 9), 3) for frame in value])
            elif name in store.manifest["statistics"]:
                stat = store.manifest["statistics"][name]
                value = np.clip(value, np.asarray(stat["lower"]), np.asarray(stat["upper"]))
            elif name == "logs":
                value = value.copy()
                value[::2] = 0
            packet["damaged"] = True
        packet["quality"] = quality(value, packet["source"], store.manifest["statistics"])
        np.save(output / packet["file"], value)
        updated.append(packet)
    manifest["packets"] = updated
    manifest["stress"] = dict(config=vars(args), original_eligible=len([packet for packet in store.packets if packet["split"] == args.split]),
                              removed=len(removed), damaged=sum(packet["damaged"] for packet in updated),
                              contradictory=sum(packet["contradictory"] for packet in updated))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
