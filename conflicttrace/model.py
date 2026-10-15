import itertools
import math

import torch
from torch import nn
from torch.nn import functional as F


SOURCES = ("process", "video", "transient", "logs")
PAIRS = tuple(itertools.combinations(range(4), 2))


def jsd(first, second):
    middle = (first + second) / 2
    return 0.5 * ((first * ((first + 1e-8).log() - (middle + 1e-8).log())).sum(-1)
                  + (second * ((second + 1e-8).log() - (middle + 1e-8).log())).sum(-1))


class ProcessEncoder(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv1d(channels, width, 3, padding=1), nn.ReLU(),
                                    nn.Conv1d(width, width, 3, padding=1), nn.ReLU())

    def forward(self, value):
        return self.layers(value.T[None]).mean(-1)[0]


class VideoEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, width // 2, 3, stride=2, padding=1), nn.ReLU(),
                                    nn.Conv2d(width // 2, width, 3, stride=2, padding=1), nn.ReLU())

    def forward(self, value):
        return self.layers(value.permute(0, 3, 1, 2) / 255).mean((0, 2, 3))


class LogEncoder(nn.Module):
    def __init__(self, vocabulary, width):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, width, padding_idx=0)
        self.gru = nn.GRU(width, width, batch_first=True)

    def forward(self, value):
        _, hidden = self.gru(self.embedding(value.long())[None])
        return hidden[0, 0]


class ConflictTrace(nn.Module):
    def __init__(self, schema, width=128, variant="full", beta=2, kappa=1, tau=30):
        super().__init__()
        self.classes = schema["classes"]
        self.variant, self.beta, self.kappa, self.tau = variant, beta, kappa, tau
        self.encoders = nn.ModuleList([ProcessEncoder(schema["process_channels"], width), VideoEncoder(width),
                                       nn.Sequential(nn.Linear(schema["transient_features"], width), nn.ReLU(), nn.Linear(width, width)),
                                       LogEncoder(schema["vocabulary"], width)])
        self.evidence = nn.ModuleList([nn.Linear(width, self.classes) for _ in SOURCES])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(width + 5, width // 2), nn.ReLU(), nn.Linear(width // 2, 1)) for _ in SOURCES])
        self.relation = nn.Sequential(nn.Linear(2 * width + 3, width), nn.ReLU(), nn.Linear(width, 3))
        self.width = width

    def encode(self, value, source):
        return self.encoders[source](value)

    def fuse(self, latents, qualities, ages, mask):
        evidence = torch.stack([F.softplus(head(latents[index])) for index, head in enumerate(self.evidence)])
        total = evidence.sum(-1)
        uncertainty = self.classes / (total + self.classes)
        belief = evidence / (total + self.classes)[:, None]
        probability = belief + uncertainty[:, None] / self.classes
        disagreement = torch.stack([sum(jsd(probability[index], probability[other]) * mask[other] for other in range(4) if other != index)
                                     / (mask.sum() - mask[index]).clamp_min(1) for index in range(4)])
        gate_input = torch.cat([latents, qualities, torch.log1p(ages)[:, None], disagreement[:, None]], -1)
        reliability = torch.stack([torch.sigmoid(gate(gate_input[index]))[0] for index, gate in enumerate(self.gates)]) * mask
        if self.variant == "no_reliability":
            reliability = mask
        discounted_u = 1 - reliability + reliability * uncertainty
        discounted_p = reliability[:, None] * belief + discounted_u[:, None] / self.classes
        discounted_e = self.classes * (1 - discounted_u) / discounted_u.clamp_min(1e-8)
        relation, confidence, distance = [], [], []
        for first, second in PAIRS:
            features = torch.cat([(latents[first] - latents[second]).abs(), latents[first] * latents[second],
                                  torch.stack([(ages[first] - ages[second]).abs(), torch.minimum(ages[first], ages[second]),
                                               torch.maximum(ages[first], ages[second])]) / self.tau])
            relation.append(self.relation(features).softmax(-1))
            confidence.append((1 - discounted_u[first]) * (1 - discounted_u[second]))
            distance.append(jsd(discounted_p[first], discounted_p[second]) / math.log(2))
        relation, confidence, distance = torch.stack(relation), torch.stack(confidence), torch.stack(distance)
        conflict = (confidence * relation[:, 2] * distance).sum() / confidence.sum().clamp_min(1e-8)
        redundancy = (confidence * relation[:, 1] * (1 - distance)).sum() / confidence.sum().clamp_min(1e-8)
        if self.variant == "no_conflict":
            conflict = conflict * 0
        if self.variant == "no_redundancy":
            redundancy = redundancy * 0
        weight = torch.exp(-ages / self.tau) * (1 - discounted_u)
        direction = (weight[:, None] * discounted_p).sum(0) / weight.sum().clamp_min(1e-8)
        direction = torch.where(weight.sum() > 0, direction, torch.full_like(direction, 1 / self.classes))
        mean_evidence = (weight * discounted_e).sum() / weight.sum().clamp_min(1e-8)
        effective = (weight.sum().square() / weight.square().sum().clamp_min(1e-8)).clamp_min(1)
        concentration = torch.exp(-self.beta * conflict) * mean_evidence * (1 + self.kappa * (1 - redundancy) * (effective - 1))
        if self.variant == "mean":
            p = (mask[:, None] * probability).sum(0) / mask.sum().clamp_min(1)
            p = torch.where(mask.sum() > 0, p, torch.full_like(p, 1 / self.classes))
            alpha = p / p.min().clamp_min(1e-8)
            fused_u = self.classes / alpha.sum()
        elif self.variant == "evidence_sum":
            alpha = 1 + (mask[:, None] * evidence).sum(0)
            p, fused_u = alpha / alpha.sum(), self.classes / alpha.sum()
        else:
            alpha = 1 + concentration * direction
            p, fused_u = alpha / alpha.sum(), self.classes / alpha.sum()
        return dict(probability=p, uncertainty=fused_u, alpha=alpha, source_alpha=evidence + 1,
                    reliability=reliability, relation=relation, conflict=conflict, redundancy=redundancy,
                    discounted_probability=discounted_p, mask=mask)


def dirichlet_loss(alpha, label, anneal):
    classes = alpha.shape[-1]
    truth = F.one_hot(torch.as_tensor(label, device=alpha.device), classes).float()
    prediction = (truth * (torch.digamma(alpha.sum(-1, keepdim=True)) - torch.digamma(alpha))).sum(-1)
    unsupported = truth + (1 - truth) * alpha
    total = unsupported.sum(-1)
    kl = torch.lgamma(total) - torch.lgamma(unsupported).sum(-1) - torch.lgamma(total.new_tensor(float(classes)))
    kl = kl + ((unsupported - 1) * (torch.digamma(unsupported) - torch.digamma(total[..., None]))).sum(-1)
    return prediction + anneal * kl


def objective(full, partial, label, removed, epoch, args):
    prediction = dirichlet_loss(full["alpha"], label, min(1, (epoch + 1) / 10))
    prediction = prediction + 0.1 * (dirichlet_loss(full["source_alpha"], label, min(1, (epoch + 1) / 10)) * full["mask"]).sum() / full["mask"].sum().clamp_min(1)
    truth = F.one_hot(torch.as_tensor(label, device=full["alpha"].device), len(full["alpha"])).float()
    brier = (full["probability"] - truth).square().sum()
    reliability = full["reliability"][removed]
    allowance = reliability * jsd(full["discounted_probability"][removed], partial["probability"])
    move = F.relu(jsd(full["probability"], partial["probability"]) - args.revision_eta * allowance)
    pairs = [index for index, pair in enumerate(PAIRS) if removed in pair and bool(full["mask"][pair[0]] > 0) and bool(full["mask"][pair[1]] > 0)]
    contradiction = full["relation"][pairs, 2].mean().detach() if pairs else full["uncertainty"].new_zeros(())
    uncertainty = contradiction * F.relu(torch.minimum(full["uncertainty"].new_tensor(1), partial["uncertainty"] + args.conflict_margin * full["conflict"]) - full["uncertainty"])
    uncertainty = uncertainty + (1 - contradiction) * reliability * F.relu(full["uncertainty"] - partial["uncertainty"])
    revision = 0 if args.variant == "no_revision" else args.revision_weight * (move + uncertainty)
    return prediction + args.brier_weight * brier + revision
