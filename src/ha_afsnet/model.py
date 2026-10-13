"""Context graph encoders, adaptive fusion and temporal event decoder."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RelationalLayer(nn.Module):
    def __init__(self, d, relations, dropout, initialization="tensor_xavier"):
        super().__init__()
        self.message = nn.Parameter(torch.empty(relations, d, d))
        self.self_weight = nn.Parameter(torch.empty(d, d))
        self.relation_weight = nn.Parameter(torch.empty(d, d))
        self.interaction_weight = nn.Parameter(torch.empty(d, d))
        self.activation = nn.RReLU()
        self.dropout = nn.Dropout(dropout)
        for parameter in self.parameters():
            if parameter is self.message and initialization == "per_relation_xavier":
                for matrix in parameter:
                    nn.init.xavier_uniform_(matrix)
            else:
                nn.init.xavier_uniform_(parameter)

    def forward(self, entity, relation, graph):
        # Row-vector convention: multiply by W^T to implement the column-vector equations.
        transformed = entity @ self.self_weight.T
        if graph["source"].numel():
            selected = entity[graph["packed_source"]]
            messages = torch.bmm(
                selected, self.message[graph["active_relations"]].transpose(1, 2)
            )
            messages = messages.reshape(-1, entity.shape[1])[graph["message_index"]]
            transformed = transformed.index_add(
                0, graph["destination"], messages * graph["normalization"][:, None]
            )
            interaction = entity[graph["source"]] * entity[graph["destination"]]
            summed = entity.new_zeros(relation.shape).index_add(
                0, graph["relation"], interaction
            )
            summed = summed / graph["relation_count"][:, None]
        else:
            summed = torch.zeros_like(relation)
        # Relations aggregate interactions from the preceding entity layer.
        relation = (
            relation @ self.relation_weight.T + summed @ self.interaction_weight.T
        )
        entity = self.dropout(self.activation(transformed))
        return entity, relation


class ContextAttention(nn.Module):
    def __init__(self, d, attention):
        super().__init__()
        self.projection = nn.Linear(d, attention)
        self.direction = nn.Parameter(torch.empty(attention))
        nn.init.normal_(self.direction, std=1 / math.sqrt(attention))

    def weights(self, states):
        return (torch.tanh(self.projection(states)) @ self.direction).softmax(dim=0)


class SituationEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activity = nn.LSTM(1, config["activity_width"], batch_first=True)
        self.composition = nn.Linear(1, config["composition_width"])
        self.projection = nn.Linear(
            config["activity_width"] + config["composition_width"],
            config["situation_width"],
        )

    def forward(self, increments, divergence):
        _, (last, _) = self.activity(increments.unsqueeze(-1))
        composition = F.gelu(self.composition(divergence.unsqueeze(-1)))
        return F.gelu(self.projection(torch.cat([last[-1], composition], dim=-1)))


class ConvTransE(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, filters, kernel = (
            config["hidden"],
            config["conv_filters"],
            config["conv_kernel"],
        )
        self.convolution = nn.Conv1d(2, filters, kernel, padding=kernel // 2)
        self.projection = nn.Linear(filters * d, d)
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, subject, relation):
        channels = torch.stack([subject, relation], dim=1)
        features = self.dropout(F.gelu(self.convolution(channels)))
        return self.projection(features.flatten(1))


class HAAFSNet(nn.Module):
    def __init__(self, entities, relations, config):
        super().__init__()
        self.config, self.k, self.d = config, config["contexts"], config["hidden"]
        self.entity = nn.Parameter(torch.empty(entities, self.d))
        self.relation = nn.Parameter(torch.empty(relations, self.d))
        self.context = nn.Parameter(torch.empty(self.k, config["context_width"]))
        for value in [self.entity, self.relation, self.context]:
            nn.init.xavier_uniform_(value)
        self.graph_layers = nn.ModuleList(
            [
                RelationalLayer(
                    self.d,
                    relations,
                    config["dropout"],
                    config.get("relation_initialization", "tensor_xavier"),
                )
                for _ in range(config["layers"])
            ]
        )
        self.entity_attention = ContextAttention(self.d, config["attention_width"])
        self.relation_attention = ContextAttention(self.d, config["attention_width"])
        self.context_weight = nn.Parameter(torch.zeros(config["context_width"]))
        self.context_bias = nn.Parameter(torch.zeros(self.k))
        if config["gate"] == "adaptive":
            self.situation = SituationEncoder(config)
            self.bilinear = nn.Parameter(
                torch.empty(config["situation_width"], config["context_width"])
            )
            self.situation_weight = nn.Parameter(torch.zeros(config["situation_width"]))
            nn.init.xavier_uniform_(self.bilinear)
        self.position = nn.Parameter(torch.empty(config["history"], self.d))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d,
            nhead=config["transformer_heads"],
            dim_feedforward=config["feedforward_width"],
            dropout=config["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            layer, config["transformer_layers"], enable_nested_tensor=False
        )
        if config.get("transformer_initialization", "cloned_default") == "independent":
            # Isolate initialization so other modules and training draws stay fixed.
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(
                    (torch.initial_seed() + 1000003) % (2**63)
                )
                for block in self.transformer.layers:
                    attention = block.self_attn
                    attention.out_proj.reset_parameters()
                    nn.init.xavier_uniform_(attention.in_proj_weight)
                    nn.init.zeros_(attention.in_proj_bias)
                    nn.init.zeros_(attention.out_proj.bias)
                    block.linear1.reset_parameters()
                    block.linear2.reset_parameters()
                    block.norm1.reset_parameters()
                    block.norm2.reset_parameters()
        self.gru = nn.GRU(self.d, self.d, batch_first=True)
        self.decoder = ConvTransE(config)

    def alpha(self, increments, divergence):
        static = self.context @ self.context_weight + self.context_bias
        if self.config["gate"] == "static":
            return static.sigmoid().expand(len(increments), -1)
        situation = self.situation(increments, divergence)
        logits = (
            situation @ self.bilinear @ self.context.T
            + situation @ self.situation_weight[:, None]
            + static
        )
        return logits.sigmoid()

    def snapshot(self, graphs):
        entity_states, relation_states = [], []
        for graph in graphs:
            entity, relation = self.entity, self.relation
            residual = entity
            for layer in self.graph_layers:
                entity, relation = layer(entity, relation, graph)
                residual = residual + entity
            entity_states.append(residual)
            relation_states.append(relation)
        entity, relation = torch.stack(entity_states), torch.stack(relation_states)
        entity_weights = self.entity_attention.weights(entity.mean(dim=1))
        shared_entity = (entity_weights[:, None, None] * entity).sum(dim=0)
        relation_weights = self.relation_attention.weights(relation)
        shared_relation = (relation_weights[:, :, None] * relation).sum(dim=0)
        return entity, relation, shared_entity, shared_relation

    def temporal(self, sequence, transformer):
        # sequence [w, K, items, d] => independent entity/context histories [K*items,w,d].
        w, k, items, d = sequence.shape
        x = sequence.permute(1, 2, 0, 3).reshape(k * items, w, d)
        if transformer:
            x = x + self.position[:w]
            mask = torch.full((w, w), -torch.inf, device=x.device).triu(diagonal=1)
            x = self.transformer(x, mask=mask)[:, -1]
        else:
            x, _ = self.gru(x)
            x = x[:, -1]
        return x.view(k, items, d)

    def encode(self, history_graphs, increments, divergences):
        coefficients = self.alpha(increments, divergences)
        fused_entities, fused_relations = [], []
        for i, graphs in enumerate(history_graphs):
            entity, relation, shared_entity, shared_relation = self.snapshot(graphs)
            alpha = coefficients[i, :, None, None]
            fused_entities.append(alpha * entity + (1 - alpha) * shared_entity[None])
            fused_relations.append(
                alpha * relation + (1 - alpha) * shared_relation[None]
            )
        swap = self.config.get("encoder_swap", False)
        entity = self.temporal(torch.stack(fused_entities), transformer=not swap)
        relation = self.temporal(torch.stack(fused_relations), transformer=swap)
        return entity, relation, coefficients

    def scores(self, entity, relation, triples, contexts, candidates=None):
        query = self.decoder(
            entity[contexts, triples[:, 0]], relation[contexts, triples[:, 1]]
        )
        if candidates is not None:
            target = entity[contexts, candidates]
            return (query * target).sum(dim=-1)
        return torch.bmm(entity[contexts], query.unsqueeze(-1)).squeeze(-1)
