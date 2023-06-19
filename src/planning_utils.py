import torch
import torch.nn as nn

from models.world_model import WorldModel


def expand_world_model_embedding(world_model: WorldModel) -> None:
    # we need to modify the Embedder of the world model, because 
    # now the there are two more actions added.
    config = world_model.config

    embedding_tables = world_model.embedder.embedding_tables
    act_vocab_size, obs_vocab_size = (
        embedding_tables[0].weight.shape[0],
        embedding_tables[1].weight.shape[0]
    )

    assert act_vocab_size == world_model.act_vocab_size

    vocab_embedding = nn.Embedding(act_vocab_size + 2, config.embed_dim)
    vocab_embedding.weight.data[:act_vocab_size] = world_model.embedder.embedding_tables[0].weight.data

    new_embedding_tables = nn.ModuleList([
        vocab_embedding,
        embedding_tables[1]
    ]).to(next(world_model.parameters()).device)

    world_model.embedder.embedding_tables = new_embedding_tables
