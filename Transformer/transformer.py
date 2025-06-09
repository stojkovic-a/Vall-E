import torch
import torch.nn as nn

# import torch.optim as optim
# import torch.utils.data as data
import math

# import copy


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert (
            d_model % num_heads == 0
        ), "d_model must be divisible by number of heads so that each head would get the same share of model features"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None
    ):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            attn_scores = attn_scores.masked_fill(mask == 0, -math.inf)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x: torch.Tensor):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class PostNormEncoderLayer(nn.Module):
    ###TODO::: Izdvojiti normalizaciju u posebnu fju ko za PreNormEncoderLayer
    def __init__(self, d_model, num_heads, d_ff, dropout, norm="instance"):
        super(PostNormEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm = norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.condition1 = None
        self.condition2 = None
        if norm == "instance":
            pass
        elif norm == "adaln":
            self.condition1 = nn.Linear(d_model, 2 * d_model)
            self.condition2 = nn.Linear(d_model, 2 * d_model)
        else:
            raise "Not implemented"
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, norm_vector=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        if self.norm == "adaln":
            scale_bias = self.condition1(norm_vector)
            scale, bias = torch.chunk(scale_bias, 2, dim=-1)
            scale = scale.unsqueeze(1)
            bias = bias.unsqueeze(1)
            x = scale * x + bias
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        if self.norm == "adaln":
            scale_bias = self.condition2(norm_vector)
            scale, bias = torch.chunk(scale_bias, 2, dim=-1)
            scale = scale.unsqueeze(1)
            bias = bias.unsqueeze(1)
            x = scale * x + bias
        return x


class PreNormEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, norm="instance"):
        super(PreNormEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm = norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.condition1 = None
        self.condition2 = None
        if norm == "instance":
            pass
        elif norm == "adaln":
            self.condition1 = nn.Linear(d_model, 2 * d_model)
            self.condition2 = nn.Linear(d_model, 2 * d_model)
        else:
            raise "Not implemented"
        self.dropout = nn.Dropout(dropout)

    def _normalize(self, x, norm_vector=None, condition=None):
        normed = self.norm1(x)
        if self.norm == "adaln":
            scale_bias = condition(norm_vector)
            scale, bias = torch.chunk(scale_bias, 2, dim=-1)
            scale = scale.unsqueeze(1)
            bias = bias.unsqueeze(1)
            normed = scale * normed + bias
        return normed

    def forward(self, x, mask, norm_vector=None):
        normed = self._normalize(x, norm_vector, self.condition1)
        attn_output = self.self_attn(normed, normed, normed, mask)
        x = x + self.dropout(attn_output)

        normed = self._normalize(x, norm_vector, self.condition2)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout(ff_output)
        return x


class PostNormDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, norm="instance"):
        super(PostNormDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm = norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.condition1 = None
        self.condition2 = None
        self.condition3 = None
        if norm == "instance":
            pass
        elif norm == "adaln":
            self.condition1 = nn.Linear(d_model, 2 * d_model)
            self.condition2 = nn.Linear(d_model, 2 * d_model)
            self.condition3 = nn.Linear(d_model, 2 * d_model)
        else:
            raise "Not Implemendted"
        self.dropout = nn.Dropout(dropout)

    def _normalize(self, x, norm_vector=None, condition=None):
        normed = self.norm1(x)
        if self.norm == "adaln":
            scale_bias = condition(norm_vector)
            scale, bias = torch.chunk(scale_bias, 2, dim=-1)
            scale = scale.unsqueeze(1)
            bias = bias.unsqueeze(1)
            normed = scale * normed + bias
        return normed

    def forward(self, x, enc_output, src_mask, tgt_mask, norm_vector=None):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self._normalize(x + self.dropout(attn_output), norm_vector, self.condition1)
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self._normalize(x + self.dropout(attn_output), norm_vector, self.condition2)
        ff_output = self.feed_forward(x)
        x = self._normalize(x + self.dropout(ff_output), norm_vector, self.condition3)
        return x


class PreNormDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, norm="instance"):
        super(PreNormDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm = norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.condition1 = None
        self.condition2 = None
        self.condition3 = None
        if norm == "instance":
            pass
        elif norm == "adaln":
            self.condition1 = nn.Linear(d_model, 2 * d_model)
            self.condition2 = nn.Linear(d_model, 2 * d_model)
            self.condition3 = nn.Linear(d_model, 2 * d_model)
        else:
            raise "Not Implemendted"
        self.dropout = nn.Dropout(dropout)

    def _normalize(self, x, norm_vector=None, condition=None):
        normed = self.norm1(x)
        if self.norm == "adaln":
            scale_bias = condition(norm_vector)
            scale, bias = torch.chunk(scale_bias, 2, dim=-1)
            scale = scale.unsqueeze(1)
            bias = bias.unsqueeze(1)
            normed = scale * normed + bias
        return normed

    def forward(self, x, enc_output, src_mask, tgt_mask, norm_vector=None):
        normed = self._normalize(x, norm_vector, self.condition1)
        attn_output = self.self_attn(normed, normed, normed, tgt_mask)
        x = x + self.dropout(attn_output)

        normed = self._normalize(x, norm_vector, self.condition2)
        attn_output = self.cross_attn(normed, enc_output, enc_output, src_mask)
        x = x + self.dropout(attn_output)

        normed = self._normalize(x, norm_vector, self.condition3)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout(ff_output)
        return x


class CasualPostNormDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, norm="instance"):
        super(CasualPostNormDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm = norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.condition1 = None
        self.condition2 = None
        if norm == "instance":
            pass
        elif norm == "adaln":
            self.condition1 = nn.Linear(d_model, 2 * d_model)
            self.condition2 = nn.Linear(d_model, 2 * d_model)
        else:
            raise Exception("Not implemented")
        self.dropout = nn.Dropout(dropout)

    def _normalize(self, x, norm_vector=None, condition=None):
        normed = self.norm1(x)
        if self.norm == "adaln":
            scale_bias = condition(norm_vector)
            scale, bias = torch.chunk(scale_bias, 2, dim=-1)
            scale = scale.unsqueeze(1)
            bias = bias.unsqueeze(1)
            normed = scale * normed + bias
        return normed

    def forward(self, x, tgt_mask, norm_vector=None):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self._normalize(x + self.dropout(attn_output), norm_vector, self.condition1)
        ff_output = self.feed_forward(x)
        x = self._normalize(x + self.dropout(ff_output), norm_vector, self.condition2)
        return x


class CasualPreNormDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, norm="instance"):
        super(CasualPreNormDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm = norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.condition1 = None
        self.condition2 = None
        if norm == "instance":
            pass
        elif norm == "adaln":
            self.condition1 = nn.Linear(d_model, 2 * d_model)
            self.condition2 = nn.Linear(d_model, 2 * d_model)
        else:
            raise Exception("Not implemented")
        self.dropout = nn.Dropout(dropout)

    def _normalize(self, x, norm_vector=None, condition=None):
        normed = self.norm1(x)
        if self.norm == "adaln":
            scale_bias = condition(norm_vector)
            scale, bias = torch.chunk(scale_bias, 2, dim=-1)
            scale = scale
            bias = bias
            normed = scale * normed + bias
        return normed

    def forward(self, x, tgt_mask, norm_vector=None):
        normed = self._normalize(x, norm_vector, self.condition1)
        attn_output = self.self_attn(normed, normed, normed, tgt_mask)
        x = x + self.dropout(attn_output)

        normed = self._normalize(x, norm_vector, self.condition2)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout(ff_output)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    ):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList(
            [
                PostNormEncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                PostNormDecoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src: torch.Tensor, tgt: torch.Tensor):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)
        ).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        # if self.training:
        #     src_mask, tgt_mask = self.generate_mask(src, tgt)
        # else:
        #     src_mask, tgt_mask = None

        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(
            self.positional_encoding(self.encoder_embedding(src))
        )
        tgt_embedded = self.dropout(
            self.positional_encoding(self.decoder_embedding(tgt))
        )

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


class ARVALLE(nn.Module):
    def __init__(
        self,
        phoneme_vocab_size,
        acoustic_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    ):
        super(ARVALLE, self).__init__()
        self.phonemes_embedding = nn.Embedding(phoneme_vocab_size, d_model)
        self.acoustic_embedding = nn.Embedding(acoustic_vocab_size, d_model)
        self.phonemes_positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.acoustic_positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.decoder_layers = nn.ModuleList(
            [
                CasualPreNormDecoderLayer(
                    d_model, num_heads, d_ff, dropout, norm="instance"
                )
                for _ in range(num_layers)
            ]
        )
        self.output_projection = nn.Linear(d_model, acoustic_vocab_size, bias=False)
        self.output_projection.weight = self.acoustic_embedding.weight
        self.dropout = nn.Dropout(dropout)

    def generate_mask(
        self,
        phonemes: torch.Tensor,
        acoustic_tokens: torch.Tensor,
        acoustic_sample_length: int,
    ):
        padding_mask = (
            ((torch.cat([phonemes, acoustic_tokens], dim=1)) != 0)
            .unsqueeze(1)
            .unsqueeze(1)
        )
        _, phonemes_length = phonemes.shape
        _, acoustic_length = acoustic_tokens.shape
        seq_length = phonemes_length + acoustic_length
        mask = torch.ones((1, seq_length, seq_length)).to("cuda")
        # mask = torch.ones((1, seq_length, seq_length)).to("cpu")
        mask[
            ...,
            0 : phonemes_length + acoustic_sample_length,
            phonemes_length + acoustic_sample_length :,
        ] = 0
        mask[
            ...,
            phonemes_length + acoustic_sample_length :,
            phonemes_length + acoustic_sample_length :,
        ] = torch.tril(
            torch.ones(
                1,
                acoustic_length - acoustic_sample_length,
                acoustic_length - acoustic_sample_length,
            ),
            diagonal=0,
        )
        return mask.bool() & padding_mask.bool()

    def forward(self, phonemes, acoustic_tokens, acoustic_sample_length):
        """
        Mora da bude vec dodat <EOS> na kraj phonemes i na kraj acoustinc_tokens
        """
        mask = self.generate_mask(phonemes, acoustic_tokens, acoustic_sample_length)
        phonemes_embedded = self.dropout(
            self.phonemes_positional_encoding(self.phonemes_embedding(phonemes))
        )
        acoustic_embedded = self.dropout(
            self.acoustic_positional_encoding(self.acoustic_embedding(acoustic_tokens))
        )
        dec_output = torch.cat([phonemes_embedded, acoustic_embedded], dim=1)
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, mask)

        output = self.output_projection(dec_output)
        return output


class NARVALLE(nn.Module):
    def __init__(
        self,
        phoneme_vocab_size,
        acoustic_vocab_size,
        num_stages,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    ):
        """
        Posto ne pise u rad, radicu tako da imamo phonemes sequence sa jedan positional encoding i acoustic sequence sa jedan
        positional encoding i na kraj svaki sequence po <EOS>
        """
        super(NARVALLE, self).__init__()
        # self.acoustic_embedding = nn.Embedding(acoustic_vocab_size, d_model)
        self.phonemes_positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.acoustic_positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.num_stages = num_stages
        self.phonemes_embedding = nn.Embedding(phoneme_vocab_size, d_model)
        self.stage_embedding = nn.Embedding(num_stages, d_model)
        self.acoustic_token_embeddings = nn.ModuleList(
            [nn.Embedding(acoustic_vocab_size, d_model) for _ in range(num_stages)]
        )

        self.decoder_layers = nn.ModuleList(
            [
                CasualPreNormDecoderLayer(
                    d_model, num_heads, d_ff, dropout, norm="adaln"
                )
                for _ in range(num_layers)
            ]
        )
        self.output_projections = nn.ModuleList(
            [
                nn.Linear(d_model, acoustic_vocab_size, bias=False)
                for _ in range(num_layers)
            ]
        )
        for i in range(num_stages):
            self.output_projections[i].weight = self.acoustic_token_embeddings[i].weight

        self.dropout = nn.Dropout(dropout)

    def generate_mask(
        self,
        phonemes: torch.Tensor,
        acoustic_prompt: torch.Tensor,
        acoustic_tokens: torch.Tensor,
    ):
        padding_mask = (
            (
                (
                    torch.cat(
                        [
                            phonemes,
                            acoustic_prompt[:, 0, :].squeeze(1),
                            acoustic_tokens[:, 0, :].squeeze(1),
                        ],
                        dim=1,
                    )
                )
                != 0
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )
        return padding_mask

    def forward(self, phonemes, acoustic_prompt, acoustic_tokens, stage):
        """
        Radi jednostavnosti podrazumevam da se acoustic_tokens salje sa <EOS> token na kraj i da phonemes ima <EOS> token na kraj
        Sta
        """
        mask = self.generate_mask(phonemes, acoustic_prompt, acoustic_tokens)
        phonemes_embedded = self.dropout(
            self.phonemes_positional_encoding(self.phonemes_embedding(phonemes))
        )
        acoustic_prompt_embedded = self.acoustic_token_embeddings[0](
            acoustic_prompt[:, 0, ...]
        )
        for i in range(1, self.num_stages):
            acoustic_prompt_embedded += self.acoustic_token_embeddings[i](
                acoustic_prompt[:, i, ...]
            )

        acoustic_token_embedded = self.acoustic_token_embeddings[0](
            acoustic_tokens[:, 0, ...]
        )
        for i in range(1, stage):
            acoustic_token_embedded += self.acoustic_token_embeddings[i](
                acoustic_tokens[:, i, ...]
            )
        acoustic_embedded = torch.cat(
            [acoustic_prompt_embedded, acoustic_token_embedded], dim=1
        )
        acoustic_embedded = self.dropout(
            self.acoustic_positional_encoding(acoustic_embedded)
        )
        stage_embedded = self.dropout(self.stage_embedding(stage))
        dec_output = torch.cat([phonemes_embedded, acoustic_embedded], dim=1)
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, mask, stage_embedded)

        output = self.output_projections[stage](dec_output)
        return output


if __name__ == "__main__":
    seq_length = 6
    mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    print(mask)
    mask2 = torch.tril(torch.ones(1, seq_length, seq_length), diagonal=0)
    print(mask2)
