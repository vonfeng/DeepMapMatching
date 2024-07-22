"""Sequence to Sequence models."""
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np


class StackedAttentionLSTM(nn.Module):
    """Deep Attention LSTM."""

    def __init__(
            self,
            input_size,
            rnn_size,
            num_layers,
            batch_first=True,
            dropout=0.
    ):
        """Initialize params."""
        super(StackedAttentionLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.batch_first = batch_first

        self.layers = []
        for i in range(num_layers):
            layer = LSTMAttentionDot(
                input_size, rnn_size, batch_first=self.batch_first
            )
            self.add_module('layer_%d' % i, layer)
            self.layers += [layer]
            input_size = rnn_size

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the layer."""
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            if ctx_mask is not None:
                ctx_mask = torch.ByteTensor(
                    ctx_mask.data.cpu().numpy().astype(np.int32).tolist()
                ).cuda()
            output, (h_1_i, c_1_i) = layer(input, (h_0, c_0), ctx, ctx_mask)

            input = output

            if i != len(self.layers):
                input = self.dropout(input)

            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class DeepBidirectionalLSTM(nn.Module):
    r"""A Deep LSTM with the first layer being bidirectional."""

    def __init__(
            self, input_size, hidden_size,
            num_layers, dropout, batch_first
    ):
        """Initialize params."""
        super(DeepBidirectionalLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.num_layers = num_layers

        self.bi_encoder = nn.LSTM(
            self.input_size,
            self.hidden_size // 2,
            1,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout
        )

        self.encoder = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            self.num_layers - 1,
            bidirectional=False,
            batch_first=True,
            dropout=self.dropout
        )

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.encoder.batch_first else input.size(1)
        h0_encoder_bi = Variable(torch.zeros(
            2,
            batch_size,
            self.hidden_size // 2
        ))
        c0_encoder_bi = Variable(torch.zeros(
            2,
            batch_size,
            self.hidden_size // 2
        ))

        h0_encoder = Variable(torch.zeros(
            self.num_layers - 1,
            batch_size,
            self.hidden_size
        ))

        c0_encoder = Variable(torch.zeros(
            self.num_layers - 1,
            batch_size,
            self.hidden_size
        ))

        return (h0_encoder_bi.cuda(), c0_encoder_bi.cuda()), \
               (h0_encoder.cuda(), c0_encoder.cuda())

    def forward(self, input):
        """Propogate input forward through the network."""
        hidden_bi, hidden_deep = self.get_state(input)
        bilstm_output, (_, _) = self.bi_encoder(input, hidden_bi)
        return self.encoder(bilstm_output, hidden_deep)


class LSTMAttention(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, context_size):
        """Initialize params."""
        super(LSTMAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_layers = 1

        self.input_weights_1 = nn.Parameter(
            torch.Tensor(4 * hidden_size, input_size)
        )
        self.hidden_weights_1 = nn.Parameter(
            torch.Tensor(4 * hidden_size, hidden_size)
        )
        self.input_bias_1 = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.hidden_bias_1 = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.input_weights_2 = nn.Parameter(
            torch.Tensor(4 * hidden_size, context_size)
        )
        self.hidden_weights_2 = nn.Parameter(
            torch.Tensor(4 * hidden_size, hidden_size)
        )
        self.input_bias_2 = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.hidden_bias_2 = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.context2attention = nn.Parameter(
            torch.Tensor(context_size, context_size)
        )
        self.bias_context2attention = nn.Parameter(torch.Tensor(context_size))

        self.hidden2attention = nn.Parameter(
            torch.Tensor(context_size, hidden_size)
        )

        self.input2attention = nn.Parameter(
            torch.Tensor(input_size, context_size)
        )

        self.recurrent2attention = nn.Parameter(torch.Tensor(context_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        stdv_ctx = 1.0 / math.sqrt(self.context_size)

        self.input_weights_1.data.uniform_(-stdv, stdv)
        self.hidden_weights_1.data.uniform_(-stdv, stdv)
        self.input_bias_1.data.fill_(0)
        self.hidden_bias_1.data.fill_(0)

        self.input_weights_2.data.uniform_(-stdv_ctx, stdv_ctx)
        self.hidden_weights_2.data.uniform_(-stdv, stdv)
        self.input_bias_2.data.fill_(0)
        self.hidden_bias_2.data.fill_(0)

        self.context2attention.data.uniform_(-stdv_ctx, stdv_ctx)
        self.bias_context2attention.data.fill_(0)

        self.hidden2attention.data.uniform_(-stdv_ctx, stdv_ctx)
        self.input2attention.data.uniform_(-stdv_ctx, stdv_ctx)

        self.recurrent2attention.data.uniform_(-stdv_ctx, stdv_ctx)

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""

        def recurrence(input, hidden, projected_input, projected_ctx):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim

            gates = F.linear(
                input, self.input_weights_1, self.input_bias_1
            ) + F.linear(hx, self.hidden_weights_1, self.hidden_bias_1)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim

            # Attention mechanism

            # Project current hidden state to context size
            hidden_ctx = F.linear(hy, self.hidden2attention)

            # Added projected hidden state to each projected context
            hidden_ctx_sum = projected_ctx + hidden_ctx.unsqueeze(0).expand(
                projected_ctx.size()
            )

            # Add this to projected input at this time step
            hidden_ctx_sum = hidden_ctx_sum + \
                             projected_input.unsqueeze(0).expand(hidden_ctx_sum.size())

            # Non-linearity
            hidden_ctx_sum = F.tanh(hidden_ctx_sum)

            # Compute alignments
            alpha = torch.bmm(
                hidden_ctx_sum.transpose(0, 1),
                self.recurrent2attention.unsqueeze(0).expand(
                    hidden_ctx_sum.size(1),
                    self.recurrent2attention.size(0),
                    self.recurrent2attention.size(1)
                )
            ).squeeze()
            alpha = F.softmax(alpha)
            weighted_context = torch.mul(
                ctx, alpha.t().unsqueeze(2).expand(ctx.size())
            ).sum(0).squeeze()

            gates = F.linear(
                weighted_context, self.input_weights_2, self.input_bias_2
            ) + F.linear(hy, self.hidden_weights_2, self.hidden_bias_2)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cy) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim

            return hy, cy

        input = input.transpose(0, 1)
        projected_ctx = torch.bmm(
            ctx,
            self.context2attention.unsqueeze(0).expand(
                ctx.size(0),
                self.context2attention.size(0),
                self.context2attention.size(1)
            ),
        )
        projected_ctx += \
            self.bias_context2attention.unsqueeze(0).unsqueeze(0).expand(
                projected_ctx.size()
            )

        projected_input = torch.bmm(
            input,
            self.input2attention.unsqueeze(0).expand(
                input.size(0),
                self.input2attention.size(0),
                self.input2attention.size(1)
            ),
        )

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(
                input[i], hidden, projected_input[i], projected_ctx
            )
            output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return output, hidden


class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim, attn_type='dot'):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        # self.linear_in = nn.Linear(dim, dim, bias=False)
        # self.sm = nn.Softmax()
        # self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        # self.tanh = nn.Tanh()
        # self.mask = None

        self.dim = dim
        self.attn_type = attn_type
        assert (self.attn_type in ["dot", "general", "mlp"]), (
            "Please select a valid attention type.")

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        self.sm = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context):
        """Propogate input through the network.

        input: batch x dim
        context: batch x sourceL x dim
        """
        # target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        # attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL

        tgt_len = 1
        h_s, h_t = context, input
        src_batch, src_len = context.shape[0], context.shape[1]
        tgt_batch = src_batch
        tgt_dim = self.dim

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_dim, tgt_len)
            # (batch, s_len, d) x (batch, d, t_len) --> (batch, s_len, t_len)
            else:
                h_t = h_t.view(tgt_batch, tgt_dim, tgt_len)
            attn = torch.bmm(h_s, h_t).squeeze(2)
        elif self.attn_type == "mlp":
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = self.tanh(wq + uh)

            attn = self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)
            attn = attn.transpose(1, 2).squeeze(2)  # tg_len = 1

        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn


class LSTMAttentionDot(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, batch_first=True, attn_type='dot'):
        """Initialize params."""
        super(LSTMAttentionDot, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

        self.attention_layer = SoftDotAttention(hidden_size, attn_type)

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""

        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim

            h_tilde, alpha = self.attention_layer(hy, ctx.transpose(0, 1))

            return h_tilde, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            # output.append(isinstance(hidden, tuple) and hidden[0] or hidden)
            output.append(hidden[0])

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden


class Seq2Seq(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(
            self,
            src_loc_emb_dim,
            src_tim_emb_dim,
            trg_seg_emb_dim,
            src_loc_vocab_size,
            src_tim_vocab_size,
            trg_seg_vocab_size,
            src_hidden_dim,
            trg_hidden_dim,
            batch_size,
            pad_token_src_loc,
            pad_token_src_tim1,
            pad_token_src_tim2,
            pad_token_trg,
            bidirectional=True,
            nlayers_src=2,
            nlayers_trg=1,
            dropout=0.,
            time_encoding='NoEncoding',
    ):
        """Initialize model."""
        super(Seq2Seq, self).__init__()
        self.src_loc_vocab_size = src_loc_vocab_size
        self.src_tim_vocab_size = src_tim_vocab_size
        self.trg_seg_vocab_size = trg_seg_vocab_size
        self.src_loc_emb_dim = src_loc_emb_dim
        self.src_tim_emb_dim = src_tim_emb_dim
        self.trg_seg_emb_dim = trg_seg_emb_dim
        self.time_encoding = time_encoding
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.nlayers_src = nlayers_src
        self.nlayers_trg = nlayers_trg
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        self.pad_token_src_loc = pad_token_src_loc
        self.pad_token_src_tim1 = pad_token_src_tim1
        self.pad_token_src_tim2 = pad_token_src_tim2
        self.pad_token_trg = pad_token_trg
        self.src_hidden_dim = src_hidden_dim // 2 if self.bidirectional else src_hidden_dim

        self.src_embedding = nn.Embedding(
            self.src_loc_vocab_size,
            self.src_loc_emb_dim,
            self.pad_token_src_loc
        )
        self.trg_embedding = nn.Embedding(
            self.trg_seg_vocab_size,
            self.trg_seg_emb_dim,
            self.pad_token_trg
        )
        if self.time_encoding == 'OneEncoding':
            self.src_time_embedding = nn.Embedding(
                self.src_tim_vocab_size[0],
                self.src_tim_emb_dim[0],
                self.pad_token_src_tim1
            )
        elif self.time_encoding == 'TwoEncoding':
            self.src_time_embedding_1 = nn.Embedding(
                self.src_tim_vocab_size[1][0],
                self.src_tim_emb_dim[1][0],
                self.pad_token_src_tim2[0]
            )
            self.src_time_embedding_2 = nn.Embedding(
                self.src_tim_vocab_size[1][1],
                self.src_tim_emb_dim[1][1],
                self.pad_token_src_tim2[1]
            )

        if self.time_encoding == 'NoEncoding':
            src_emb_dim = self.src_loc_emb_dim
        elif self.time_encoding == 'OneEncoding':
            src_emb_dim = self.src_loc_emb_dim + self.src_tim_emb_dim[0]
        elif self.time_encoding == 'TwoEncoding':
            src_emb_dim = self.src_loc_emb_dim + self.src_tim_emb_dim[1][0] + self.src_tim_emb_dim[1][1]
        else:
            raise RuntimeError('error time encoding')
        self.encoder = nn.LSTM(
            src_emb_dim,
            self.src_hidden_dim,
            self.nlayers_src,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=self.dropout
        )

        self.decoder = nn.LSTM(
            self.trg_seg_emb_dim,
            self.trg_hidden_dim,
            self.nlayers_trg,
            dropout=self.dropout,
            batch_first=True
        )

        self.encoder2decoder = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            self.trg_hidden_dim
        )
        self.decoder2vocab = nn.Linear(self.trg_hidden_dim, trg_seg_vocab_size).cuda()

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.encoder.batch_first else input.size(1)
        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ))
        c0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ))

        return h0_encoder.cuda(), c0_encoder.cuda()

    def get_decoder_state(self, input):
        """Get decoder cell states and hidden states."""
        batch_size = input.size(0) if self.decoder.batch_first else input.size(1)
        h0_decoder = Variable(torch.zeros(
            self.decoder.num_layers - 1,
            batch_size,
            self.trg_hidden_dim
        ))
        c0_decoder = Variable(torch.zeros(
            self.decoder.num_layers - 1,
            batch_size,
            self.trg_hidden_dim
        ))

        return h0_decoder.cuda(), c0_decoder.cuda()

    def forward(self, input_src, input_trg, input_time=None, ctx_mask=None, trg_mask=None):
        """Propogate input through the network."""
        src_emb = self.src_embedding(input_src)
        trg_emb = self.trg_embedding(input_trg)
        if self.time_encoding == 'NoEncoding':
            src_time_emb = src_emb
        elif self.time_encoding == 'OneEncoding':
            time_emb = self.src_time_embedding(input_time)
            src_time_emb = torch.cat((src_emb, time_emb), dim=2)
        elif self.time_encoding == 'TwoEncoding':
            time_emb_1 = self.src_time_embedding_1(input_time[0])
            time_emb_2 = self.src_time_embedding_2(input_time[1])
            src_time_emb = torch.cat((src_emb, time_emb_1, time_emb_2), dim=2)
        else:
            raise RuntimeError('error time encoding')

        self.h0_encoder, self.c0_encoder = self.get_state(input_src)

        src_h, (src_h_t, src_c_t) = self.encoder(src_time_emb, (self.h0_encoder, self.c0_encoder))

        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))
        if self.nlayers_trg > 1:
            self.h0_decoder, self.c0_decoder = self.get_decoder_state(input_src)
            self.h0_decoder = torch.cat(
                (decoder_init_state.view(1, decoder_init_state.size(0), decoder_init_state.size(1)), self.h0_decoder),
                0)
            self.c0_decoder = torch.cat((c_t.view(1, c_t.size(0), c_t.size(1)), self.c0_decoder), 0)
        else:
            self.h0_decoder = decoder_init_state.view(1, decoder_init_state.size(0), decoder_init_state.size(1))
            self.c0_decoder = c_t.view(1, c_t.size(0), c_t.size(1))

        trg_h, (_, _) = self.decoder(trg_emb, (self.h0_decoder, self.c0_decoder))

        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size(0) * trg_h.size(1),
            trg_h.size(2)
        )

        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size(0),
            trg_h.size(1),
            decoder_logit.size(1)
        )

        return decoder_logit

    def decode(self, logits):
        """Return probability distribution over words."""
        logits_reshape = logits.view(-1, self.trg_seg_vocab_size)
        word_probs = F.softmax(logits_reshape)  # , dim=-1)
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs


class Seq2SeqAutoencoder(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(
            self,
            src_emb_dim,
            trg_emb_dim,
            src_vocab_size,
            src_hidden_dim,
            trg_hidden_dim,
            batch_size,
            pad_token_src,
            bidirectional=False,
            nlayers=1,
            nlayers_trg=1,
            dropout=0.,
    ):
        """Initialize model."""
        super(Seq2SeqAutoencoder, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.nlayers = nlayers
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        self.pad_token_src = pad_token_src

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            src_emb_dim,
            self.pad_token_src
        )
        self.trg_embedding = nn.Embedding(
            src_vocab_size,
            trg_emb_dim,
            self.pad_token_src
        )

        if self.bidirectional and self.nlayers > 1:
            self.encoder = DeepBidirectionalLSTM(
                self.src_emb_dim,
                self.src_hidden_dim,
                self.nlayers,
                self.dropout,
                True
            )

        else:
            hidden_dim = self.src_hidden_dim // 2 \
                if self.bidirectional else self.src_hidden_dim
            self.encoder = nn.LSTM(
                src_emb_dim,
                hidden_dim,
                nlayers,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=self.dropout
            )

        self.decoder = nn.LSTM(
            trg_emb_dim,
            trg_hidden_dim,
            nlayers_trg,
            dropout=self.dropout,
            batch_first=True
        )

        self.encoder2decoder = nn.Linear(
            self.src_hidden_dim,
            trg_hidden_dim
        )
        self.decoder2vocab = nn.Linear(trg_hidden_dim, src_vocab_size).cuda()

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.encoder.batch_first else input.size(1)
        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ))
        c0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ))

        return h0_encoder.cuda(), c0_encoder.cuda()

    def forward(self, input, ctx_mask=None, trg_mask=None):
        """Propogate input through the network."""
        src_emb = self.src_embedding(input)
        trg_emb = self.trg_embedding(input)

        if self.bidirectional and self.nlayers > 1:
            src_h, (src_h_t, src_c_t) = self.encoder(src_emb)

        else:
            self.h0_encoder, self.c0_encoder = self.get_state(input)

            src_h, (src_h_t, src_c_t) = self.encoder(
                src_emb, (self.h0_encoder, self.c0_encoder)
            )

        if self.bidirectional and self.nlayers == 1:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))

        trg_h, (_, _) = self.decoder(
            trg_emb,
            (
                decoder_init_state.view(
                    self.decoder.num_layers,
                    decoder_init_state.size(0),
                    decoder_init_state.size(1)
                ),
                c_t.view(
                    self.decoder.num_layers,
                    c_t.size(0),
                    c_t.size(1)
                )
            )
        )
        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size(0) * trg_h.size(1),
            trg_h.size(2)
        )
        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size(0),
            trg_h.size(1),
            decoder_logit.size(1)
        )

        return decoder_logit

    def decode(self, logits):
        """Return probability distribution over words."""
        logits_reshape = logits.view(-1, self.src_vocab_size)
        word_probs = F.softmax(logits_reshape)
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs


class Seq2SeqAttention(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(
            self,
            src_loc_emb_dim,
            src_tim_emb_dim,
            trg_seg_emb_dim,
            src_loc_vocab_size,
            src_tim_vocab_size,
            trg_seg_vocab_size,
            src_hidden_dim,
            trg_hidden_dim,
            batch_size,
            pad_token_src_loc,
            pad_token_src_tim1,
            pad_token_src_tim2,
            pad_token_trg,
            bidirectional=True,
            nlayers_src=2,
            dropout=0.,
            time_encoding='NoEncoding',
            rnn_type='LSTM',
            attn_type='Dot'
    ):
        """Initialize model."""
        super(Seq2SeqAttention, self).__init__()
        self.src_loc_vocab_size = src_loc_vocab_size
        self.src_tim_vocab_size = src_tim_vocab_size
        self.trg_seg_vocab_size = trg_seg_vocab_size
        self.src_loc_emb_dim = src_loc_emb_dim
        self.src_tim_emb_dim = src_tim_emb_dim
        self.trg_seg_emb_dim = trg_seg_emb_dim
        self.time_encoding = time_encoding
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.nlayers_src = nlayers_src
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        self.pad_token_src_loc = pad_token_src_loc
        self.pad_token_src_tim1 = pad_token_src_tim1
        self.pad_token_src_tim2 = pad_token_src_tim2
        self.pad_token_trg = pad_token_trg
        self.src_hidden_dim = src_hidden_dim // 2 if self.bidirectional else src_hidden_dim

        self.rnn_type = rnn_type
        self.attn_type = attn_type

        if self.time_encoding == 'NoEncoding':
            src_emb_dim = self.src_loc_emb_dim
        elif self.time_encoding == 'OneEncoding':
            src_emb_dim = self.src_loc_emb_dim + self.src_tim_emb_dim[0]
        elif self.time_encoding == 'TwoEncoding':
            src_emb_dim = self.src_loc_emb_dim + self.src_tim_emb_dim[1][0] + self.src_tim_emb_dim[1][1]
        else:
            raise RuntimeError('error time encoding')

        self.src_embedding = nn.Embedding(
            self.src_loc_vocab_size,
            self.src_loc_emb_dim,
            self.pad_token_src_loc
        )
        self.trg_embedding = nn.Embedding(
            self.trg_seg_vocab_size,
            self.trg_seg_emb_dim,
            self.pad_token_trg
        )

        if self.time_encoding == 'OneEncoding':
            self.src_time_embedding = nn.Embedding(
                self.src_tim_vocab_size[0],
                self.src_tim_emb_dim[0],
                self.pad_token_src_tim1
            )
        elif self.time_encoding == 'TwoEncoding':
            self.src_time_embedding_1 = nn.Embedding(
                self.src_tim_vocab_size[1][0],
                self.src_tim_emb_dim[1][0],
                self.pad_token_src_tim2[0]
            )
            self.src_time_embedding_2 = nn.Embedding(
                self.src_tim_vocab_size[1][1],
                self.src_tim_emb_dim[1][1],
                self.pad_token_src_tim2[1]
            )

        self.encoder = getattr(nn, self.rnn_type)(
            src_emb_dim,
            self.src_hidden_dim,
            self.nlayers_src,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=self.dropout
        )

        self.decoder = LSTMAttentionDot(
            self.trg_seg_emb_dim,
            self.trg_hidden_dim,
            batch_first=True,
            attn_type=self.attn_type
        )

        self.encoder2decoder = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            self.trg_hidden_dim
        )
        self.decoder2vocab = nn.Linear(self.trg_hidden_dim, trg_seg_vocab_size).cuda()

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.encoder.batch_first else input.size(1)
        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)
        c0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)

        return h0_encoder.cuda(), c0_encoder.cuda()

    def get_decoder_state(self, input):
        """Get decoder cell states and hidden states."""
        batch_size = input.size(0) if self.decoder.batch_first else input.size(1)
        h0_decoder = Variable(torch.zeros(
            self.decoder.num_layers - 1,
            batch_size,
            self.trg_hidden_dim
        ))
        c0_decoder = Variable(torch.zeros(
            self.decoder.num_layers - 1,
            batch_size,
            self.trg_hidden_dim
        ))

        return h0_decoder.cuda(), c0_decoder.cuda()

    def forward(self, input_src, input_trg, input_time=None, ctx_mask=None, trg_mask=None):
        """Propogate input through the network."""
        src_emb = self.src_embedding(input_src)
        trg_emb = self.trg_embedding(input_trg)
        if self.time_encoding == 'NoEncoding':
            src_time_emb = src_emb
        elif self.time_encoding == 'OneEncoding':
            time_emb = self.src_time_embedding(input_time)
            src_time_emb = torch.cat((src_emb, time_emb), dim=2)
        elif self.time_encoding == 'TwoEncoding':
            time_emb_1 = self.src_time_embedding_1(input_time[0])
            time_emb_2 = self.src_time_embedding_2(input_time[1])
            src_time_emb = torch.cat((src_emb, time_emb_1, time_emb_2), dim=2)
        else:
            raise RuntimeError('error time encoding')

        self.h0_encoder, self.c0_encoder = self.get_state(input_src)

        if self.rnn_type == "LSTM":
            src_h, (src_h_t, src_c_t) = self.encoder(src_time_emb, (self.h0_encoder, self.c0_encoder))
        else:
            src_h, src_h_t = self.encoder(src_time_emb, self.h0_encoder)
            src_c_t = self.c0_encoder

        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))

        # attention
        ctx = src_h.transpose(0, 1)
        trg_h, (_, _) = self.decoder(trg_emb, (decoder_init_state, c_t), ctx, ctx_mask)

        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size()[0] * trg_h.size()[1],
            trg_h.size()[2]
        )
        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size()[0],
            trg_h.size()[1],
            decoder_logit.size()[1]
        )
        return decoder_logit

    def decode(self, logits):
        """Return probability distribution over words."""
        logits_reshape = logits.view(-1, self.trg_seg_vocab_size)
        word_probs = F.softmax(logits_reshape)
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs


class Seq2SeqAttentionSharedEmbedding(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(
            self,
            emb_dim,
            vocab_size,
            src_hidden_dim,
            trg_hidden_dim,
            ctx_hidden_dim,
            attention_mode,
            batch_size,
            pad_token_src,
            pad_token_trg,
            bidirectional=True,
            nlayers=2,
            nlayers_trg=2,
            dropout=0.,
    ):
        """Initialize model."""
        super(Seq2SeqAttentionSharedEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.ctx_hidden_dim = ctx_hidden_dim
        self.attention_mode = attention_mode
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.nlayers = nlayers
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        self.pad_token_src = pad_token_src
        self.pad_token_trg = pad_token_trg

        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            self.pad_token_src
        )

        self.src_hidden_dim = src_hidden_dim // 2 \
            if self.bidirectional else src_hidden_dim

        self.encoder = nn.LSTM(
            emb_dim,
            self.src_hidden_dim,
            nlayers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=self.dropout
        )

        self.decoder = LSTMAttentionDot(
            emb_dim,
            trg_hidden_dim,
            batch_first=True
        )

        self.encoder2decoder = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            trg_hidden_dim
        )
        self.decoder2vocab = nn.Linear(trg_hidden_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.encoder.batch_first else input.size(1)

        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)

        c0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)

        return h0_encoder.cuda(), c0_encoder.cuda()

    def forward(self, input_src, input_trg, trg_mask=None, ctx_mask=None):
        """Propogate input through the network."""
        src_emb = self.embedding(input_src)
        trg_emb = self.embedding(input_trg)

        self.h0_encoder, self.c0_encoder = self.get_state(input_src)

        src_h, (src_h_t, src_c_t) = self.encoder(
            src_emb, (self.h0_encoder, self.c0_encoder)
        )

        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))

        ctx = src_h.transpose(0, 1)

        trg_h, (_, _) = self.decoder(
            trg_emb,
            (decoder_init_state, c_t),
            ctx,
            ctx_mask
        )

        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size()[0] * trg_h.size()[1],
            trg_h.size()[2]
        )

        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size()[0],
            trg_h.size()[1],
            decoder_logit.size()[1]
        )
        return decoder_logit

    def decode(self, logits):
        """Return probability distribution over words."""
        logits_reshape = logits.view(-1, self.vocab_size)
        word_probs = F.softmax(logits_reshape)
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs


class Seq2SeqFastAttention(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(
            self,
            src_emb_dim,
            trg_emb_dim,
            src_vocab_size,
            trg_vocab_size,
            src_hidden_dim,
            trg_hidden_dim,
            batch_size,
            pad_token_src,
            pad_token_trg,
            bidirectional=True,
            nlayers=2,
            nlayers_trg=2,
            dropout=0.,
    ):
        """Initialize model."""
        super(Seq2SeqFastAttention, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.nlayers = nlayers
        self.nlayers_trg = nlayers_trg
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        self.pad_token_src = pad_token_src
        self.pad_token_trg = pad_token_trg

        assert trg_hidden_dim == src_hidden_dim
        self.src_embedding = nn.Embedding(
            src_vocab_size,
            src_emb_dim,
            self.pad_token_src
        )
        self.trg_embedding = nn.Embedding(
            trg_vocab_size,
            trg_emb_dim,
            self.pad_token_trg
        )

        self.src_hidden_dim = src_hidden_dim // 2 \
            if self.bidirectional else src_hidden_dim
        self.encoder = nn.LSTM(
            src_emb_dim,
            self.src_hidden_dim,
            nlayers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=self.dropout
        )

        self.decoder = nn.LSTM(
            trg_emb_dim,
            trg_hidden_dim,
            self.nlayers_trg,
            batch_first=True,
            dropout=self.dropout
        )

        self.encoder2decoder = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            trg_hidden_dim
        )
        self.decoder2vocab = nn.Linear(2 * trg_hidden_dim, trg_vocab_size)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.encoder.batch_first else input.size(1)
        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)
        c0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)

        return h0_encoder.cuda(), c0_encoder.cuda()

    def get_decoder_state(self, input):
        """Get decoder cell states and hidden states."""
        batch_size = input.size(0) if self.decoder.batch_first else input.size(1)
        h0_decoder = Variable(torch.zeros(
            self.decoder.num_layers - 1,
            batch_size,
            self.trg_hidden_dim
        ))
        c0_decoder = Variable(torch.zeros(
            self.decoder.num_layers - 1,
            batch_size,
            self.trg_hidden_dim
        ))

        return h0_decoder.cuda(), c0_decoder.cuda()

    def forward(self, input_src, input_trg, trg_mask=None, ctx_mask=None):
        """Propogate input through the network."""
        src_emb = self.src_embedding(input_src)
        trg_emb = self.trg_embedding(input_trg)

        self.h0_encoder, self.c0_encoder = self.get_state(input_src)

        src_h, (src_h_t, src_c_t) = self.encoder(
            src_emb, (self.h0_encoder, self.c0_encoder)
        )  # bsize x seqlen x dim

        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]
        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))

        if self.nlayers_trg > 1:
            self.h0_decoder, self.c0_decoder = self.get_decoder_state(input_src)
            self.h0_decoder = torch.cat(
                (decoder_init_state.view(1, decoder_init_state.size(0), decoder_init_state.size(1)), self.h0_decoder),
                0)
            self.c0_decoder = torch.cat((c_t.view(1, c_t.size(0), c_t.size(1)), self.c0_decoder), 0)
        else:
            self.h0_decoder = decoder_init_state.view(1, decoder_init_state.size(0), decoder_init_state.size(1))
            self.c0_decoder = c_t.view(1, c_t.size(0), c_t.size(1))

        trg_h, (_, _) = self.decoder(trg_emb, (self.h0_decoder, self.c0_decoder))

        # trg_h, (_, _) = self.decoder(
        #     trg_emb,
        #     (
        #         decoder_init_state.view(
        #             self.decoder.num_layers,
        #             decoder_init_state.size(0),
        #             decoder_init_state.size(1)
        #         ),
        #         c_t.view(
        #             self.decoder.num_layers,
        #             c_t.size(0),
        #             c_t.size(1)
        #         )
        #     )
        # )  # bsize x seqlen x dim

        # Fast Attention dot product

        # bsize x seqlen_src x seq_len_trg
        alpha = torch.bmm(src_h, trg_h.transpose(1, 2))
        # bsize x seq_len_trg x dim
        alpha = torch.bmm(alpha.transpose(1, 2), src_h)
        # bsize x seq_len_trg x (2 * dim)
        trg_h_reshape = torch.cat((trg_h, alpha), 2)

        trg_h_reshape = trg_h_reshape.view(
            trg_h_reshape.size(0) * trg_h_reshape.size(1),
            trg_h_reshape.size(2)
        )
        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size()[0],
            trg_h.size()[1],
            decoder_logit.size()[1]
        )
        return decoder_logit

    def decode(self, logits):
        """Return probability distribution over words."""
        logits_reshape = logits.view(-1, self.trg_vocab_size)
        word_probs = F.softmax(logits_reshape)
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs
