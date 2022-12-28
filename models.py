import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils import *
import math
from torch.distributions import MultivariateNormal, Categorical
from sklearn.mixture import GaussianMixture as GMM


def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(*shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class CouplingLayer(nn.Module):
    """Coupling layer in RealNVP.
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the `s` and `t` network.
        num_blocks (int): Number of residual blocks in the `s` and `t` network.
        mask_type (MaskType): One of `MaskType.CHECKERBOARD` or `MaskType.CHANNEL_WISE`.
        reverse_mask (bool): Whether to reverse the mask. Useful for alternating masks.
    """

    def __init__(self, latent_dim, reverse_mask, hidden_dims=None):
        super(CouplingLayer, self).__init__()
        # Save mask info
        self.reverse_mask = reverse_mask

        # Build scale and translate network
        if hidden_dims is None:
            hidden_dims = [8]

        modules = []
        in_channels = latent_dim // 2
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(nn.Linear(hidden_dims[-1], latent_dim))
        self.st_net = nn.Sequential(*modules)

        # Learnable scale for s
        self.rescale = nn.utils.weight_norm(Rescale(latent_dim // 2))

    def forward(self, x, sldj=None, reverse=False):
        # Channel-wise mask
        if self.reverse_mask:
            x_id, x_change = x.chunk(2, dim=2)
        else:
            x_change, x_id = x.chunk(2, dim=2)

        st = self.st_net(x_id)
        s, t = st.chunk(2, dim=2)
        s = self.rescale(torch.tanh(s))

        # Scale and translate
        if reverse:
            inv_exp_s = s.mul(-1).exp()
            if torch.isnan(inv_exp_s).any():
                raise RuntimeError('Scale factor has NaN entries')
            x_change = x_change * inv_exp_s - t
        else:
            exp_s = s.exp()
            if torch.isnan(exp_s).any():
                raise RuntimeError('Scale factor has NaN entries')
            x_change = (x_change + t) * exp_s

            # Add log-determinant of the Jacobian
            sldj += s.view(s.size(0), s.size(1), -1).sum(-1)

        if self.reverse_mask:
            x = torch.cat((x_id, x_change), dim=2)
        else:
            x = torch.cat((x_change, x_id), dim=2)

        return x, sldj


class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """

    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))

    def forward(self, x):
        x = self.weight * x
        return x


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):
        bs, n = h.size()[:2]
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn

    def __repr__(self):
        return (
                self.__class__.__name__
                + " ("
                + str(self.n_head)
                + " -> "
                + str(self.f_in)
                + " -> "
                + str(self.f_out)
                + ")"
        )


class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
                )
            )

        self.norm_list = [
            torch.nn.InstanceNorm1d(32).cuda(),
            torch.nn.InstanceNorm1d(64).cuda(),
        ]

    def forward(self, x):
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x, attn = gat_layer(x)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x


class GATEncoder(nn.Module):
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(GATEncoder, self).__init__()
        self.gat_net = GAT(n_units, n_heads, dropout, alpha)

    def forward(self, obs_traj_embedding, seq_start_end):
        graph_embeded_data = []
        for start, end in seq_start_end.data:
            curr_seq_embedding_traj = obs_traj_embedding[:, start:end, :]
            curr_seq_graph_embedding = self.gat_net(curr_seq_embedding_traj)
            graph_embeded_data.append(curr_seq_graph_embedding)
        graph_embeded_data = torch.cat(graph_embeded_data, dim=1)
        return graph_embeded_data


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))

    def forward(self, input, state):
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih + torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.hardsigmoid(ingate)
        forgetgate = F.hardsigmoid(forgetgate)
        cellgate = F.hardtanh(cellgate)
        outgate = F.hardsigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.hardtanh(cy)

        return hy, cy


class Encoder(nn.Module):
    def __init__(
            self,
            obs_len,
            fut_len,
            n_coordinates,
            traj_lstm_hidden_size,
            n_units,
            n_heads,
            graph_network_out_dims,
            dropout,
            alpha,
            graph_lstm_hidden_size,
            add_confidence=True,
    ):
        super(Encoder, self).__init__()

        self.obs_len = obs_len
        self.fut_len = fut_len

        self.gatencoder = GATEncoder(
            n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha
        )

        self.graph_lstm_hidden_size = graph_lstm_hidden_size
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.n_coordinates = n_coordinates
        self.add_confidence = add_confidence

        self.traj_lstm_model = LSTMCell(
            n_coordinates + add_confidence,
            traj_lstm_hidden_size
        )
        self.graph_lstm_model = LSTMCell(
            graph_network_out_dims,
            graph_lstm_hidden_size
        )

        self.traj_hidden2pos = nn.Linear(
            traj_lstm_hidden_size,
            n_coordinates
        )
        self.traj_gat_hidden2pos = nn.Linear(
            traj_lstm_hidden_size + graph_lstm_hidden_size,
            n_coordinates
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTMCell):
                m.weight_hh.data.normal_(0, 0.1)
                m.weight_ih.data.normal_(0, 0.1)
                m.bias_hh.data.zero_()
                m.bias_ih.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.zero_()

    def init_hidden_traj_lstm(self, batch):
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )

    def init_hidden_graph_lstm(self, batch):
        return (
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
        )

    def forward(
            self,
            obs_traj_rel,
            seq_start_end,
            training_step=3,
    ):

        num_peds = obs_traj_rel.shape[1]
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(num_peds)
        graph_lstm_h_t, graph_lstm_c_t = self.init_hidden_graph_lstm(num_peds)
        pred_traj_rel = []
        traj_lstm_hidden_states = []
        graph_lstm_hidden_states = []

        # traj_lstm (used in step 1,2,3)
        for i in range(self.obs_len):
            traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model(
                obs_traj_rel[i], (traj_lstm_h_t, traj_lstm_c_t)
            )
            if training_step == "P1":
                output = self.traj_hidden2pos(traj_lstm_h_t)
                pred_traj_rel += [output]
            else:
                traj_lstm_hidden_states += [traj_lstm_h_t]

        # graph_lstm (used in step 2,3)
        if training_step != "P1":
            graph_lstm_input = self.gatencoder(
                torch.stack(traj_lstm_hidden_states), seq_start_end
            )
            for i in range(self.obs_len):
                graph_lstm_h_t, graph_lstm_c_t = self.graph_lstm_model(
                    graph_lstm_input[i], (graph_lstm_h_t, graph_lstm_c_t)
                )
                if training_step == "P2":
                    encoded_before_noise_hidden = torch.cat(
                        (traj_lstm_hidden_states[i], graph_lstm_h_t), dim=1
                    )
                    output = self.traj_gat_hidden2pos(encoded_before_noise_hidden)
                    pred_traj_rel += [output]
                else:
                    graph_lstm_hidden_states += [graph_lstm_h_t]

        if training_step in ["P1", "P2"]:
            return torch.stack(pred_traj_rel)

        else:
            encoded_before_noise_hidden = torch.cat((traj_lstm_hidden_states[-1], graph_lstm_hidden_states[-1]), dim=1)

            return encoded_before_noise_hidden


class Predictor(nn.Module):
    def __init__(
            self,
            obs_len,
            fut_len,
            n_coordinates,
            s_dim,
            z_dim,
            teacher_forcing_ratio=0.5,
            hidden_dims=None,
            noise_dim=(8,),
            noise_type="gaussian",
    ):
        super(Predictor, self).__init__()

        if hidden_dims is None:
            hidden_dims = [8]

        modules = []
        in_channels = s_dim + z_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(nn.Linear(hidden_dims[-1], s_dim + z_dim))

        self.mapping = nn.Sequential(*modules)

        self.obs_len = obs_len
        self.fut_len = fut_len
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.n_coordinates = n_coordinates
        self.pred_lstm_hidden_size = z_dim + s_dim + noise_dim[0]
        self.pred_hidden2pos = nn.Linear(self.pred_lstm_hidden_size, n_coordinates)
        self.pred_lstm_model = LSTMCell(n_coordinates, self.pred_lstm_hidden_size)
        self.noise_dim = noise_dim
        self.noise_type = noise_type

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTMCell):
                m.weight_hh.data.normal_(0, 0.1)
                m.weight_ih.data.normal_(0, 0.1)
                m.bias_hh.data.zero_()
                m.bias_ih.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.zero_()

    def add_noise(self, _input, seq_start_end):
        noise_shape = (seq_start_end.size(0),) + self.noise_dim
        z_decoder = get_noise(noise_shape, self.noise_type)

        _list = []
        for idx, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            _vec = z_decoder[idx].view(1, -1)
            _to_cat = _vec.repeat(_input.shape[0], end - start, 1)
            _list_cat = torch.cat([_input[:, start:end, :], _to_cat], dim=2)
            _list.append(_list_cat)
        decoder_h = torch.cat(_list, dim=1)

        return decoder_h

    def forward(
            self,
            obs_traj_rel,
            fut_traj_rel,
            seq_start_end,
            pred_lstm_hidden,
    ):

        input_t = obs_traj_rel[self.obs_len - 1, :, :self.n_coordinates].repeat(len(pred_lstm_hidden), 1, 1)
        output = input_t
        pred_lstm_hidden = self.mapping(pred_lstm_hidden)
        pred_lstm_hidden = self.add_noise(pred_lstm_hidden, seq_start_end)
        pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()
        pred_q_rel = []
        if self.training:
            for i in range(self.fut_len):
                if i >= 1:
                    teacher_force = random.random() < self.teacher_forcing_ratio
                    if teacher_force:
                        input_t = fut_traj_rel[i - 1, :, :self.n_coordinates]  # with teacher help
                    else:
                        input_t = output

                # average over s and z
                output = []
                pred_lstm_hidden_list, pred_lstm_c_t_list = [], []
                for j in range(len(pred_lstm_hidden)):
                    h, c = self.pred_lstm_model(input_t[j], (pred_lstm_hidden[j], pred_lstm_c_t[j]))
                    pred_lstm_hidden_list += [h]
                    pred_lstm_c_t_list += [c]
                    output += [self.pred_hidden2pos(h)]

                pred_lstm_hidden = torch.stack(pred_lstm_hidden_list)
                pred_lstm_c_t = torch.stack(pred_lstm_c_t_list)
                output = torch.stack(output)
                pred_q_rel += [output]

            return torch.stack(pred_q_rel)

        else:
            pred_traj_rel = []
            for i in range(self.fut_len):
                input_t = output

                # average over s and z
                output = []
                pred_lstm_hidden_list, pred_lstm_c_t_list = [], []
                for j in range(len(pred_lstm_hidden)):
                    h, c = self.pred_lstm_model(input_t[j], (pred_lstm_hidden[j], pred_lstm_c_t[j]))
                    pred_lstm_hidden_list += [h]
                    pred_lstm_c_t_list += [c]
                    output += [self.pred_hidden2pos(h)]

                pred_lstm_hidden = torch.stack(pred_lstm_hidden_list)
                pred_lstm_c_t = torch.stack(pred_lstm_c_t_list)
                output = torch.stack(output)
                pred_traj_rel += [output]

            return torch.stack(pred_traj_rel)


class Decoder(nn.Module):
    def __init__(
            self,
            obs_len,
            n_coordinates,
            z_dim,
            s_dim,
            hidden_dims=None,
            var_p=0.5
    ):
        super(Decoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [8]

        modules = []
        in_channels = s_dim + z_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(nn.Linear(hidden_dims[-1], s_dim + z_dim))
        self.mapping = nn.Sequential(*modules)

        self.obs_len = obs_len
        self.n_coordinates = n_coordinates
        self.pred_lstm_hidden_size = z_dim + s_dim
        self.pred_hidden2pos = nn.Linear(self.pred_lstm_hidden_size, n_coordinates)
        self.pred_lstm_model = LSTMCell(n_coordinates, self.pred_lstm_hidden_size)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTMCell):
                m.weight_hh.data.normal_(0, 0.1)
                m.weight_ih.data.normal_(0, 0.1)
                m.bias_hh.data.zero_()
                m.bias_ih.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.zero_()

    def forward(
            self,
            obs_traj_rel,
            pred_lstm_hidden,
    ):

        pred_traj_rel = []
        pred_lstm_hidden = self.mapping(pred_lstm_hidden)
        pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()
        for i in range(self.obs_len):
            if i >= 1:
                input_t = obs_traj_rel[i - 1, :, :self.n_coordinates].repeat(len(pred_lstm_hidden), 1, 1)
            else:
                input_t = obs_traj_rel[0, :, :self.n_coordinates].repeat(len(pred_lstm_hidden), 1, 1)

            # average over s and z
            output = []
            pred_lstm_hidden_list, pred_lstm_c_t_list = [], []
            for j in range(len(pred_lstm_hidden)):
                h, c = self.pred_lstm_model(input_t[j], (pred_lstm_hidden[j], pred_lstm_c_t[j]))
                pred_lstm_hidden_list += [h]
                pred_lstm_c_t_list += [c]
                output += [self.pred_hidden2pos(h)]

            pred_lstm_hidden = torch.stack(pred_lstm_hidden_list)
            pred_lstm_c_t = torch.stack(pred_lstm_c_t_list)
            output = torch.stack(output)
            pred_traj_rel += [output]

        return torch.stack(pred_traj_rel)


class Mapping(nn.Module):
    def __init__(self,
                 traj_lstm_hidden_size: int,
                 graph_lstm_hidden_size: int,
                 s_dim: int,
                 hidden_dims=None,
                 **kwargs) -> None:
        super(Mapping, self).__init__()

        if hidden_dims is None:
            hidden_dims = [8]

        modules = []
        in_channels = traj_lstm_hidden_size + graph_lstm_hidden_size
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.mapping = nn.Sequential(*modules)

        self.s_dim = s_dim
        self.fc_mu = nn.Linear(hidden_dims[-1], s_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], s_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTMCell):
                m.weight_hh.data.normal_(0, 0.1)
                m.weight_ih.data.normal_(0, 0.1)
                m.bias_hh.data.zero_()
                m.bias_ih.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.zero_()

    def forward(self, hidden_states, mode="normal"):
        if mode == "normal":
            s_vec = self.fc_mu(self.mapping(hidden_states))

            return s_vec
        else:
            mu = self.fc_mu(self.mapping(hidden_states))
            logvar = self.fc_logvar(self.mapping(hidden_states))
            ps = MultivariateNormal(mu, torch.diag_embed(torch.exp(logvar)))

            return ps


class SimpleStyleEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleStyleEncoder, self).__init__()

        # style encoder
        self.encoder = nn.Sequential(
            nn.Linear(40, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size * 2)
        )

    def forward(self, style_input):
        # for batch size 68
        # style 20 x 128 x 2
        style_input = torch.stack(style_input.split(2, dim=1), dim=1)[:, :, 1, :]  # 20 x 64 x 2
        style_input = torch.permute(style_input, (1, 0, 2))  # 64 x 20 x 2
        style_input = torch.flatten(style_input, 1)  # 64 x 40

        # MLP
        style_seq = self.encoder(style_input)
        encoded = torch.stack(style_seq.split(style_seq.shape[1] // 2, dim=1), dim=1)
        encoded = encoded.flatten(start_dim=0, end_dim=1)

        return encoded


class SimpleEncoder(nn.Module):
    def __init__(
            self,
            obs_len,
            hidden_size,
            number_agents,
            add_confidence,
    ):
        super(SimpleEncoder, self).__init__()

        # num of frames per sequence
        self.obs_len = obs_len
        self.number_agents = number_agents

        self.mlp = nn.Sequential(
            nn.Linear(obs_len * number_agents * (2 + add_confidence), hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size * number_agents),
        )

        self.mu = nn.Linear(hidden_size, hidden_size)
        self.logvar = nn.Linear(hidden_size, hidden_size)

    def forward(self, obs_traj_rel):
        splits = obs_traj_rel.split(self.number_agents, dim=1)
        if splits[-1].shape[1] != splits[0].shape[1]:
            splits = splits[:-1]
        obs_traj_rel = torch.stack(splits, dim=1)
        obs_traj_rel = torch.permute(obs_traj_rel, (1, 2, 0, 3))
        obs_traj_rel = obs_traj_rel.flatten(start_dim=1)

        encoded = self.mlp(obs_traj_rel)

        encoded = torch.stack(encoded.split(encoded.shape[1] // self.number_agents, dim=1), dim=1)
        encoded = encoded.flatten(start_dim=0, end_dim=1)

        return encoded


class SimpleDecoder(nn.Module):
    def __init__(
            self,
            seq_len,
            hidden_size,
            number_of_agents,
    ):
        super(SimpleDecoder, self).__init__()

        # num of frames per sequence
        self.seq_len = seq_len

        self.noise_fixed = False

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * number_of_agents, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size * 8),
            nn.ReLU(),
            nn.Linear(hidden_size * 8, hidden_size * 16),
            nn.ReLU(),
            nn.Linear(hidden_size * 16, hidden_size * 32),
            nn.ReLU(),
            nn.Linear(hidden_size * 32, 2 * seq_len * number_of_agents)
        )

        self.number_of_agents = number_of_agents

    def forward(self, latent_space):
        traj_lstm_hidden_state = torch.stack(latent_space.split(self.number_of_agents, dim=1), dim=1)
        out = traj_lstm_hidden_state.flatten(start_dim=2)

        out = self.mlp(out)

        out = torch.reshape(out, (out.shape[0], out.shape[1], self.number_of_agents, self.seq_len, 2))

        out = out.flatten(start_dim=1, end_dim=2)

        out = torch.permute(out, (2, 0, 1, 3))

        return out


class CRMF(nn.Module):
    def __init__(self, args):
        super(CRMF, self).__init__()

        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.obs_len = args.obs_len
        self.z_dim = args.z_dim
        self.fut_len = args.fut_len
        self.num_samples = args.num_samples
        self.n_coordinates = args.n_coordinates

        self.num_envs = args.num_envs
        self.pi_priore = nn.Parameter(-1 * torch.ones(args.num_envs))
        self.logvar_priors = nn.Parameter(torch.randn(args.num_envs, args.s_dim))
        self.mean_priors = nn.Parameter(torch.zeros(args.num_envs, args.s_dim))
        self.gmm = GMM(n_components=args.num_envs, covariance_type='diag')
        self.beta_scheduler = get_beta(0, 1500, 1000)
        self.iter = 1

        self.coupling_layers_z = nn.ModuleList([
            CouplingLayer(args.z_dim, reverse_mask=False),
            CouplingLayer(args.z_dim, reverse_mask=True),
            CouplingLayer(args.z_dim, reverse_mask=False)
        ])

        self.coupling_layers_s = nn.ModuleList([
            CouplingLayer(args.s_dim, reverse_mask=False),
            CouplingLayer(args.s_dim, reverse_mask=True),
            CouplingLayer(args.s_dim, reverse_mask=False)
        ])

        self.pw = MultivariateNormal(torch.zeros(args.z_dim).cuda(), torch.diag(torch.ones(args.z_dim).cuda()))

        self.ps = []
        for i in range(self.num_envs):
            self.ps += [MultivariateNormal(i * torch.ones(args.s_dim).cuda(),
                                           torch.diag((i + 1) * torch.ones(args.s_dim).cuda()))]

        self.x_to_z = Mapping(2, 0, args.z_dim)
        self.x_to_s = Mapping(2, 0, args.s_dim)

        if args.model_name == "lstm":

            self.variant_encoder = Encoder(args.obs_len, args.fut_len, args.n_coordinates,
                                           args.traj_lstm_hidden_size, args.n_units, args.n_heads,
                                           args.graph_network_out_dims, args.dropout, args.alpha,
                                           args.graph_lstm_hidden_size, args.add_confidence)

            self.invariant_encoder = Encoder(args.obs_len, args.fut_len, args.n_coordinates,
                                             args.traj_lstm_hidden_size, args.n_units, args.n_heads,
                                             args.graph_network_out_dims, args.dropout, args.alpha,
                                             args.graph_lstm_hidden_size, args.add_confidence)

            self.past_decoder = Decoder(args.obs_len, args.n_coordinates, args.z_dim, args.s_dim)

            self.future_decoder = Predictor(args.obs_len, args.fut_len, args.n_coordinates, args.s_dim,
                                            args.z_dim, args.teachingratio)

        elif args.model_name == "mlp":
            self.variant_encoder = SimpleEncoder(args.obs_len, 2,
                                                 NUMBER_PERSONS, args.add_confidence)

            self.invariant_encoder = SimpleEncoder(args.obs_len, 2,
                                                   NUMBER_PERSONS, args.add_confidence)

            self.past_decoder = Decoder(args.obs_len, args.n_coordinates, args.z_dim, args.s_dim)

            self.future_decoder = Predictor(args.obs_len, args.fut_len, args.n_coordinates, args.s_dim,
                                            args.z_dim, args.teachingratio)

            # self.past_decoder = SimpleDecoder(
            #     args.obs_len,
            #     args.s_dim,
            #     NUMBER_PERSONS,
            # )
            #
            # self.future_decoder = SimpleDecoder(
            #     args.fut_len,
            #     args.s_dim,
            #     NUMBER_PERSONS,
            # )

        else:
            raise ValueError('Unrecognized model name "%s"' % args.model_name)

    def forward(self, batch, training_step, **kwargs):
        if self.dataset_name in ('eth', 'hotel', 'univ', 'zara1', 'zara2'):
            obs_traj, fut_traj, obs_traj_rel, fut_traj_rel, seq_start_end, = batch

        elif 'synthetic' in self.dataset_name or self.dataset_name in ['synthetic', 'v2', 'v2full', 'v4']:
            obs_traj, fut_traj, obs_traj_rel, fut_traj_rel, seq_start_end, augm_data, seq_start_end_augm = batch
        else:
            raise ValueError('Unrecognized dataset name "%s"' % self.dataset_name)

        if self.training:
            if training_step in ["P1", "P2"]:
                pred_past_rel1 = self.variant_encoder(obs_traj_rel, seq_start_end, training_step)
                pred_past_rel2 = self.invariant_encoder(obs_traj_rel, seq_start_end, training_step)

                return pred_past_rel1, pred_past_rel2

            elif training_step == "P3":
                if self.model_name == "lstm":
                    z_vec = self.x_to_z(self.invariant_encoder(obs_traj_rel, seq_start_end, training_step),
                                        mode="normal")
                    s_vec = self.x_to_s(self.variant_encoder(obs_traj_rel, seq_start_end, training_step), mode="normal")
                else:
                    z_vec = self.x_to_z(self.invariant_encoder(obs_traj_rel), mode="normal")
                    s_vec = self.x_to_s(self.variant_encoder(obs_traj_rel), mode="normal")

                if self.model_name == "lstm":
                    pred_past_rel = self.past_decoder(obs_traj_rel, torch.cat((z_vec, s_vec), dim=1))
                elif self.model_name == "mlp":
                    # pred_past_rel = self.past_decoder(torch.cat((z_vec.unsqueeze(0), s_vec.unsqueeze(0)), dim=2)).squeeze(1)
                    pred_past_rel= self.past_decoder(obs_traj_rel, torch.cat((z_vec, s_vec), dim=1))

                if self.model_name == "lstm":
                    pred_fut_rel = self.future_decoder(obs_traj_rel, fut_traj_rel, seq_start_end,
                                                       torch.cat((z_vec, s_vec), dim=1))
                if self.model_name == "mlp":
                    pred_fut_rel= self.future_decoder(obs_traj_rel, fut_traj_rel, seq_start_end,
                                                       torch.cat((z_vec, s_vec), dim=1))

                return pred_past_rel, pred_fut_rel

            elif training_step == "P4":
                if self.model_name == "lstm":
                    s_vec = self.x_to_s(self.variant_encoder(obs_traj_rel, seq_start_end, training_step), mode="normal")
                else:
                    s_vec = self.x_to_s(self.variant_encoder(obs_traj_rel), mode="normal")

                return s_vec

            else:
                pe = Categorical(logits=self.pi_priore)
                if self.model_name == "lstm":
                    q_zgx = self.x_to_z(self.invariant_encoder(obs_traj_rel, seq_start_end, training_step),
                                        mode="variational")
                    q_sgx = self.x_to_s(self.variant_encoder(obs_traj_rel, seq_start_end, training_step),
                                        mode="variational")
                elif self.model_name == "mlp":
                    q_zgx = self.x_to_z(self.invariant_encoder(obs_traj_rel), mode="variational")
                    q_sgx = self.x_to_s(self.variant_encoder(obs_traj_rel), mode="variational")

                s_vec = q_sgx.rsample([self.num_samples, ])
                z_vec = q_zgx.rsample([self.num_samples, ])

                sldj = torch.zeros((self.num_samples, z_vec.shape[1]), device=z_vec.device)
                s_vec_c = s_vec
                for coupling in self.coupling_layers_s:
                    s_vec_c, sldj = coupling(s_vec_c, sldj)

                # calculate p(s|e)
                Et = []
                for j in range(self.num_envs):
                    psge = self.ps[j]
                    log_psge = psge.log_prob(s_vec_c)
                    log_pe = pe.log_prob(torch.tensor(j).cuda())
                    Et.append(torch.exp(log_psge) * torch.exp(log_pe))

                log_ps = torch.log(torch.stack(Et).sum(0)) + sldj

                # calculate log(q(z|x))
                log_qzgx = q_zgx.log_prob(z_vec)

                # calculate log(q(s|x))
                log_qsgx = q_sgx.log_prob(s_vec)

                # calculate log(p(z))
                sldj = torch.zeros((self.num_samples, z_vec.shape[1]), device=z_vec.device)
                z_vec_c = z_vec
                for coupling in self.coupling_layers_z:
                    z_vec_c, sldj = coupling(z_vec_c, sldj)
                log_pz = self.pw.log_prob(z_vec_c) + sldj

                # calculate log(p(x|z,s))
                if self.model_name == "lstm":
                    px = self.past_decoder(obs_traj_rel, torch.cat((z_vec, s_vec), dim=2))
                elif self.model_name == "mlp":
                    px = self.past_decoder(obs_traj_rel, torch.cat((z_vec, s_vec), dim=2))

                # log_px = - l2_loss(px, obs_traj_rel, mode="raw") - 0.5 * 2 * self.obs_len * torch.log(
                #     torch.tensor(2 * math.pi)) - 0.5 * self.obs_len * torch.log(torch.tensor(0.25))
                log_px = - l2_loss(px, obs_traj_rel, mode="raw")

                # calculate q(y|x)
                if self.model_name == "lstm":
                    py = self.future_decoder(obs_traj_rel, fut_traj_rel, seq_start_end, torch.cat((z_vec, s_vec), dim=2))
                if self.model_name == "mlp":
                    py = self.future_decoder(obs_traj_rel, fut_traj_rel, seq_start_end, torch.cat((z_vec, s_vec), dim=2))

                # log_py = - l2_loss(py, fut_traj_rel, mode="raw") - 0.5 * 2 * self.fut_len * torch.log(
                #     torch.tensor(2 * math.pi)) - 0.5 * self.fut_len * torch.log(torch.tensor(0.25))
                log_py = - l2_loss(py, fut_traj_rel, mode="raw")

                E1 = torch.multiply(torch.exp(log_py), log_px).mean(0)
                E2 = torch.multiply(torch.exp(log_py), log_ps - log_qsgx).mean(0)
                E3 = torch.multiply(torch.exp(log_py), log_pz - log_qzgx).mean(0)

                return log_py, E1, E2, E3

        else:
            if training_step == "P7":
                if self.model_name == "lstm":
                    q_zgx = self.x_to_z(self.invariant_encoder(obs_traj_rel, seq_start_end, training_step),
                                        mode="variational")
                    q_sgx = self.x_to_s(self.variant_encoder(obs_traj_rel, seq_start_end, training_step),
                                        mode="variational")
                elif self.model_name == "mlp":
                    q_zgx = self.x_to_z(self.invariant_encoder(obs_traj_rel), mode="variational")
                    q_sgx = self.x_to_s(self.variant_encoder(obs_traj_rel), mode="variational")

                return q_zgx, q_sgx

            else:
                if self.model_name == "lstm":
                    q_zgx = self.x_to_z(self.invariant_encoder(obs_traj_rel, seq_start_end, training_step),
                                        mode="variational")
                    q_sgx = self.x_to_s(self.variant_encoder(obs_traj_rel, seq_start_end, training_step),
                                        mode="variational")
                elif self.model_name == "mlp":
                    q_zgx = self.x_to_z(self.invariant_encoder(obs_traj_rel), mode="variational")
                    q_sgx = self.x_to_s(self.variant_encoder(obs_traj_rel), mode="variational")

                # calculate q(y|theta, x)
                z_vec = q_zgx.rsample([self.num_samples, ])
                s_vec = q_sgx.rsample([self.num_samples, ])
                if self.model_name == "lstm":
                    q = self.future_decoder(obs_traj_rel, fut_traj_rel, seq_start_end, torch.cat((z_vec, s_vec), dim=2))
                elif self.model_name == "mlp":
                    q = self.future_decoder(obs_traj_rel, fut_traj_rel, seq_start_end, torch.cat((z_vec, s_vec), dim=2))

                return q[:, 0, :, :]
