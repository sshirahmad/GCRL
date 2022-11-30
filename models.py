import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils import *
import math
from torch.distributions import MultivariateNormal, Gamma, Poisson


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
            hidden_dims = [16, 32]

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
            x_id, x_change = x.chunk(2, dim=1)
        else:
            x_change, x_id = x.chunk(2, dim=1)

        st = self.st_net(x_id)
        s, t = st.chunk(2, dim=1)
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
            sldj += s.view(s.size(0), -1).sum(-1)

        if self.reverse_mask:
            x = torch.cat((x_id, x_change), dim=1)
        else:
            x = torch.cat((x_change, x_id), dim=1)

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


class STGAT_encoder_inv(nn.Module):
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
            z_dim,
            hidden_dims=None,
            add_confidence=True,
    ):
        super(STGAT_encoder_inv, self).__init__()

        self.obs_len = obs_len
        self.fut_len = fut_len

        self.gatencoder = GATEncoder(
            n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha
        )

        if hidden_dims is None:
            hidden_dims = [64, 128]

        modules = []
        in_channels = traj_lstm_hidden_size + graph_lstm_hidden_size
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.z_dim = z_dim
        self.mapping = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], z_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], z_dim)

        self.graph_lstm_hidden_size = graph_lstm_hidden_size
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.n_coordinates = n_coordinates
        self.add_confidence = add_confidence

        self.traj_lstm_model = nn.LSTMCell(
            n_coordinates + add_confidence,
            traj_lstm_hidden_size
        )
        self.graph_lstm_model = nn.LSTMCell(
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
            batch,
            training_step=3,
    ):
        (_, _, obs_traj_rel, _, seq_start_end) = batch
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
            mu = self.fc_mu(self.mapping(encoded_before_noise_hidden))
            logvar = self.fc_var(self.mapping(encoded_before_noise_hidden))
            z = MultivariateNormal(mu, torch.diag_embed(torch.exp(logvar)))

            return z


class STGAT_encoder_var(nn.Module):
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
        super(STGAT_encoder_var, self).__init__()

        self.obs_len = obs_len
        self.fut_len = fut_len

        self.gatencoder = GATEncoder(
            n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha
        )

        self.graph_lstm_hidden_size = graph_lstm_hidden_size
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.n_coordinates = n_coordinates
        self.add_confidence = add_confidence

        self.traj_lstm_model = nn.LSTMCell(
            n_coordinates + add_confidence,
            traj_lstm_hidden_size
        )
        self.graph_lstm_model = nn.LSTMCell(
            graph_network_out_dims,
            graph_lstm_hidden_size
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
            batch,
            training_step=3,
    ):
        (_, _, obs_traj_rel, _, seq_start_end) = batch
        num_peds = obs_traj_rel.shape[1]
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(num_peds)
        graph_lstm_h_t, graph_lstm_c_t = self.init_hidden_graph_lstm(num_peds)
        traj_lstm_hidden_states = []
        graph_lstm_hidden_states = []
        # traj_lstm (used in step 1,2,3)
        for i in range(self.obs_len):
            traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model(
                obs_traj_rel[i], (traj_lstm_h_t, traj_lstm_c_t)
            )

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

                graph_lstm_hidden_states += [graph_lstm_h_t]

        if training_step == "P1":
            return traj_lstm_hidden_states

        elif training_step == "P2":
            hidden_states = torch.stack([torch.cat((traj_lstm_hidden_states[i], graph_lstm_hidden_states[i]), dim=1) for i in range(self.obs_len)])

            return hidden_states

        else:
            return torch.cat((graph_lstm_hidden_states[-1], traj_lstm_hidden_states[-1]), dim=1)


class future_STGAT_decoder(nn.Module):
    def __init__(
            self,
            obs_len,
            fut_len,
            n_coordinates,
            s_dim,
            z_dim,
            teacher_forcing_ratio=0.5,
            noise_dim=(8,),
            noise_type="gaussian",
    ):
        super(future_STGAT_decoder, self).__init__()

        self.obs_len = obs_len
        self.fut_len = fut_len
        self.logvar = nn.Parameter(torch.randn(n_coordinates)) #TODO change this
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.n_coordinates = n_coordinates
        self.pred_lstm_hidden_size = z_dim + s_dim
        self.pred_hidden2pos = nn.Linear(self.pred_lstm_hidden_size, n_coordinates)
        self.pred_lstm_model = nn.LSTMCell(n_coordinates, self.pred_lstm_hidden_size)
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
            _to_cat = _vec.repeat(end - start, 1)
            _list_cat = torch.stack([torch.cat([_input[i, start:end], _to_cat], dim=1) for i in range(len(_input))])
            _list.append(_list_cat)
            print(_list)
        decoder_h = torch.cat(_list, dim=1)
        return decoder_h

    def forward(
            self,
            batch,
            pred_lstm_hidden,
    ):

        (_, _, obs_traj_rel, fut_traj_rel, seq_start_end) = batch

        input_t = obs_traj_rel[self.obs_len - 1, :, :self.n_coordinates]
        output = input_t
        # pred_lstm_hidden = self.add_noise(pred_lstm_hidden, seq_start_end)
        pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()
        p = []
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
                h, c = self.pred_lstm_model(input_t, (pred_lstm_hidden[j], pred_lstm_c_t[j]))
                pred_lstm_hidden_list += [h]
                pred_lstm_c_t_list += [c]
                output += [self.pred_hidden2pos(h)]

            pred_lstm_hidden = torch.stack(pred_lstm_hidden_list)
            pred_lstm_c_t = torch.stack(pred_lstm_c_t_list)
            output = torch.mean(torch.stack(output), dim=0)
            dist = MultivariateNormal(output, torch.diag(torch.exp(self.logvar)))
            output = dist.rsample()
            p += [dist]

        return p


class past_decoder(nn.Module):
    def __init__(
            self,
            obs_len,
            n_coordinates,
            z_dim,
            s_dim,
            var_p=0.5
    ):
        super(past_decoder, self).__init__()

        self.obs_len = obs_len

        self.var_p = var_p
        self.n_coordinates = n_coordinates
        self.pred_lstm_hidden_size = z_dim + s_dim
        self.pred_hidden2pos = nn.Linear(self.pred_lstm_hidden_size, n_coordinates)
        self.pred_lstm_model = nn.LSTMCell(n_coordinates, self.pred_lstm_hidden_size)

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
            batch,
            pred_lstm_hidden,
    ):

        (_, _, obs_traj_rel, _, seq_start_end) = batch

        pred_traj_rel = []
        pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()
        for i in range(self.obs_len):
            if i >= 1:
                input_t = obs_traj_rel[i - 1, :, :self.n_coordinates]
            else:
                input_t = obs_traj_rel[0, :, :self.n_coordinates]

            pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(input_t, (pred_lstm_hidden, pred_lstm_c_t))
            output = self.pred_hidden2pos(pred_lstm_hidden)
            pred_traj_rel += [output]

        return torch.stack(pred_traj_rel)


class regressor(nn.Module):
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
            hidden_dims,
            latent_dim,
            add_confidence=True,
    ):
        super(regressor, self).__init__()

        self.obs_len = obs_len
        self.fut_len = fut_len

        self.gatencoder = GATEncoder(
            n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha
        )

        self.graph_lstm_hidden_size = graph_lstm_hidden_size
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.n_coordinates = n_coordinates
        self.add_confidence = add_confidence

        self.traj_lstm_model = nn.LSTMCell(
            n_coordinates + add_confidence,
            traj_lstm_hidden_size
        )
        self.graph_lstm_model = nn.LSTMCell(
            graph_network_out_dims,
            graph_lstm_hidden_size
        )

        if hidden_dims is None:
            hidden_dims = [32, 64]

        modules = []
        in_channels = n_coordinates * obs_len
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(nn.Linear(hidden_dims[-1], latent_dim))
        self.mapping = nn.Sequential(*modules)

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
            input,
    ):
        # (_, _, obs_traj_rel, _, seq_start_end) = batch
        # num_peds = obs_traj_rel.shape[1]
        # traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(num_peds)
        # graph_lstm_h_t, graph_lstm_c_t = self.init_hidden_graph_lstm(num_peds)
        # traj_lstm_hidden_states = []
        # graph_lstm_hidden_states = []
        # # traj_lstm (used in step 1,2,3)
        # for i in range(self.obs_len):
        #     traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model(
        #         obs_traj_rel[i], (traj_lstm_h_t, traj_lstm_c_t)
        #     )
        #
        #     traj_lstm_hidden_states += [traj_lstm_h_t]
        #
        # # graph_lstm (used in step 2,3)
        #
        # graph_lstm_input = self.gatencoder(
        #     torch.stack(traj_lstm_hidden_states), seq_start_end
        # )
        # for i in range(self.obs_len):
        #     graph_lstm_h_t, graph_lstm_c_t = self.graph_lstm_model(
        #         graph_lstm_input[i], (graph_lstm_h_t, graph_lstm_c_t)
        #     )
        #
        #     graph_lstm_hidden_states += [graph_lstm_h_t]

        result = torch.flatten(input.permute(1, 0, 2), start_dim=1)
        pred_theta = self.mapping(result)

        return pred_theta


class simple_mapping(nn.Module):
    def __init__(self,
                 traj_lstm_hidden_size: int,
                 graph_lstm_hidden_size: int,
                 latent_dim,
                 s_dim: int,
                 obs_len,
                 hidden_dims=None,
                 **kwargs) -> None:
        super(simple_mapping, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 32]

        modules = []
        in_channels = latent_dim + traj_lstm_hidden_size
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.mapping1 = nn.Sequential(*modules)

        modules = []
        in_channels = latent_dim + traj_lstm_hidden_size + graph_lstm_hidden_size
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.mapping2 = nn.Sequential(*modules)

        self.obs_len = obs_len
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

    def forward(self, hidden_states, training_step, theta=None):
        if training_step == "P1":
            if theta is not None:
                if len(theta.size()) == 1:
                    theta_rep = theta.repeat(hidden_states.shape[0], 1)
                else:
                    theta_rep = theta
                mu = self.fc_mu(self.mapping1(torch.cat((hidden_states, theta_rep), dim=1)))
                logvar = self.fc_logvar(self.mapping1(torch.cat((hidden_states, theta_rep), dim=1)))
            else:
                mu = self.fc_mu(self.mapping1(hidden_states))
                logvar = self.fc_logvar(self.mapping1(hidden_states))

        else:
            if theta is not None:
                if len(theta.size()) == 1:
                    theta_rep = theta.repeat(hidden_states.shape[0], 1)
                else:
                    theta_rep = theta
                mu = self.fc_mu(self.mapping2(torch.cat((hidden_states, theta_rep), dim=1)))
                logvar = self.fc_logvar(self.mapping2(torch.cat((hidden_states, theta_rep), dim=1)))
            else:
                mu = self.fc_mu(self.mapping2(hidden_states))
                logvar = self.fc_logvar(self.mapping2(hidden_states))

        ps = MultivariateNormal(mu, torch.diag_embed(torch.exp(logvar)))

        return ps


class CRMF(nn.Module):
    def __init__(self, args):
        super(CRMF, self).__init__()

        self.obs_len = args.obs_len
        self.z_dim = args.z_dim
        self.latent_dim = args.latent_dim
        self.fut_len = args.fut_len
        self.num_samples = args.num_samples
        self.n_coordinates = args.n_coordinates

        self.theta = nn.Parameter(torch.randn(args.num_envs, args.latent_dim))
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
        self.coupling_layers_theta = nn.ModuleList([
            CouplingLayer(args.latent_dim, reverse_mask=False),
            CouplingLayer(args.latent_dim, reverse_mask=True),
            CouplingLayer(args.latent_dim, reverse_mask=False)
        ])
        self.pw = MultivariateNormal(torch.zeros(args.z_dim).cuda(), torch.diag(torch.ones(args.z_dim).cuda()))

        self.covee = torch.diag(torch.ones(args.latent_dim).cuda())
        self.covww = torch.diag(torch.ones(args.latent_dim).cuda())
        self.covwe = 0.5 * torch.diag(torch.ones(args.latent_dim).cuda())

        # self.invariant_encoder = STGAT_encoder_inv(args.obs_len, args.fut_len, args.n_coordinates,
        #                                          args.traj_lstm_hidden_size, args.n_units, args.n_heads,
        #                                          args.graph_network_out_dims, args.dropout, args.alpha,
        #                                          args.graph_lstm_hidden_size, args.z_dim, None, args.add_confidence)

        self.variant_encoder = STGAT_encoder_var(args.obs_len, args.fut_len, args.n_coordinates,
                                                 args.traj_lstm_hidden_size, args.n_units, args.n_heads,
                                                 args.graph_network_out_dims, args.dropout, args.alpha,
                                                 args.graph_lstm_hidden_size, args.add_confidence)

        self.x_to_s = simple_mapping(args.traj_lstm_hidden_size, args.graph_lstm_hidden_size, args.latent_dim,
                                     args.s_dim, self.obs_len)

        self.x_to_z = simple_mapping(args.traj_lstm_hidden_size, args.graph_lstm_hidden_size, 0,
                                     args.z_dim, self.obs_len)

        self.mapping = regressor(args.obs_len, args.fut_len, args.n_coordinates,
                                 args.traj_lstm_hidden_size, args.n_units, args.n_heads,
                                 args.graph_network_out_dims, args.dropout, args.alpha,
                                 args.graph_lstm_hidden_size, None, args.latent_dim, args.add_confidence)

        self.past_decoder = past_decoder(args.obs_len, args.n_coordinates, args.z_dim, args.s_dim)

        self.future_decoder = future_STGAT_decoder(args.obs_len, args.fut_len, args.n_coordinates, args.s_dim,
                                                   args.z_dim, args.teachingratio)

    def forward(self, batch, training_step, **kwargs):

        obs_traj, fut_traj, obs_traj_rel, fut_traj_rel, seq_start_end, = batch

        if self.training:
            if training_step in ["P1", "P2"]:
                env_idx = kwargs.get("env_idx")
                hidden_states = self.variant_encoder(batch, training_step)

                total_recon_loss = 0
                total_e_loss = 0
                for i in range(self.obs_len):
                    q_zgx = self.x_to_z(hidden_states[i], training_step)
                    q_sgthetax = self.x_to_s(hidden_states[i], training_step, self.theta[env_idx])

                    recon_loss = []
                    first_E = []
                    for _ in range(self.num_samples):
                        z_vec = q_zgx.rsample()
                        s_vec = q_sgthetax.rsample()

                        # calculate log(q(z|x))
                        log_qzgx = q_zgx.log_prob(z_vec)

                        # calculate log(q(s|theta, x))
                        log_qsgthetax = q_sgthetax.log_prob(s_vec)

                        # calculate log(p(z))
                        log_pz = self.pw.log_prob(z_vec)

                        # calculate log(p(s|theta))
                        mean = torch.matmul(torch.matmul(self.covwe, torch.inverse(self.covee)), self.theta[env_idx])
                        covmat = self.covww - torch.matmul(torch.matmul(self.covwe, torch.inverse(self.covee)), self.covwe)
                        pwe = MultivariateNormal(mean, covmat)
                        log_psgtheta = pwe.log_prob(s_vec)

                        # calculate log(p(x|z,s))
                        pred_past_rel = self.past_decoder(torch.cat((z_vec, s_vec), dim=1))
                        reconstruction_loss = - l2_loss(pred_past_rel, obs_traj_rel[i], mode="average")

                        recon_loss.append(reconstruction_loss)
                        first_E.append(log_psgtheta + log_pz - log_qzgx - log_qsgthetax)

                    E1 = torch.mean(torch.stack(first_E), dim=0)
                    total_e_loss += torch.mean(E1)
                    total_recon_loss += torch.mean(torch.stack(recon_loss), dim=0)

                return total_recon_loss, total_e_loss

            elif training_step == "P3":
                env_idx = kwargs.get("env_idx")
                concat_hidden_states = self.variant_encoder(batch, training_step)

                q_zgx = self.x_to_z(concat_hidden_states, training_step)
                q_sgthetax = self.x_to_s(concat_hidden_states, training_step, self.theta[env_idx])

                # calculate q(y|theta, x)
                s_vec = []
                z_vec = []
                for _ in range(self.num_samples):
                    z_vec += [q_zgx.rsample()]
                    s_vec += [q_sgthetax.rsample()]

                z_vec = torch.stack(z_vec)
                s_vec = torch.stack(s_vec)
                qygx = self.future_decoder(batch, torch.cat((z_vec, s_vec), dim=2))

                first_E = []
                for _ in range(self.num_samples):
                    z_vec = q_zgx.rsample()
                    s_vec = q_sgthetax.rsample()

                    # calculate log(q(z|x))
                    log_qzgx = q_zgx.log_prob(z_vec)

                    # calculate log(q(s|theta, x))
                    log_qsgthetax = q_sgthetax.log_prob(s_vec)

                    # calculate log(p(z))
                    sldj = torch.zeros(z_vec.shape[0], device=z_vec.device)
                    z_vec_c = z_vec
                    for coupling in self.coupling_layers_z:
                        z_vec_c, sldj = coupling(z_vec_c, sldj)

                    log_pz = self.pw.log_prob(z_vec_c) + sldj

                    # calculate log(p(s|theta))
                    sldj_s = torch.zeros(s_vec.shape[0], device=z_vec.device)
                    s_vec_c = s_vec
                    for coupling in self.coupling_layers_s:
                        s_vec_c, sldj_s = coupling(s_vec_c, sldj_s)

                    sldj_t = torch.zeros(s_vec.shape[0], device=z_vec.device)
                    t_vec_c = self.theta[env_idx].unsqueeze(0)
                    for coupling in self.coupling_layers_theta:
                        t_vec_c, sldj_t = coupling(t_vec_c, sldj_t)

                    mean = torch.matmul(torch.matmul(self.covwe, torch.inverse(self.covee)), t_vec_c.squeeze(0))
                    covmat = self.covww - torch.matmul(torch.matmul(self.covwe, torch.inverse(self.covee)), self.covwe)
                    pwe = MultivariateNormal(mean, covmat)
                    log_psgtheta = pwe.log_prob(s_vec_c) + sldj_s

                    # calculate p(y|z,s,x)
                    p = self.future_decoder(batch, torch.cat((z_vec.unsqueeze(0), s_vec.unsqueeze(0)), dim=2))
                    log_py = torch.zeros(fut_traj_rel.shape[1]).cuda()
                    for i in range(self.fut_len):
                        log_py += p[i].log_prob(fut_traj_rel[i])

                    p_ygzs = torch.exp(log_py)

                    # calculate log(p(x|z,s))
                    pred_past_rel = self.past_decoder(batch, torch.cat((z_vec, s_vec), dim=1))
                    reconstruction_loss = - l2_loss(pred_past_rel, obs_traj_rel, mode="raw") - 0.5 * obs_traj_rel.shape[0] * torch.log(torch.tensor(2 * math.pi * 0.5))

                    A1 = torch.multiply(p_ygzs, reconstruction_loss + log_psgtheta + log_pz - log_qzgx - log_qsgthetax)

                    first_E.append(A1)

                E1 = torch.mean(torch.stack(first_E), dim=0)

                return qygx, E1

            elif training_step == "P4":
                pred_theta = self.mapping(obs_traj_rel)

                return pred_theta

            else:
                pred_theta = self.mapping(obs_traj_rel)
                concat_hidden_states = self.variant_encoder(batch, training_step)

                q_zgx = self.x_to_z(concat_hidden_states, training_step)
                q_sgthetax = self.x_to_s(concat_hidden_states, training_step, pred_theta)

                # calculate q(y|theta, x)
                s_vec = []
                z_vec = []
                for _ in range(self.num_samples):
                    z_vec += [q_zgx.rsample()]
                    s_vec += [q_sgthetax.rsample()]

                z_vec = torch.stack(z_vec)
                s_vec = torch.stack(s_vec)
                qygx = self.future_decoder(batch, torch.cat((z_vec, s_vec), dim=2))

                first_E = []
                for _ in range(self.num_samples):
                    z_vec = q_zgx.rsample()
                    s_vec = q_sgthetax.rsample()

                    # calculate log(q(z|x))
                    log_qzgx = q_zgx.log_prob(z_vec)

                    # calculate log(q(s|theta, x))
                    log_qsgthetax = q_sgthetax.log_prob(s_vec)

                    # calculate log(p(z))
                    sldj = torch.zeros(z_vec.shape[0], device=z_vec.device)
                    z_vec_c = z_vec
                    for coupling in self.coupling_layers_z:
                        z_vec_c, sldj = coupling(z_vec_c, sldj)

                    log_pz = self.pw.log_prob(z_vec_c) + sldj

                    # calculate log(p(s|theta))
                    sldj_s = torch.zeros(s_vec.shape[0], device=z_vec.device)
                    s_vec_c = s_vec
                    for coupling in self.coupling_layers_s:
                        s_vec_c, sldj_s = coupling(s_vec_c, sldj_s)

                    sldj_t = torch.zeros(s_vec.shape[0], device=z_vec.device)
                    t_vec_c = pred_theta
                    for coupling in self.coupling_layers_theta:
                        t_vec_c, sldj_t = coupling(t_vec_c, sldj_t)

                    mean = torch.matmul(torch.matmul(self.covwe, torch.inverse(self.covee)), torch.mean(t_vec_c, dim=0))
                    covmat = self.covww - torch.matmul(torch.matmul(self.covwe, torch.inverse(self.covee)), self.covwe)
                    pwe = MultivariateNormal(mean, covmat)
                    log_psgtheta = pwe.log_prob(s_vec_c) + sldj_s

                    # calculate p(y|z,s,x)
                    p = self.future_decoder(batch, torch.cat((z_vec.unsqueeze(0), s_vec.unsqueeze(0)), dim=2))
                    log_py = torch.zeros(fut_traj_rel.shape[1]).cuda()
                    for i in range(self.fut_len):
                        log_py += p[i].log_prob(fut_traj_rel[i])

                    p_ygzs = torch.exp(log_py)

                    # calculate log(p(x|s,z))
                    pred_past_rel = self.past_decoder(batch, torch.cat((z_vec, s_vec), dim=1))
                    reconstruction_loss = - l2_loss(pred_past_rel, obs_traj_rel, mode="raw") - \
                                          0.5 * 1 / obs_traj_rel.shape[0] * torch.log(torch.tensor(2 * math.pi * 0.5))

                    A1 = torch.multiply(p_ygzs, reconstruction_loss)
                    A2 = torch.multiply(p_ygzs, log_pz + log_psgtheta - log_qzgx - log_qsgthetax)

                    first_E.append(A1 + A2)

                E = torch.mean(torch.stack(first_E), dim=0)

                return qygx, E

        else:
            if training_step == "P7":
                env_idx = kwargs.get("env_idx")
                if env_idx is None:
                    concat_hidden_states = self.variant_encoder(batch, training_step)
                    pred_theta = self.mapping(batch)
                    ps = self.x_to_s(concat_hidden_states, pred_theta)
                    q_zgx = self.x_to_z(concat_hidden_states)

                else:
                    concat_hidden_states = self.variant_encoder(batch, training_step)
                    ps = self.x_to_s(concat_hidden_states, self.theta[env_idx])
                    q_zgx = self.x_to_z(concat_hidden_states)

                return q_zgx, ps

            else:
                env_idx = kwargs.get("env_idx")
                if env_idx is None:
                    concat_hidden_states = self.variant_encoder(batch, training_step)
                    pred_theta = self.mapping(obs_traj_rel)
                    q_sgthetax = self.x_to_s(concat_hidden_states, training_step, pred_theta)
                    q_zgx = self.x_to_z(concat_hidden_states, training_step)

                    # calculate q(y|theta, x)
                    s_vec = []
                    z_vec = []
                    for _ in range(self.num_samples):
                        z_vec += [q_zgx.rsample()]
                        s_vec += [q_sgthetax.rsample()]

                    z_vec = torch.stack(z_vec)
                    s_vec = torch.stack(s_vec)
                    qygx = self.future_decoder(batch, torch.cat((z_vec, s_vec), dim=2))

                else:
                    concat_hidden_states = self.variant_encoder(batch, training_step)
                    q_sgthetax = self.x_to_s(concat_hidden_states, training_step, self.theta[env_idx])
                    q_zgx = self.x_to_z(concat_hidden_states, training_step)

                    # calculate q(y|theta, x)
                    s_vec = []
                    z_vec = []
                    for _ in range(self.num_samples):
                        z_vec += [q_zgx.rsample()]
                        s_vec += [q_sgthetax.rsample()]

                    z_vec = torch.stack(z_vec)
                    s_vec = torch.stack(s_vec)
                    qygx = self.future_decoder(batch, torch.cat((z_vec, s_vec), dim=2))

            return qygx
