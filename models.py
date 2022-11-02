import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils import *
import math
from torch.distributions import MultivariateNormal, Gamma, Poisson

eps = 0.0


def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(*shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


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
            add_confidence=True,
    ):
        super(STGAT_encoder_inv, self).__init__()

        self.obs_len = obs_len
        self.fut_len = fut_len

        self.gatencoder = GATEncoder(
            n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha
        )

        self.graph_lstm_hidden_size = graph_lstm_hidden_size
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.n_coordinates = n_coordinates
        self.add_confidence = add_confidence

        self.fc_mu = nn.Linear(traj_lstm_hidden_size + graph_lstm_hidden_size, z_dim)
        self.fc_var = nn.Linear(traj_lstm_hidden_size + graph_lstm_hidden_size, z_dim)

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
            mu = self.fc_mu(encoded_before_noise_hidden)
            logvar = self.fc_var(encoded_before_noise_hidden)

            z = MultivariateNormal(mu, torch.diag_embed(torch.exp(logvar) + eps))

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
            z_dim,
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
            return torch.cat((graph_lstm_hidden_states[-1], traj_lstm_hidden_states[-1]), dim=1)


class future_STGAT_decoder(nn.Module):
    def __init__(
            self,
            obs_len,
            fut_len,
            n_coordinates,
            c_dim,
            z_dim,
            teacher_forcing_ratio=0.5,
            noise_dim=(8,),
            noise_type="gaussian",
            var_p=0.5,
    ):
        super(future_STGAT_decoder, self).__init__()

        self.obs_len = obs_len
        self.fut_len = fut_len
        self.var_p = var_p
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.noise_dim = noise_dim
        self.noise_type = noise_type

        self.n_coordinates = n_coordinates
        self.pred_lstm_hidden_size1 = z_dim + c_dim
        self.pred_lstm_hidden_size2 = z_dim
        self.pred_hidden2pos = nn.ModuleList([nn.Linear(self.pred_lstm_hidden_size1, n_coordinates),
                                              nn.Linear(self.pred_lstm_hidden_size2, n_coordinates)])
        self.pred_lstm_model = nn.ModuleList([nn.LSTMCell(n_coordinates, self.pred_lstm_hidden_size1),
                                              nn.LSTMCell(n_coordinates, self.pred_lstm_hidden_size2)])

        self.scale = torch.tensor(1.).cuda().requires_grad_()

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
            _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
        decoder_h = torch.cat(_list, dim=0)
        return decoder_h

    def forward(
            self,
            batch,
            pred_lstm_hidden,
            training_step,
            variant_feats,
    ):

        (_, _, obs_traj_rel, fut_traj_rel, seq_start_end) = batch
        pred_traj_rel = []

        input_t = obs_traj_rel[self.obs_len - 1, :, :self.n_coordinates]
        output = input_t
        pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()
        p = []
        # during training
        if self.training:
            for i in range(self.fut_len):
                if i >= 1:
                    teacher_force = random.random() < self.teacher_forcing_ratio
                    if teacher_force:
                        input_t = fut_traj_rel[i - 1, :, :self.n_coordinates]  # with teacher help
                    else:
                        input_t = output

                if variant_feats:
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model[0](input_t,
                                                                              (pred_lstm_hidden, pred_lstm_c_t))
                    output = self.pred_hidden2pos[0](pred_lstm_hidden)
                else:
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model[1](input_t,
                                                                              (pred_lstm_hidden, pred_lstm_c_t))
                    output = self.pred_hidden2pos[1](pred_lstm_hidden)
                pred_traj_rel += [output]
                p += [MultivariateNormal(output, torch.diag_embed(self.var_p * torch.ones(fut_traj_rel.size(1), 2).cuda()))]

            return torch.stack(pred_traj_rel)

        # during test
        else:
            for i in range(self.fut_len):
                if training_step == "P3":
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model[1](output, (pred_lstm_hidden, pred_lstm_c_t))
                    output = self.pred_hidden2pos[1](pred_lstm_hidden)
                else:
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model[0](output, (pred_lstm_hidden, pred_lstm_c_t))
                    output = self.pred_hidden2pos[0](pred_lstm_hidden)
                pred_traj_rel += [output]

            return torch.stack(pred_traj_rel)


class past_decoder(nn.Module):
    def __init__(
            self,
            obs_len,
            n_coordinates,
            z_dim,
            c_dim,
    ):
        super(past_decoder, self).__init__()

        self.obs_len = obs_len

        self.n_coordinates = n_coordinates
        self.pred_lstm_hidden_size1 = z_dim + c_dim
        self.pred_lstm_hidden_size2 = z_dim
        self.pred_hidden2pos = nn.ModuleList([nn.Linear(self.pred_lstm_hidden_size1, n_coordinates),
                                              nn.Linear(self.pred_lstm_hidden_size2, n_coordinates)])
        self.pred_lstm_model = nn.ModuleList([nn.LSTMCell(n_coordinates, self.pred_lstm_hidden_size1),
                                              nn.LSTMCell(n_coordinates, self.pred_lstm_hidden_size2)])
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
            variant_feats
    ):

        (_, _, obs_traj_rel, _, seq_start_end) = batch
        pred_traj_rel = []

        pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()
        for i in range(self.obs_len):
            if i >= 1:
                input_t = obs_traj_rel[i - 1, :, :self.n_coordinates]
            else:
                input_t = obs_traj_rel[0, :, :self.n_coordinates]

            if variant_feats:
                pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model[0](input_t, (pred_lstm_hidden, pred_lstm_c_t))
                output = self.pred_hidden2pos[0](pred_lstm_hidden)
            else:
                pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model[1](input_t, (pred_lstm_hidden, pred_lstm_c_t))
                output = self.pred_hidden2pos[1](pred_lstm_hidden)

            pred_traj_rel += [output]

        return torch.stack(pred_traj_rel)


class VE(nn.Module):
    def __init__(self,
                 traj_lstm_hidden_size: int,
                 graph_lstm_hidden_size: int,
                 latent_dim: int,
                 **kwargs) -> None:
        super(VE, self).__init__()

        in_channels = traj_lstm_hidden_size + graph_lstm_hidden_size
        self.latent_dim = latent_dim
        self.fc_mu_theta = nn.Linear(in_channels, latent_dim)
        self.fc_var_theta = nn.Linear(in_channels, latent_dim)

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

    def encode(self, lstm_hiddens):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = torch.flatten(lstm_hiddens, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu_theta(result)
        logvar = self.fc_var_theta(result)

        return [mu, logvar]

    def forward(self, input):
        mu, logvar = self.encode(input)
        qtheta = MultivariateNormal(mu, torch.diag_embed(torch.exp(logvar) + eps))

        return qtheta, mu, logvar


class simple_mapping(nn.Module):
    def __init__(self,
                 traj_lstm_hidden_size: int,
                 graph_lstm_hidden_size: int,
                 latent_dim: int,
                 hidden_dims: list,
                 s_dim: int,
                 **kwargs) -> None:
        super(simple_mapping, self).__init__()


        if hidden_dims is None:
            hidden_dims = [16, 32]

        modules = []
        in_channels = latent_dim + traj_lstm_hidden_size + graph_lstm_hidden_size
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.mapping = nn.Sequential(*modules)
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

    def forward(self, theta, hidden_states):

        if len(theta.size()) == 1:
            theta_rep = theta.repeat(hidden_states.shape[0], 1)
        else:
            theta_rep = theta
        vec = torch.cat((theta_rep, hidden_states), dim=1)
        mu = self.fc_mu(self.mapping(vec))
        logvar = self.fc_logvar(self.mapping(vec))
        ps = MultivariateNormal(mu, torch.diag_embed(torch.exp(logvar) + eps))

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
        self.ptheta = MultivariateNormal(torch.zeros(self.latent_dim).cuda(), torch.diag(torch.ones(self.latent_dim)).cuda())
        self.pz = MultivariateNormal(torch.zeros(self.z_dim).cuda(), torch.diag(torch.ones(self.z_dim)).cuda())
        self.invariant_encoder = STGAT_encoder_inv(args.obs_len, args.fut_len, args.n_coordinates,
                                               args.traj_lstm_hidden_size, args.n_units, args.n_heads,
                                               args.graph_network_out_dims, args.dropout, args.alpha,
                                               args.graph_lstm_hidden_size,
                                               args.z_dim, args.add_confidence)

        self.variant_encoder = STGAT_encoder_var(args.obs_len, args.fut_len, args.n_coordinates,
                                             args.traj_lstm_hidden_size, args.n_units, args.n_heads,
                                             args.graph_network_out_dims, args.dropout, args.alpha,
                                             args.graph_lstm_hidden_size,
                                             args.z_dim, args.add_confidence)

        self.variational_mapping = VE(args.traj_lstm_hidden_size, args.graph_lstm_hidden_size, args.latent_dim)

        self.theta_to_s = simple_mapping(args.traj_lstm_hidden_size, args.graph_lstm_hidden_size, args.latent_dim, None, args.s_dim)

        self.past_decoder = past_decoder(args.obs_len, args.n_coordinates, args.z_dim, args.s_dim)

        self.future_decoder = future_STGAT_decoder(args.obs_len, args.fut_len, args.n_coordinates, args.s_dim,
                                                   args.z_dim, args.teachingratio,
                                                   args.noise_dim, args.noise_type)

    def forward(self, batch, training_step, **kwargs):

        obs_traj, fut_traj, obs_traj_rel, fut_traj_rel, seq_start_end, = batch

        if self.training:
            if training_step in ["P1", "P2"]:
                pred_traj_rel_inv = self.invariant_encoder(batch, training_step)
                pred_traj_rel_var = self.variant_encoder(batch, training_step)

                return pred_traj_rel_inv, pred_traj_rel_var

            elif training_step == 'P3':
                dummy_w = kwargs.get("dummy_w")
                # calculate q(y|x)
                first_E = []
                q_zgx = self.invariant_encoder(batch, training_step)
                for _ in range(self.num_samples):
                    z_vec = q_zgx.rsample()
                    pred_traj_rel_fut = self.future_decoder(batch, z_vec, training_step, False)
                    predict_loss = -l2_loss(pred_traj_rel_fut * dummy_w, fut_traj_rel, mode="raw") - \
                                   0.5 * 1/fut_traj_rel.shape[0] * torch.log(torch.tensor(2 * math.pi * 0.5))
                    first_E.append(predict_loss)

                log_q_ygx = torch.mean(torch.stack(first_E), dim=0)

                first_E = []
                for _ in range(self.num_samples):
                    z_vec = q_zgx.rsample()
                    qprob_z = q_zgx.log_prob(z_vec)
                    prob_z = self.pz.log_prob(z_vec)
                    pred_past_rel = self.past_decoder(batch, z_vec, False)
                    reconstruction_loss = -l2_loss(pred_past_rel * dummy_w, obs_traj_rel, mode="raw") -\
                                          0.5 * 1/obs_traj_rel.shape[0] * torch.log(torch.tensor(2 * math.pi * 0.5))
                    pred_traj_rel_fut = self.future_decoder(batch, z_vec, training_step, False)
                    predict_loss = -l2_loss(pred_traj_rel_fut * dummy_w, fut_traj_rel, mode="raw") -\
                                   0.5 * 1/fut_traj_rel.shape[0] * torch.log(torch.tensor(2 * math.pi * 0.5))
                    p_ygz = torch.exp(predict_loss)

                    A1 = torch.multiply(p_ygz, reconstruction_loss)
                    A2 = torch.multiply(p_ygz, prob_z - qprob_z)

                    first_E.append(A1 + A2)

                E = torch.mean(torch.stack(first_E), dim=0)

                return log_q_ygx, E

            else:
                env_idx = kwargs.get("env_idx")

                concat_hidden_states = self.variant_encoder(batch, training_step)

                first_E = []
                q_zgx = self.invariant_encoder(batch, training_step)
                q_thetagx, mu_theta, logvar_theta = self.variational_mapping(concat_hidden_states)
                p_sgtheta = self.theta_to_s(self.theta[env_idx], concat_hidden_states)

                # calculate q(y|theta, x)
                for _ in range(self.num_samples):
                    z_vec = q_zgx.rsample()
                    s_vec = p_sgtheta.rsample()
                    pred_traj_rel_fut = self.future_decoder(batch, torch.cat((z_vec, s_vec), dim=1), training_step, True)
                    predict_loss = -l2_loss(pred_traj_rel_fut, fut_traj_rel, mode="raw") -\
                                   0.5 * 1/fut_traj_rel.shape[0] * torch.log(torch.tensor(2 * math.pi * 0.5))
                    first_E.append(predict_loss)

                log_qygthetax = torch.mean(torch.stack(first_E), dim=0)
                log_qthetagx = q_thetagx.log_prob(self.theta[env_idx].repeat(obs_traj_rel.shape[1], 1))
                log_ptheta = self.ptheta.log_prob(self.theta[env_idx].repeat(obs_traj_rel.shape[1], 1))

                first_E = []
                for _ in range(self.num_samples):
                    z_vec = q_zgx.rsample()
                    s_vec = p_sgtheta.rsample()

                    log_pz = self.pz.log_prob(z_vec)
                    log_qzgx = q_zgx.log_prob(z_vec)

                    pred_traj_rel_fut = self.future_decoder(batch, torch.cat((z_vec, s_vec), dim=1), training_step, True)
                    predict_loss = -l2_loss(pred_traj_rel_fut, fut_traj_rel, mode="raw") - \
                                   0.5 * 1 / fut_traj_rel.shape[0] * torch.log(torch.tensor(2 * math.pi * 0.5))
                    p_ygzs = torch.exp(predict_loss)

                    pred_past_rel = self.past_decoder(batch, torch.cat((z_vec, s_vec), dim=1), True)
                    reconstruction_loss = -l2_loss(pred_past_rel, obs_traj_rel, mode="raw") - \
                                          0.5 * 1 / obs_traj_rel.shape[0] * torch.log(torch.tensor(2 * math.pi * 0.5))

                    A1 = torch.multiply(p_ygzs, reconstruction_loss)
                    A2 = torch.multiply(p_ygzs, log_pz + log_ptheta - log_qzgx - log_qthetagx)

                    first_E.append(A1 + A2)

                E = torch.mean(torch.stack(first_E), dim=0)

                return log_qygthetax, log_qthetagx, E

        else:
            if training_step == "P3":
                q_zgx = self.invariant_encoder(batch, training_step)
                pred_traj_rel = []
                for _ in range(self.num_samples):
                    # P(z|x)
                    z_vec = q_zgx.rsample()

                    # p(y|z,c)
                    pred_traj_rel += [self.future_decoder(batch, z_vec, training_step, False)]

            else:
                env_idx = kwargs.get("env_idx")
                concat_hidden_states = self.variant_encoder(batch, training_step)

                if env_idx is not None:
                    ps = self.theta_to_s(self.theta[env_idx], concat_hidden_states)
                    p_zgx = self.invariant_encoder(batch, training_step)
                    pred_traj_rel = []
                    for _ in range(self.num_samples):
                        # p(s|theta,x)
                        s_vec = ps.sample()

                        # P(z|x)
                        z_vec = p_zgx.sample()

                        # p(y|z,c)
                        pred_traj_rel += [self.future_decoder(batch, torch.cat((z_vec, s_vec), dim=1), training_step, True)]
                else:

                    qtheta = self.variational_mapping(concat_hidden_states)
                    theta = qtheta.sample()
                    ps = self.theta_to_s(theta, concat_hidden_states)
                    p_zgx = self.invariant_encoder(batch, training_step)
                    pred_traj_rel = []
                    for _ in range(self.num_samples):
                        # p(s|theta,x)
                        s_vec = ps.sample()

                        # P(z|x)
                        z_vec = p_zgx.sample()

                        # p(y|z,c)
                        pred_traj_rel += [self.future_decoder(batch, torch.cat((z_vec, s_vec), dim=1), training_step, True)]

            return pred_traj_rel

