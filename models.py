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


class encoder(nn.Module):
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
        super(encoder, self).__init__()

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


class future_decoder(nn.Module):
    def __init__(
            self,
            obs_len,
            fut_len,
            n_coordinates,
            s_dim,
            z_dim,
            teacher_forcing_ratio=0.5,
            var_p=0.5,
    ):
        super(future_decoder, self).__init__()

        self.obs_len = obs_len
        self.fut_len = fut_len
        self.var_p = var_p
        self.teacher_forcing_ratio = teacher_forcing_ratio

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

                pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(input_t, (pred_lstm_hidden, pred_lstm_c_t))
                output = self.pred_hidden2pos(pred_lstm_hidden)

                pred_traj_rel += [output]
                p += [MultivariateNormal(output,
                                         torch.diag_embed(self.var_p * torch.ones(fut_traj_rel.size(1), 2).cuda()))]

            return p

        # during test
        else:
            for i in range(self.fut_len):
                pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(output, (pred_lstm_hidden, pred_lstm_c_t))
                output = self.pred_hidden2pos(pred_lstm_hidden)
                pred_traj_rel += [output]

            return torch.stack(pred_traj_rel)


class past_decoder(nn.Module):
    def __init__(
            self,
            obs_len,
            n_coordinates,
            z_dim,
            s_dim,
    ):
        super(past_decoder, self).__init__()

        self.obs_len = obs_len

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
    def __init__(self,
                 obs_len,
                 n_coordinates,
                 latent_dim: int,
                 hidden_dims=None,
                 **kwargs) -> None:
        super(regressor, self).__init__()

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

        self.mapping = nn.Sequential(*modules)
        self.final = nn.Linear(hidden_dims[-1], latent_dim)

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

    def forward(self, input):
        result = torch.flatten(input.permute(1, 0, 2), start_dim=1)
        pred_theta = self.final(self.mapping(result))

        return pred_theta


class simple_mapping(nn.Module):
    def __init__(self,
                 traj_lstm_hidden_size: int,
                 graph_lstm_hidden_size: int,
                 latent_dim: int,
                 s_dim: int,
                 hidden_dims=None,
                 **kwargs) -> None:
        super(simple_mapping, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64]

        modules = []
        in_channels = latent_dim + traj_lstm_hidden_size + graph_lstm_hidden_size
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.s_dim = s_dim
        self.mapping = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], s_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], s_dim)
        self.fc_cov = nn.Linear(hidden_dims[-1], (s_dim * s_dim - s_dim) // 2)

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
        if hidden_states is None:
            mu = self.fc_mu(self.mapping(theta))
            logvar = self.fc_logvar(self.mapping(theta))

            ps = MultivariateNormal(mu, torch.diag_embed(torch.exp(logvar)))

        else:
            if len(theta.size()) == 1:
                theta_rep = theta.repeat(hidden_states.shape[0], 1)
            else:
                theta_rep = theta
            vec = torch.cat((theta_rep, hidden_states), dim=1)
            mu = self.fc_mu(self.mapping(vec))
            logvar = self.fc_logvar(self.mapping(vec))

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
        self.mean = nn.Parameter(torch.zeros(args.z_dim, device="cuda"))
        self.logvar = nn.Parameter(torch.zeros(args.z_dim, device="cuda"))
        self.pz = MultivariateNormal(self.mean, torch.diag(torch.exp(self.logvar)))
        # self.pz = MultivariateNormal(torch.zeros(self.z_dim).cuda(), torch.diag(torch.ones(self.z_dim).cuda()))

        self.encoder = encoder(args.obs_len, args.fut_len, args.n_coordinates,
                               args.traj_lstm_hidden_size, args.n_units, args.n_heads,
                               args.graph_network_out_dims, args.dropout, args.alpha,
                               args.graph_lstm_hidden_size, args.add_confidence)

        self.theta_to_s = simple_mapping(0, 0, args.latent_dim, args.s_dim)
        self.thetax_to_s = simple_mapping(args.traj_lstm_hidden_size, args.graph_lstm_hidden_size, args.latent_dim,
                                          args.s_dim)
        self.thetax_to_z = simple_mapping(args.traj_lstm_hidden_size, args.graph_lstm_hidden_size, args.latent_dim,
                                          args.z_dim)

        self.mapping = regressor(args.obs_len, args.n_coordinates, args.latent_dim)

        self.past_decoder = past_decoder(args.obs_len, args.n_coordinates, args.z_dim, args.s_dim)

        self.future_decoder = future_decoder(args.obs_len, args.fut_len, args.n_coordinates, args.s_dim,
                                             args.z_dim, args.teachingratio)

    def forward(self, batch, training_step, **kwargs):

        obs_traj, fut_traj, obs_traj_rel, fut_traj_rel, seq_start_end, = batch

        if self.training:
            if training_step in ["P1", "P2"]:
                pred_traj_rel = self.encoder(batch, training_step)

                return pred_traj_rel

            elif training_step == "P3":
                env_idx = kwargs.get("env_idx")
                concat_hidden_states = self.encoder(batch, training_step)

                q_zgtx = self.thetax_to_z(self.theta[env_idx], concat_hidden_states)
                p_sgt = self.theta_to_s(self.theta[env_idx], None)
                q_sgtx = self.thetax_to_s(self.theta[env_idx], concat_hidden_states)

                # calculate q(y|theta, x)
                first_E = []
                for _ in range(self.num_samples):
                    z_vec = q_zgtx.rsample()
                    s_vec = q_sgtx.rsample()
                    p_ygz = self.future_decoder(batch, torch.cat((z_vec, s_vec), dim=1))
                    proby_mat = torch.stack(
                        [torch.exp(p_ygz[i].log_prob(fut_traj_rel[i])) for i in range(fut_traj.shape[0])])

                    first_E.append(torch.prod(proby_mat, dim=0))

                qygthetax = torch.mean(torch.stack(first_E), dim=0)

                first_E = []
                for _ in range(self.num_samples):
                    z_vec = q_zgtx.rsample()
                    s_vec = q_sgtx.rsample()

                    log_pz = self.pz.log_prob(z_vec)
                    log_qzgtx = q_zgtx.log_prob(z_vec)
                    log_psgt = p_sgt.log_prob(s_vec)
                    log_qsgtx = q_sgtx.log_prob(s_vec)

                    p_ygz = self.future_decoder(batch, torch.cat((z_vec, s_vec), dim=1))
                    proby_mat = torch.stack(
                        [torch.exp(p_ygz[i].log_prob(fut_traj_rel[i])) for i in range(fut_traj.shape[0])])

                    p_ygzs = torch.prod(proby_mat, dim=0)

                    pred_past_rel = self.past_decoder(batch, torch.cat((z_vec, s_vec), dim=1))
                    reconstruction_loss = - l2_loss(pred_past_rel, obs_traj_rel, mode="raw") - \
                                          0.5 * 1 / obs_traj_rel.shape[0] * torch.log(torch.tensor(2 * math.pi * 0.5))

                    A1 = torch.multiply(p_ygzs, reconstruction_loss)
                    A2 = torch.multiply(p_ygzs, log_pz + log_psgt - log_qzgtx - log_qsgtx)

                    first_E.append(A1 + A2)

                E = torch.mean(torch.stack(first_E), dim=0)

                return qygthetax, E

            elif training_step == "P4":
                pred_theta = self.mapping(obs_traj_rel)

                return pred_theta

            else:
                pred_theta = self.mapping(obs_traj_rel)
                concat_hidden_states = self.encoder(batch, training_step)

                q_zgtx = self.thetax_to_z(pred_theta, concat_hidden_states)
                p_sgt = self.theta_to_s(pred_theta, None)
                q_sgtx = self.thetax_to_s(pred_theta, concat_hidden_states)

                # calculate q(y|theta, x)
                first_E = []
                for _ in range(self.num_samples):
                    z_vec = q_zgtx.rsample()
                    s_vec = q_sgtx.rsample()
                    p_ygz = self.future_decoder(batch, torch.cat((z_vec, s_vec), dim=1))
                    proby_mat = torch.stack(
                        [torch.exp(p_ygz[i].log_prob(fut_traj_rel[i])) for i in range(fut_traj.shape[0])])

                    first_E.append(torch.prod(proby_mat, dim=0))

                qygthetax = torch.mean(torch.stack(first_E), dim=0)

                first_E = []
                for _ in range(self.num_samples):
                    z_vec = q_zgtx.rsample()
                    s_vec = q_sgtx.rsample()

                    log_pz = self.pz.log_prob(z_vec)
                    log_qzgtx = q_zgtx.log_prob(z_vec)
                    log_psgt = p_sgt.log_prob(s_vec)
                    log_qsgtx = q_sgtx.log_prob(s_vec)

                    p_ygz = self.future_decoder(batch, torch.cat((z_vec, s_vec), dim=1))
                    proby_mat = torch.stack(
                        [torch.exp(p_ygz[i].log_prob(fut_traj_rel[i])) for i in range(fut_traj.shape[0])])

                    p_ygzs = torch.prod(proby_mat, dim=0)

                    pred_past_rel = self.past_decoder(batch, torch.cat((z_vec, s_vec), dim=1))
                    reconstruction_loss = - l2_loss(pred_past_rel, obs_traj_rel, mode="raw") - \
                                          0.5 * 1 / obs_traj_rel.shape[0] * torch.log(torch.tensor(2 * math.pi * 0.5))

                    A1 = torch.multiply(p_ygzs, reconstruction_loss)
                    A2 = torch.multiply(p_ygzs, log_pz + log_psgt - log_qzgtx - log_qsgtx)

                    first_E.append(A1 + A2)

                E = torch.mean(torch.stack(first_E), dim=0)

                return qygthetax, E

        else:
            if training_step == "P3":
                env_idx = kwargs.get("env_idx")

                concat_hidden_states = self.encoder(batch, training_step)

                p_sgtx = self.thetax_to_s(self.theta[env_idx], concat_hidden_states)
                p_zgtx = self.thetax_to_z(self.theta[env_idx], concat_hidden_states)

                # p(s|theta,x)
                s_vec = p_sgtx.sample()

                # P(z|x)
                z_vec = p_zgtx.sample()

                # p(y|z,c)
                pred_traj_rel = self.future_decoder(batch, torch.cat((z_vec, s_vec), dim=1))

            elif training_step == "P7":
                env_idx = kwargs.get("env_idx")
                if env_idx is None:
                    concat_hidden_states = self.encoder(batch, training_step)
                    theta = self.mapping(obs_traj_rel)
                    ps = self.thetax_to_s(theta, concat_hidden_states)
                    q_zgx = self.thetax_to_z(theta, concat_hidden_states)

                else:
                    concat_hidden_states = self.encoder(batch, training_step)
                    ps = self.thetax_to_s(self.theta[env_idx], concat_hidden_states)
                    q_zgx = self.thetax_to_z(self.theta[env_idx], concat_hidden_states)

                return q_zgx, ps

            else:
                concat_hidden_states = self.encoder(batch, training_step)
                theta = self.mapping(obs_traj_rel)
                p_sgtx = self.thetax_to_s(theta, concat_hidden_states)
                p_zgtx = self.thetax_to_z(theta, concat_hidden_states)

                # p(s|theta,x)
                s_vec = p_sgtx.sample()

                # P(z|x)
                z_vec = p_zgtx.sample()

                # p(y|z,c)
                pred_traj_rel = self.future_decoder(batch, torch.cat((z_vec, s_vec), dim=1))

            return pred_traj_rel
