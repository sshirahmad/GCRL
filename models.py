import torch
import torch.nn as nn
import torch.nn.functional as F
import random


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


class STGAT_encoder(nn.Module):
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
            noise_dim=(8,),
            noise_type="gaussian",
            add_confidence=True,
    ):
        super(STGAT_encoder, self).__init__()

        self.obs_len = obs_len
        self.fut_len = fut_len

        self.gatencoder = GATEncoder(n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha)

        self.graph_lstm_hidden_size = graph_lstm_hidden_size
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.n_coordinates = n_coordinates
        self.add_confidence = add_confidence

        self.pred_lstm_hidden_size = (traj_lstm_hidden_size + graph_lstm_hidden_size + noise_dim[0])

        self.traj_lstm_model = nn.LSTMCell(n_coordinates + add_confidence, traj_lstm_hidden_size)
        self.graph_lstm_model = nn.LSTMCell(graph_network_out_dims, graph_lstm_hidden_size)

        # for P1 and P2 and P3
        self.traj_hidden2pos = nn.Linear(traj_lstm_hidden_size, n_coordinates)
        self.traj_gat_hidden2pos = nn.Linear(traj_lstm_hidden_size + graph_lstm_hidden_size, n_coordinates)

        self.noise_dim = noise_dim
        self.noise_type = noise_type

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

    def forward(self, batch):
        (_, _, obs_traj_rel, _, seq_start_end) = batch
        num_peds = obs_traj_rel.shape[1]
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(num_peds)
        graph_lstm_h_t, graph_lstm_c_t = self.init_hidden_graph_lstm(num_peds)
        traj_lstm_hidden_states = []
        graph_lstm_hidden_states = []

        # traj_lstm (used in step 1,2,3)
        for i in range(self.obs_len):
            traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model(obs_traj_rel[i], (traj_lstm_h_t, traj_lstm_c_t))

            traj_lstm_hidden_states += [traj_lstm_h_t]


        graph_lstm_input = self.gatencoder(torch.stack(traj_lstm_hidden_states), seq_start_end)
        for i in range(self.obs_len):
            graph_lstm_h_t, graph_lstm_c_t = self.graph_lstm_model(graph_lstm_input[i],
                                                                   (graph_lstm_h_t, graph_lstm_c_t))

            graph_lstm_hidden_states += [graph_lstm_h_t]

        encoded_before_noise_hidden = torch.cat((traj_lstm_hidden_states[-1], graph_lstm_hidden_states[-1]), dim=1)
        encoded_future_pred = self.add_noise(encoded_before_noise_hidden, seq_start_end)

        return encoded_future_pred, graph_lstm_hidden_states, traj_lstm_hidden_states


class future_STGAT_decoder(nn.Module):
    def __init__(
            self,
            obs_len,
            fut_len,
            n_coordinates,
            c_dim,
            traj_lstm_hidden_size,
            graph_lstm_hidden_size,
            teacher_forcing_ratio=0.5,
            noise_dim=(8,),
    ):
        super(future_STGAT_decoder, self).__init__()

        self.obs_len = obs_len
        self.fut_len = fut_len
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.n_coordinates = n_coordinates
        self.pred_lstm_hidden_size1 = (traj_lstm_hidden_size + graph_lstm_hidden_size + c_dim + noise_dim[0])
        self.pred_lstm_hidden_size2 = (traj_lstm_hidden_size + graph_lstm_hidden_size + noise_dim[0])
        self.pred_hidden2pos = nn.ModuleList([nn.Linear(self.pred_lstm_hidden_size1, n_coordinates),
                                              nn.Linear(self.pred_lstm_hidden_size2, n_coordinates)])
        self.pred_lstm_model = nn.ModuleList([nn.LSTMCell(n_coordinates, self.pred_lstm_hidden_size1),
                                              nn.LSTMCell(n_coordinates, self.pred_lstm_hidden_size2)])

    def forward(
            self,
            batch,
            pred_lstm_hidden,
            variant_feats,
    ):

        (_, _, obs_traj_rel, _, _) = batch
        pred_traj_rel = []

        pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()
        output = obs_traj_rel[self.obs_len - 1, :, :self.n_coordinates]
        # during training
        if self.training:
            for i in range(self.fut_len):
                teacher_force = random.random() < self.teacher_forcing_ratio
                if teacher_force:
                    input_t = obs_traj_rel[self.obs_len - 2 + i, :, :self.n_coordinates]  # with teacher help
                else:
                    input_t = output

                if variant_feats:
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model[0](input_t, (pred_lstm_hidden, pred_lstm_c_t))
                    output = self.pred_hidden2pos[0](pred_lstm_hidden)
                else:
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model[1](input_t, (pred_lstm_hidden, pred_lstm_c_t))
                    output = self.pred_hidden2pos[1](pred_lstm_hidden)
                pred_traj_rel += [output]
        # during test
        else:
            for i in range(self.fut_len):
                pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model[0](output, (pred_lstm_hidden, pred_lstm_c_t))
                output = self.pred_hidden2pos[0](pred_lstm_hidden)
                pred_traj_rel += [output]

        return torch.stack(pred_traj_rel)


class past_decoder(nn.Module):
    def __init__(
            self,
            n_coordinates,
            c_dim,
    ):
        super(past_decoder, self).__init__()

        self.n_coordinates = n_coordinates
        self.pred_hidden2pos = nn.Linear(c_dim, n_coordinates)

    def forward(self, c_vec):

        pred_traj_rel, = self.pred_hidden2pos(c_vec)

        return pred_traj_rel

class VE(nn.Module):
    def __init__(self,
                 traj_lstm_hidden_size: int,
                 graph_lstm_hidden_size: int,
                 latent_dim: int,
                 num_samples: int,
                 **kwargs) -> None:
        super(VE, self).__init__()

        in_channels = traj_lstm_hidden_size + graph_lstm_hidden_size
        self.latent_dim = latent_dim
        self.fc_mu_theta1 = nn.Linear(in_channels, latent_dim)
        self.fc_var_theta1 = nn.Linear(in_channels, latent_dim)
        self.fc_mu_theta2 = nn.Linear(in_channels, latent_dim)
        self.fc_var_theta2 = nn.Linear(in_channels, latent_dim)
        self.num_samples = num_samples

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
        mu1 = self.fc_mu_theta1(result)
        log_var1 = self.fc_var_theta1(result)
        mu2 = self.fc_mu_theta2(result)
        log_var2 = self.fc_var_theta2(result)

        return [mu1, log_var1, mu2, log_var2]

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu1, log_var1, mu2, log_var2 = self.encode(input)
        theta1 = self.reparameterize(mu1, log_var1)  # TODO num_samples
        theta2 = self.reparameterize(mu2, log_var2)

        return [theta1, theta2, mu1, log_var1, mu2, log_var2]


class simple_mapping(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dims: list,
                 c_dim: int,
                 num_samples: int,
                 **kwargs) -> None:
        super(simple_mapping, self).__init__()

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.mapping = nn.Sequential(*modules)
        self.final_layer = nn.Linear(hidden_dims[-1], c_dim)

    def forward(self, theta2):

        C = self.final_layer(self.mapping(theta2))

        return C


class CRMF(nn.Module):
    def __init__(self,
                 obs_len, fut_len,
                 n_coordinates,
                 traj_lstm_hidden_size,
                 n_units,
                 n_heads,
                 graph_network_out_dims,
                 dropout,
                 alpha,
                 graph_lstm_hidden_size,
                 latent_dim,
                 theta_c_hidden_dim,
                 c_dim,
                 noise_dim=(8,),
                 noise_type="gaussian",
                 teacher_forcing_ratio=0.5,
                 num_samples=20,
                 add_confidence=False):
        super(CRMF, self).__init__()

        self.obs_len = obs_len
        self.num_samples = num_samples
        self.invariant_encoder = STGAT_encoder(obs_len, fut_len, n_coordinates,
                                               traj_lstm_hidden_size, n_units, n_heads,
                                               graph_network_out_dims, dropout, alpha, graph_lstm_hidden_size,
                                               noise_dim, noise_type, add_confidence)

        self.variant_encoder = STGAT_encoder(obs_len, fut_len, n_coordinates,
                                             traj_lstm_hidden_size, n_units, n_heads,
                                             graph_network_out_dims, dropout, alpha, graph_lstm_hidden_size,
                                             noise_dim, noise_type, add_confidence)

        self.variational_mapping = VE(traj_lstm_hidden_size, graph_lstm_hidden_size, latent_dim, num_samples)

        self.theta_to_c = simple_mapping(latent_dim, theta_c_hidden_dim, c_dim, num_samples)

        self.past_decoder = past_decoder(n_coordinates, c_dim)

        self.future_decoder = future_STGAT_decoder(obs_len, fut_len, n_coordinates, traj_lstm_hidden_size,
                                                   graph_lstm_hidden_size, teacher_forcing_ratio, noise_dim)

    def forward(self, batch, training_step):
        encoded_future_pred, _, _ = self.invariant_encoder(batch)

        if training_step == 'P1' and self.training:
            pred_traj_rel = self.future_decoder(batch, encoded_future_pred, False)

            return pred_traj_rel

        if training_step == 'P2' and self.training:
            _, graph_lstm_hidden_states, traj_lstm_hidden_states = self.variant_encoder(batch)

            past_traj_rel = []
            for i in range(self.obs_len):
                theta1, theta2, mu1, log_var1, mu2, log_var2 = self.variational_mapping(torch.cat((graph_lstm_hidden_states[i],
                                                                                                   traj_lstm_hidden_states[i]), dim=1))

                c_vec = self.theta_to_c(theta2)
                past_traj_rel += [self.past_decoder(c_vec)]


            pred_traj_rel = self.future_decoder(batch, torch.cat((encoded_future_pred,c_vec), dim=1), True)










