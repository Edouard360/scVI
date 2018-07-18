import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

from scvi.models.utils import one_hot


class PlanarNormalizingFlow(nn.Module):
    """
    Planar normalizing flow [Rezende & Mohamed 2015].
    Provides a tighter bound on the ELBO by giving more expressive
    power to the approximate distribution, such as by introducing
    covariance between terms.
    """

    def __init__(self, in_features):
        super(PlanarNormalizingFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(in_features))
        self.w = nn.Parameter(torch.randn(in_features))
        self.b = nn.Parameter(torch.ones(1))

    def forward(self, z):
        # Create uhat such that it is parallel to w
        uw = torch.dot(self.u, self.w)
        muw = -1 + F.softplus(uw)
        uhat = self.u + (muw - uw) * self.w / torch.sum(self.w ** 2)  # .transpose(self.w, 0, -1): for gradient = 0 ?

        # Equation 21 - Transform z
        zwb = torch.mv(z, self.w) + self.b

        f_z = z + (uhat.view(1, -1) * F.tanh(zwb).view(-1, 1))

        # Compute the Jacobian using the fact that
        # tanh(x) dx = 1 - tanh(x)**2
        psi = (1 - F.tanh(zwb) ** 2).view(-1, 1) * self.w.view(1, -1)
        psi_u = torch.mv(psi, uhat)

        # Return the transformed output along
        # with log determninant of J
        logdet_jacobian = torch.log(torch.abs(1 + psi_u) + 1e-8)

        return f_z, logdet_jacobian


class NormalizingFlows(nn.Module):
    """
    Presents a sequence of normalizing flows as a torch.nn.Module.
    """

    def __init__(self, in_features, flow_type=PlanarNormalizingFlow, n_flows=1):
        super(NormalizingFlows, self).__init__()
        self.flows = nn.ModuleList([flow_type(in_features) for _ in range(n_flows)])

    def forward(self, z):
        log_det_jacobian = []

        for flow in self.flows:
            z, j = flow(z)
            log_det_jacobian.append(j)

        return z, sum(log_det_jacobian)


class HF(nn.Module):
    def __init__(self):
        super(HF, self).__init__()

    def forward(self, v, z):
        '''
        :param v: batch_size (B) x latent_size (L)
        :param z: batch_size (B) x latent_size (L)
        :return: z_new = z - 2* v v_T / norm(v,2) * z
        '''
        # v * v_T
        vvT = torch.bmm(v.unsqueeze(2), v.unsqueeze(1))  # v * v_T : batch_dot( B x L x 1 * B x 1 x L ) = B x L x L
        # v * v_T * z
        vvTz = torch.bmm(vvT, z.unsqueeze(2)).squeeze(
            2)  # A * z : batchdot( B x L x L * B x L x 1 ).squeeze(2) = (B x L x 1).squeeze(2) = B x L
        # calculate norm ||v||^2
        norm_sq = torch.sum(v * v, 1)  # calculate norm-2 for each row : B x 1
        norm_sq = norm_sq.expand(norm_sq.size(0), v.size(1))  # expand sizes : B x L
        # calculate new z
        z_new = z - 2 * vvTz / norm_sq  # z - 2 * v * v_T  * z / norm2(v)
        return z_new


class ConditionalCoupling(nn.Module):  # VAPNEV - 2016 - Deep Variational Inference Without Pixel-Wise Reconstruction
    def __init__(self, x_dim, z_dim):
        super(ConditionalCoupling, self).__init__()
        self.alpha = nn.Parameter(torch.randn(x_dim, 2))  # 2 because there is l_z and m_z
        self.beta_1 = nn.Parameter(torch.randn(x_dim, 2))
        self.beta_2 = nn.Parameter(torch.randn(x_dim, 2))
        self.b = nn.Parameter(torch.randn(x_dim, 2))

        self.l1 = nn.Linear(x_dim, x_dim)
        self.l2 = nn.Linear(z_dim, x_dim)

        # We can predefine two masks for x_dim to be split in [1;x_dim//2] [x_dim//2;x_dim]
        self.mask = torch.tensor([1, 0], dtype=torch.uint8).repeat(x_dim // 2 + 1)[:x_dim]

    def forward(self, x, z, mask_id=1):
        # x1..d xd+1..D
        # Conditional coupling should involve z
        # (2016) paper.
        # However it seems implementation from https://github.com/taesung89/real-nvp/blob/master/real_nvp/nn.py
        # Is only using x

        if mask_id:
            x_ = torch.masked_select(x, self.mask)
        else:
            x_ = torch.masked_select(x, 1 - self.mask)
        l_z = self.l1(x_) * self.l2(z) + self.beta_1[:, 0] * self.l1(x_) \
              + self.beta_2[:, 0] * self.l2(z) + self.b[:, 0]
        m_z = self.l1(x_) * self.l2(z) + self.beta_1[:, 1] * self.l1(x_) \
              + self.beta_2[:, 1] * self.l2(z) + self.b[:, 1]
        # log(det(J = df/dx)) = sum(l_z)
        return l_z, m_z

    def y_x(self, x, z, mask_id=1):
        l_z, m_z = self(x, z, mask_id=mask_id)
        mask = self.mask if mask_id else 1 - self.mask
        return torch.masked_select(x, mask) + torch.masked_select(x * torch.exp(l_z) + m_z, 1 - mask)

    def x_y(self, y, z, mask_id=1):
        l_z, m_z = self(y, z, mask_id=mask_id)
        mask = self.mask if mask_id else 1 - self.mask
        return torch.masked_select(y, mask) + torch.masked_select((y - m_z) * torch.exp(-l_z), 1 - mask)


class Coupling(nn.Module):  # 2017 - DENSITY ESTIMATION USING REAL NVP - Google Brain
    def __init__(self, x_dim):
        super(Coupling, self).__init__()

        dim_identity = x_dim // 2
        dim_coupling = x_dim - dim_identity

        self.s = nn.Linear(x_dim, x_dim)  # nn.Linear(x_dim, x_dim) # nn.Parameter(torch.zeros(x_dim)) #
        self.t = nn.Linear(x_dim, x_dim)  # nn.Linear(x_dim, x_dim) # nn.Parameter(torch.zeros(x_dim)) #
        self.s.weight.data[:] = 0
        self.s.bias.data[:] = 0
        self.t.weight.data[:] = 0
        self.t.bias.data[:] = 0

        self.mask = nn.Parameter(torch.cat((torch.ones(dim_identity), torch.zeros(dim_coupling)), dim=0),
                                 requires_grad=False)

    def forward(self, x, logdet):
        mask = self.mask
        logdet += (self.s(x * mask) * (1 - mask)).sum(dim=-1)
        return x * mask + (x * torch.exp(self.s(x * mask)) + self.t(x * mask)) * (1 - mask), logdet


class Permutation(nn.Module):  # A learnable operation that generalizes permutation
    def __init__(self, n_input):
        super(Permutation, self).__init__()
        self.n_input = n_input
        np_w = scipy.linalg.qr(np.random.randn(n_input, n_input))[0].astype('float32')

        np_p, np_l, np_u = scipy.linalg.lu(np_w)
        np_s = np.diag(np_u)
        np_sign_s = np.sign(np_s)
        np_log_s = np.log(abs(np_s))
        np_u = np.triu(np_u, k=1)

        self.sign_s = nn.Parameter(torch.from_numpy(np_sign_s), requires_grad=False)

        self.log_s = nn.Parameter(torch.from_numpy(np_log_s))
        self.u_mat = nn.Parameter(torch.from_numpy(np_u))
        self.l_mat = nn.Parameter(torch.from_numpy(np_l))

        self.p_mat = nn.Parameter(torch.from_numpy(np_p), requires_grad=False)

        self.l_mask = nn.Parameter(torch.tensor(np.tril(np.ones((n_input, n_input), dtype=np.float32), -1)),
                                   requires_grad=False)

    def forward(self, input, logdet):
        logdet += self.log_s.sum(dim=0)
        l_mat_mask = self.l_mat * self.l_mask + torch.eye(self.n_input, dtype=torch.float32).to(input.device)
        u_mat_mask = self.u_mat * self.l_mask.transpose(0, 1) + torch.diag(self.sign_s * torch.exp(self.log_s))
        w = torch.matmul(self.p_mat, torch.matmul(l_mat_mask, u_mat_mask))
        z = torch.matmul(input, w)
        return z, logdet

        # def backward(self, input):
        #     l_mat_mask = self.l_mat * self.l_mask + torch.eye(self.n_input, dtype=torch.float32).to(input.device)
        #     u_mat_mask = self.u_mat * self.l_mask.transpose(0,1) + torch.diag(self.sign_s * torch.exp(self.log_s))
        #     #w = F.linear(self.p_mat, F.linear(l_mat_mask, u_mat_mask))
        #     w = torch.matmul(self.p_mat, torch.matmul(l_mat_mask, u_mat_mask))
        #     torch.inverse(w)
        #     output = torch.matmul(input, w)
        #     return output


class ActNorm(nn.Module):  # Activation Normalization  - along the "gene" dimension
    def __init__(self, n_input):
        super(ActNorm, self).__init__()
        # n_channels = n_input = nb_genes (input to the "permutation")

        self.mean = nn.Parameter(torch.zeros(n_input))
        self.log_s = nn.Parameter(torch.zeros(n_input))  # .init - 0.5*torch.log((input**2).mean(dim=-1))

    def reset_parameters(self, mean, log_s):
        self.mean.data = mean
        self.log_s.data = log_s

    def forward(self, input, log_det):  # .forward is from input:x to output:z
        '''
        :param input:
        :return: centered_input, log_det
        '''
        log_det += (- torch.sum(self.log_s))
        return (input - self.mean) * torch.exp(-self.log_s), log_det


class LinearStatic(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearStatic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)

    def set_parameters(self, gene_dataset):  # forces an invertible linear mapping from z to mu_z
        data = gene_dataset.X.T

        gmm = GaussianMixture(n_components=self.in_features)
        gmm.fit(data)
        numpy_clusters = gmm.predict(data).reshape(-1, 1)
        print([(numpy_clusters == i).mean() for i in range(self.in_features)])

        gene_clusters = torch.from_numpy(numpy_clusters).to(self.weight.device).type(torch.int)
        self.weight = nn.Parameter(one_hot(gene_clusters, self.in_features), requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight)

    def backward(self, x):
        t = self.weight.transpose(0, 1)
        # inverse = torch.inverse(F.linear(self.weight.T,self.weight).detach())
        return F.linear(x, F.linear(torch.inverse(F.linear(t, t)), self.weight))


class LinearInvertible(nn.Linear):  # The linear will most surely be invertible (little chance otherwise)
    def forward(self, input):
        return F.sigmoid(super(LinearInvertible, self).forward(input))

    def inverse_matrix(self):
        w = self.weight
        wt = self.weight.transpose(0, 1)
        w_tag = torch.inverse(torch.matmul(wt, w))
        return torch.matmul(w_tag, wt)

    def backward(self, x):
        x = torch.log(x + 1e-8) - torch.log(1 - x + 1e-8)
        return F.linear(x - self.bias, self.inverse_matrix())
