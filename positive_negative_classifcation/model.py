import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
from typing import List, Tuple, Optional, Dict
from resnet_no_bn import resnet18

from distribution_pooling_filter import DistributionPoolingFilter

import math

EPS = 1e-8


class GMM(nn.Module):
    """
    Distribution-based pooling via an amortized, bag-conditioned GMM.

    Key ideas (vs your current pooled-MLP -> params):
    1) Learn soft assignments r_{i,k} from instance features (attention/soft-clustering).
    2) Optional instance importance weights w_i (paper-like beta_i).
    3) Compute (pi, mu, sigma) as weighted moments -> stable, permutation-invariant, set-conditioned.
    4) Create a fixed-length "distribution embedding" by evaluating log p(v_m) on learned D-dim probes
       (analogous to KDE bins, but in 512D via probes instead of an impossible grid).

    Inputs:
      f: [B, N, D] instance features (D=512, N=64)

    Outputs (model_params dict):
      pis:   [B, K]
      means: [B, K, D]
      sigmas:[B, K, D]  (diagonal stddev)
      r:     [B, N, K]  (responsibilities)
      w:     [B, N]     (instance weights)
    """

    def __init__(
        self,
        D: int = 512,
        gmm_K: int = 8,
        hidden: int = 256,
        use_instance_weights: bool = False,
        min_sigma: float = 0.05,
        drop_gaussian_const: bool = True,  # drop -0.5*D*log(2pi) in embeddings
        gmm_temp: float = 1.0,  # scale log densities by 1/tau
        gmm_M: int = 64,  # number of learned probes (distribution "bins")
        normalize_features: bool = True,  # LayerNorm on instance features NOTE WE ALREADY DOING
        gmm_normalize_probes: bool = True,  # LayerNorm on probes for scale compatibility,
        gmm_add_tanh: bool = False,
        gmm_apply_l2_norm: bool = False,
    ):
        super().__init__()
        self.D = int(D)
        self.gmm_K = int(gmm_K)
        self.min_sigma = float(min_sigma)
        self.use_instance_weights = bool(use_instance_weights)
        self.drop_gaussian_const = bool(drop_gaussian_const)
        self.gmm_temp = float(gmm_temp)
        self.normalize_features = bool(normalize_features)
        self.normalize_probes = bool(gmm_normalize_probes)
        self.add_tanh = bool(gmm_add_tanh)
        self.apply_l2_norm = bool(gmm_apply_l2_norm)

        if (gmm_apply_l2_norm == gmm_add_tanh) & (gmm_add_tanh == True):
            raise ValueError("Not applicable")
        # self.learnable_temp = nn.Parameter(torch.tensor(self.gmm_temp))
        # Normalization helps a LOT in D=512 for stable densities
        self.feat_norm = nn.LayerNorm(D) if normalize_features else nn.Identity()
        self.probe_norm = nn.LayerNorm(D) if gmm_normalize_probes else nn.Identity()
        if self.add_tanh:
            self.probe_norm = nn.Identity()
        if self.apply_l2_norm:
            self.probe_norm = nn.Identity()
            self.feat_norm = nn.Identity()
        # self.logP_norm = nn.LayerNorm(gmm_M)

        # Responsibility network: g(f_i) -> logits over K components
        self.r_net = nn.Sequential(
            nn.Linear(D, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, gmm_K),
        )

        # Optional instance-importance network: a(f_i) -> scalar logit, then softmax over instances
        if self.use_instance_weights:
            self.w_net = nn.Sequential(
                nn.Linear(D, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.w_net = None

        # Learned probes (distribution "bins") in R^D
        self.gmm_M = int(gmm_M)
        if self.gmm_M > 0:
            self.probes = nn.Parameter(torch.randn(self.gmm_M, D) * 0.02)
            # self.probes = nn.Parameter(
            #     torch.empty(self.gmm_M, D).uniform_(-1, 1)
            # )  # to make values between [-1,1]
        else:
            self.probes = None

    # -----------------------------
    # Set-conditioned GMM parameterization
    # -----------------------------
    def forward(self, f: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        f: [B, N, D]
        returns dict(pis, means, sigmas, r, w)
        """
        assert f.dim() == 3, f"Expected f [B,N,D], got {tuple(f.shape)}"
        B, N, D = f.shape
        assert D == self.D, f"Expected D={self.D}, got D={D}"

        f = self.feat_norm(f)  # [B,N,D]
        if self.add_tanh:
            f = (f - f.min()) / (f.max() - f.min())
        if self.apply_l2_norm:
            f = f.normalize(f, p=2, dim=-1)

        # responsibilities: r_{i,k}
        r_logits = self.r_net(f)  # [B,N,K]
        assert r_logits.shape == (B, N, self.gmm_K), f"r_logits {r_logits.shape}"
        r = F.softmax(r_logits, dim=-1)  # [B,N,K]

        # instance weights: w_i  (paper-like beta_i)
        if self.use_instance_weights:
            w_logits = self.w_net(f).squeeze(-1)  # [B,N]
            assert w_logits.shape == (B, N), f"w_logits {w_logits.shape}"
            w = F.softmax(w_logits, dim=1)  # [B,N]
        else:
            w = f.new_full((B, N), 1.0 / float(N))

        # combined weights a_{i,k} = w_i * r_{i,k}
        a = r * w.unsqueeze(-1)  # [B,N,K]
        assert a.shape == (B, N, self.gmm_K), f"a {a.shape}"

        # mixture weights pi_k
        pis = a.sum(dim=1)  # [B,K]
        pis = pis / (pis.sum(dim=-1, keepdim=True) + EPS)
        assert pis.shape == (B, self.gmm_K), f"pis {pis.shape}"

        # denom per component
        denom = a.sum(dim=1) + EPS  # [B,K]
        assert denom.shape == (B, self.gmm_K), f"denom {denom.shape}"

        # means: mu_k = sum_i a_{i,k} f_i / sum_i a_{i,k}
        # do it as (a^T @ f) with explicit transpose to avoid broadcast errors
        # a: [B,N,K] -> [B,K,N]
        means_num = torch.bmm(a.transpose(1, 2), f)  # [B,K,D]
        means = means_num / denom.unsqueeze(-1)  # [B,K,D]
        assert means.shape == (B, self.gmm_K, D), f"means {means.shape}"

        # variances (diagonal): E[(f-mu)^2] under weights a
        # Compute diff with explicit expand so dim order is guaranteed.
        f_exp = f.unsqueeze(2).expand(B, N, self.gmm_K, D)  # [B,N,K,D]
        mu_exp = means.unsqueeze(1).expand(B, N, self.gmm_K, D)  # [B,N,K,D]
        diff2 = (f_exp - mu_exp).pow(2)  # [B,N,K,D]
        assert diff2.shape == (B, N, self.gmm_K, D), f"diff2 {diff2.shape}"

        # weighted sum over N: sum_i a_{i,k} * diff2_{i,k,d}
        var_num = (a.unsqueeze(-1) * diff2).sum(dim=1)  # [B,K,D]
        var = var_num / denom.unsqueeze(-1)  # [B,K,D]
        var = var + (self.min_sigma**2)
        sigmas = torch.sqrt(var + EPS)  # [B,K,D]
        sigmas = torch.clamp(sigmas, min=self.min_sigma)

        return dict(pis=pis, means=means, sigmas=sigmas, r=r, w=w, f=f)

    # -----------------------------
    # Core log-density: log p(x) under diagonal-cov GMM
    # -----------------------------
    def gmm_log_density(
        self, model_params: Dict[str, torch.Tensor], x: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [B, M, D]
        returns logp: [B, M]
        """
        pis = model_params["pis"]  # [B, K]
        mus = model_params["means"]  # [B, K, D]
        sig = model_params["sigmas"]  # [B, K, D]

        assert x.dim() == 3 and x.size(-1) == self.D
        B, M, D = x.shape

        # stabilize
        sig = torch.clamp(sig, min=self.min_sigma)

        # expand
        x_exp = x.unsqueeze(2)  # [B, M, 1, D]
        mu_exp = mus.unsqueeze(1)  # [B, 1, K, D]
        sig_exp = sig.unsqueeze(1)  # [B, 1, K, D]

        var = sig_exp * sig_exp + EPS
        sq = ((x_exp - mu_exp) ** 2) / var  # [B, M, K, D]

        # log N(x | mu, diag(sig^2))
        log_comp = -0.5 * sq.sum(dim=-1)  # [B, M, K]
        log_norm = -(sig_exp.log()).sum(dim=-1)  # [B, M, K]

        if not self.drop_gaussian_const:
            const = -0.5 * D * math.log(2.0 * math.pi)
            log_comp = log_comp + const

        log_comp = log_comp + log_norm  # [B, M, K]

        log_pis = torch.log(pis.unsqueeze(1) + EPS)  # [B, 1, K]
        log_weights = log_pis + log_comp  # [B, M, K]
        logp = torch.logsumexp(log_weights, dim=-1)  # [B, M]

        if self.gmm_temp != 1.0:
            logp = logp / self.gmm_temp
        return logp

    # -----------------------------
    # Responsibilities for arbitrary points x (posterior over components)
    # -----------------------------
    def responsibilities(
        self, model_params: Dict[str, torch.Tensor], x: torch.Tensor
    ) -> torch.Tensor:
        """
        r_{b,m,k} = p(k | x_{b,m})
        x: [B, M, D]
        returns r: [B, M, K]
        """
        pis = model_params["pis"]  # [B, K]
        mus = model_params["means"]  # [B, K, D]
        sig = model_params["sigmas"]  # [B, K, D]

        sig = torch.clamp(sig, min=self.min_sigma)
        B, M, D = x.shape

        x_exp = x.unsqueeze(2)  # [B, M, 1, D]
        mu_exp = mus.unsqueeze(1)  # [B, 1, K, D]
        sig_exp = sig.unsqueeze(1)  # [B, 1, K, D]

        var = sig_exp * sig_exp + EPS
        sq = ((x_exp - mu_exp) ** 2) / var
        log_comp = -0.5 * sq.sum(dim=-1)  # [B, M, K]
        log_norm = -(sig_exp.log()).sum(dim=-1)  # [B, M, K]
        if not self.drop_gaussian_const:
            log_comp = log_comp + (-0.5 * D * math.log(2.0 * math.pi))
        log_comp = log_comp + log_norm

        log_weights = torch.log(pis.unsqueeze(1) + EPS) + log_comp
        r = F.softmax(log_weights, dim=-1)
        return r

    # -----------------------------
    # Distribution pooling features (KDE-like) via probes
    # -----------------------------
    def bag_features_probes(
        self, model_params: Dict[str, torch.Tensor], scale_by_dim: bool = True
    ) -> torch.Tensor:
        """
        Returns a fixed-length bag embedding by evaluating log density on M learned probes.
        output: [B, M]
        """
        assert (
            self.probes is not None and self.gmm_M > 0
        ), "Set probes_M>0 to use probe features."
        B = model_params["pis"].shape[0]

        probes = self.probe_norm(self.probes)  # [M, D]
        # if self.add_tanh:
        #     probes = torch.sigmoid(self.probes)
        if self.apply_l2_norm:
            probes = F.normalize(probes, p=2, dim=-1)
        probes = probes.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]

        logp = self.gmm_log_density(model_params, probes)  # [B, M]

        if scale_by_dim:
            logp = logp / (float(self.D))

        # numeric safety
        # logp = (logp - logp.mean(dim=1, keepdim=True)) / (logp.std(dim=1, keepdim=True) + 1e-6)

        logp = torch.nan_to_num(logp, neginf=-1e4, posinf=1e4)
        logp = torch.clamp(logp, -1e4, 1e4)

        # logp = torch.clamp(logp, -100, 1e4)
        # logp = (logp-(-100))/(0-(-100))

        # z = logp / float(self.D)
        # with torch.no_grad():
        #     x = (z.std() / 1.5).clamp(min=1e-3)  # makes (z/x).std ~ 1.5
        # z = z / x

        # Often helpful: normalize by dimension so magnitudes are comparable across setups
        return logp

    # -----------------------------
    # Optional: parameter embedding (sometimes useful alongside probe features)
    # -----------------------------
    def bag_features_params(
        self, model_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Concatenate (pis, means, log(sigmas)) as a vector: [B, K + K*D + K*D]
        """
        pis = model_params["pis"]  # [B, K]
        means = model_params["means"]  # [B, K, D]
        sigmas = model_params["sigmas"]  # [B, K, D]
        feat = torch.cat(
            [pis, means.flatten(1), torch.log(sigmas + EPS).flatten(1)], dim=-1
        )
        return feat


class FeatureExtractor(nn.Module):

    def __init__(self, num_out=32, no_sigmoid=False):
        super(FeatureExtractor, self).__init__()

        self._model_conv = resnet18()

        num_ftrs = self._model_conv.fc.in_features
        self._model_conv.fc = nn.Linear(num_ftrs, num_out)
        # print(self._model_conv)
        if no_sigmoid:
            self.sigmoid = nn.Identity()
        else:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self._model_conv(x)
        out = self.sigmoid(out)

        return out


class RepresentationTransformation(nn.Module):
    def __init__(self, num_in=32, num_out=10):
        super(RepresentationTransformation, self).__init__()

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_in, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_out),
        )

    def forward(self, x):

        out = self.fc(x)

        return out


class Attention(nn.Module):
    def __init__(self, num_in=32, num_instances=32):
        super(Attention, self).__init__()
        self._num_instances = num_instances

        self.fc = nn.Sequential(nn.Linear(num_in, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, x):

        out = self.fc(x)
        out = torch.reshape(out, (-1, self._num_instances, 1))
        out = F.softmax(out, dim=1)

        return out


class Attention2(nn.Module):
    def __init__(self, num_in=32, num_instances=32):
        super(Attention2, self).__init__()
        self._num_instances = num_instances

        self.fc = nn.Sequential(nn.Linear(num_in, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):

        out = self.fc(x)
        out = torch.reshape(out, (-1, self._num_instances, 1))
        out = torch.sigmoid(out)

        return out


class Model(nn.Module):

    def __init__(
        self,
        num_classes=10,
        num_instances=32,
        num_features=32,
        mil_pooling_filter="distribution",
        num_bins=11,
        sigma=0.1,
        M=4,
        K=2,
        T=2,
        no_sigmoid=False,
    ):
        super(Model, self).__init__()
        self._num_classes = num_classes
        self._num_instances = num_instances
        self._num_features = num_features
        self._num_bins = num_bins
        self._sigma = sigma
        self._mil_pooling_filter = mil_pooling_filter

        self.M = M
        self.K = K
        self.T = T
        self.no_sigmoid = no_sigmoid
        self._feature_extractor = FeatureExtractor(
            num_out=num_features, no_sigmoid=self.no_sigmoid
        )
        # distribution pooling
        if mil_pooling_filter == "distribution":
            self._attention = Attention(
                num_in=num_features, num_instances=num_instances
            )
            self._attention2 = Attention2(
                num_in=num_features, num_instances=num_instances
            )
            self._distribution_pooling_filter = DistributionPoolingFilter(
                num_bins=num_bins, sigma=sigma
            )
            self._representation_transformation = RepresentationTransformation(
                num_in=num_features * num_bins, num_out=num_classes
            )

        # attention pooling
        elif mil_pooling_filter == "attention":
            self._attention = Attention(
                num_in=num_features, num_instances=num_instances
            )
            self._representation_transformation = RepresentationTransformation(
                num_in=num_features, num_out=num_classes
            )

        # max and mean pooling
        elif mil_pooling_filter == "gmm":
            # M20 K 10 T 10
            self._gmm = GMM(
                D=num_features,
                gmm_K=self.K,
                gmm_M=self.M,
                gmm_temp=self.T,
            )

            self._representation_transformation = nn.Sequential(
                nn.Linear(self.M, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes),
            )

        else:
            self._representation_transformation = RepresentationTransformation(
                num_in=num_features, num_out=num_classes
            )

        # initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, return_gmm_bag_embedding=False):

        extracted_features = self._feature_extractor(x)

        if self._mil_pooling_filter == "distribution":
            attention_values = self._attention(extracted_features)
            attention_values2 = self._attention2(extracted_features)
            extracted_features = torch.reshape(
                extracted_features, (-1, self._num_instances, self._num_features)
            )
            out = attention_values2 * extracted_features
            out = self._distribution_pooling_filter(out, attention_values)
            out = torch.reshape(out, (-1, self._num_features * self._num_bins))

        elif self._mil_pooling_filter == "attention":
            attention_values = self._attention(extracted_features)
            extracted_features = torch.reshape(
                extracted_features, (-1, self._num_instances, self._num_features)
            )
            out = torch.matmul(
                torch.transpose(attention_values, 2, 1), extracted_features
            )
            out = torch.squeeze(out, dim=1)

        elif self._mil_pooling_filter == "mean":
            extracted_features = torch.reshape(
                extracted_features, (-1, self._num_instances, self._num_features)
            )
            out = torch.mean(extracted_features, dim=1, keepdim=False)

        elif self._mil_pooling_filter == "max":
            extracted_features = torch.reshape(
                extracted_features, (-1, self._num_instances, self._num_features)
            )
            out = torch.max(extracted_features, dim=1)[0]

        elif self._mil_pooling_filter == "gmm":
            z = extracted_features.view(
                -1, self._num_instances, self._num_features
            )  # [B,N,D]
            gmm_params = self._gmm(z)
            logp = self._gmm.bag_features_probes(gmm_params)  # [B, M]
            pis = gmm_params["pis"]  # [B, K]
            mu = gmm_params["means"].reshape(z.size(0), -1)  # [B, K*D]
            logs = torch.log(gmm_params["sigmas"] + 1e-8).reshape(
                z.size(0), -1
            )  # [B, K*D]

            # out = torch.cat([logp, pis, mu, logs], dim=1)  # [B, M + K + 2*K*D]
            out = torch.cat([logp], dim=1)  # [B, M + K + 2*K*D]

            # print("logp")
            # print(out.std(dim=0).mean())
            # print("means")
            # print(gmm_params["pis"].mean(0))
            # print("std")
            # print(gmm_params["sigmas"].mean())
            bag_embeddings = {
                "probes": logp.detach().cpu(),
                "pis": gmm_params["pis"].detach().cpu(),
                "means": gmm_params["means"].detach().cpu(),
                "sigmas": gmm_params["sigmas"].detach().cpu(),
                "features": gmm_params["f"].detach().cpu(),
            }
            # only compute stats when requested
            if return_gmm_bag_embedding:
                gmm_stats = {
                    "logp_std_mean": logp.std(dim=0).mean().detach(),
                    "pis_mean": gmm_params["pis"].mean(0).detach(),  # [K]
                    "sigmas_mean": gmm_params["sigmas"].mean().detach(),  # scalar
                    "sigmas_min": gmm_params["sigmas"].min().detach(),
                    "sigmas_max": gmm_params["sigmas"].max().detach(),
                }
        out = self._representation_transformation(out)
        if return_gmm_bag_embedding:
            # return out, bag_embeddings
            return out, bag_embeddings

        return out
