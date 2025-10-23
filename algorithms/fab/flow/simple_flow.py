import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "float32")

import flax.linen as nn
import optax
import distrax
from flax import struct
from flax.training import train_state
from typing import Callable, Dict, Tuple
from functools import partial
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import field
def create_prior(input_dims: int):
    loc = jnp.zeros(input_dims, dtype=jnp.float32)
    cov = jnp.eye(input_dims, dtype=jnp.float32)
    return distrax.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=cov)

def count_parameters(params):
    return sum(jnp.prod(jnp.array(p.shape)) for p in jax.tree_util.tree_leaves(params))
class FeedForwardNetwork(struct.PyTreeNode):
    """Container for init/apply functions of a feed-forward network."""
    init: Callable[[jax.random.PRNGKey, int, jnp.dtype], Dict]
    apply: Callable[[Dict, str, jnp.ndarray, jax.random.PRNGKey, int], Tuple[jnp.ndarray, jnp.ndarray]]
import jax
import jax.numpy as jnp
import flax.linen as nn

def _squeeze_leading_one(a):
    # If params were replicated with axis size 1, drop that axis.
    return a[0] if (a.ndim >= 2 and a.shape[0] == 1) else a

class InvertiblePLU(nn.Module):
    features: int
    key: jax.Array = field(default_factory=lambda: jax.random.PRNGKey(0))
    
    def setup(self):
        d = self.features
        key = self.key
        
        w_shape = (d, d)
        w_init = nn.initializers.orthogonal()(key, w_shape)
        P, L, U = jax.scipy.linalg.lu(w_init)
        s = jnp.diag(U)
        U = U - jnp.diag(s)

        self.P = P
        self.P_inv = jax.scipy.linalg.inv(P)
        
        self.L_init = jnp.tril(L, k=-1)
        self.U_init = jnp.triu(U, k=1)
        self.s_init = s

    @nn.compact
    def __call__(self, x, reverse: bool = False):
        d = self.features

        L_free = self.param("L", lambda rng: self.L_init)
        U_free = self.param("U", lambda rng: self.U_init)
        
        L = jnp.tril(L_free, k=-1) + jnp.eye(d)
        U = jnp.triu(U_free, k=1)
        s = self.param("s", lambda rng: self.s_init)
        
        W = self.P @ L @ (U + jnp.diag(s))

        if not reverse:
            z = jnp.dot(x, W)
            logdet = jnp.sum(jnp.log(jnp.abs(s)))
            return z , jnp.expand_dims( logdet, 0)
        
        else:
            
            U_inv = jax.scipy.linalg.solve_triangular(U + jnp.diag(s), jnp.eye(self.features), lower=False)
            L_inv = jax.scipy.linalg.solve_triangular(L, jnp.eye(self.features), lower=True, unit_diagonal=True)
            
            W_inv = U_inv @ L_inv @ self.P_inv
            
            z = jnp.dot(x, W_inv)
            logdet = jnp.sum(jnp.log(jnp.abs(s)))
            return z, -jnp.expand_dims( logdet, 0)

class MetaBlock(nn.Module):
    in_channels: int
    channels: int
    block_idx: int

    def setup(self):
        def kernel_init(key, shape, dtype=jnp.float32):
            in_features = shape[0]  # Dense weight shape = (in, out)
            k = jnp.sqrt(1.0 / in_features)
            return jax.random.uniform(key, shape, dtype, minval=-k, maxval=k)

        def bias_init(key, shape, dtype, in_features):
            k = jnp.sqrt(1.0 / in_features)
            return jax.random.uniform(key, shape, dtype, minval=-k, maxval=k)

        self.l = InvertiblePLU(features=self.in_channels)

        in_half = (self.in_channels) // 2 
        # t-network: inputs are x_cond only (no y)
        self.t = nn.Sequential([
            nn.Dense(self.channels, kernel_init=kernel_init,
                     bias_init=partial(bias_init, in_features=in_half)),
            nn.leaky_relu,
            nn.LayerNorm(),
            nn.Dense(self.channels, kernel_init=kernel_init,
                     bias_init=partial(bias_init, in_features=self.channels)),
            nn.leaky_relu,
            nn.LayerNorm(),
            nn.Dense(in_half, kernel_init=nn.initializers.zeros)
        ])

        # s-network: same shape as t-network
        self.s = nn.Sequential([
            nn.Dense(self.channels, kernel_init=kernel_init,
                     bias_init=partial(bias_init, in_features=in_half)),
            nn.leaky_relu,
            nn.LayerNorm(),
            nn.Dense(self.channels, kernel_init=kernel_init,
                     bias_init=partial(bias_init, in_features=self.channels)),
            nn.leaky_relu,
            nn.LayerNorm(),
            nn.Dense(in_half, kernel_init=nn.initializers.zeros)
        ])

    def __call__(self, x, reverse: bool = False):
        return self.reverse(x) if reverse else self.forward(x)

    def forward(self, x):
        assert len(x.shape) == 2
        x, log_det = self.l(x)  # (B, Dx), (B,)
        x_cond, x_trans = jnp.array_split(x, 2, axis=1)
        s = self.s(x_cond)
        t = self.t(x_cond)
        x_trans = (x_trans - t) * jnp.exp(-s)
        x = jnp.concatenate((x_cond, x_trans), axis=1)
        log_det = log_det - jnp.sum(s, axis=1)  # (B,)
        
        return x, log_det

    def reverse(self, z):
        assert len(z.shape) == 2
        z_cond, z_trans = jnp.array_split(z, 2, axis=1)
        s = self.s(z_cond)
        t = self.t(z_cond)
        z_trans = z_trans * jnp.exp(s) + t
        z = jnp.concatenate((z_cond, z_trans), axis=1)
        z, log_det = self.l(z, reverse=True)  # (B, Dx), (B,)
        log_det = log_det + jnp.sum(s, axis=1)  # (B,)
        return z, log_det

#--bijections--

class BoxSigmoid(distrax.Bijector):
    """Squashing bijector: R^d  <->  (low, high)^d."""
    def __init__(self, low: jnp.ndarray, high: jnp.ndarray, eps: float = 1e-6):
        # `event_ndims_in=1`  says that log-det sums over the last axis (the event dim)
        super().__init__(event_ndims_in=1)
        self.low  = jnp.asarray(low)
        self.high = jnp.asarray(high)
        self.scale = self.high - self.low          # element-wise (d,)
        self.eps = eps                             # clip to avoid log(0)

    # ---------- forward: y -> x ------------------------------------------------
    def forward_and_log_det(self, y: jnp.ndarray):
        s   = jax.nn.sigmoid(y)                    # (b,d) in (0,1)
        x   = self.low + self.scale * s
        # log|dx/dy| = sum_i [ log(scale_i) + log σ(y_i) + log(1-σ(y_i)) ]
        log_det = jnp.sum(
            jnp.log(self.scale) +
            jax.nn.log_sigmoid(y) + jax.nn.log_sigmoid(-y),
            axis=-1
        )
        return x, log_det

    # ---------- inverse: x -> y ------------------------------------------------
    def inverse_and_log_det(self, x: jnp.ndarray):
        u = (x - self.low) / self.scale            # (0,1)
        u = jnp.clip(u, self.eps, 1.0 - self.eps)  # guard logit
        y = jnp.log(u) - jnp.log1p(-u)             # logit
        # log|dy/dx| = −log|dx/dy|
        log_det = -jnp.sum(
            jnp.log(self.scale) +
            jnp.log(u) + jnp.log1p(-u),
            axis=-1
        )
        return y, log_det
class RealNVP(nn.Module):
    num_blocks: int
    in_channels: int
    channels: int

    def setup(self):
        self.blocks = [
            MetaBlock(
                in_channels=self.in_channels,
                channels=self.channels,
                block_idx=i
            ) for i in range(self.num_blocks)
        ]
    def __call__(self, x, reverse: bool = False):
        return self.reverse(x) if reverse else self.forward(x)

    def forward(self, x):
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, 0)
        log_dets = jnp.zeros(x.shape[0], dtype=x.dtype)
        for block in self.blocks:
            x, log_det = block(x)
            log_dets = log_dets + log_det
        return x.squeeze(), log_dets.squeeze()

    def reverse(self, x):
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, 0)
        log_dets = jnp.zeros(x.shape[0], dtype=x.dtype)
        for block in reversed(self.blocks):
            x, log_det = block(x, reverse=True)
            log_dets = log_dets + log_det
        return x.squeeze(), log_dets.squeeze()

# --- builder ---


def make_realnvp_flow_networks(
    num_blocks: int = 6,
    in_channels: int =14, # params dim,
    channels: int = 256,
) -> FeedForwardNetwork:
    """Build init/apply APIs for your RealNVP flow p(x|y).

    Returns:
      FlowAPIs with:
        - init_fn(rng, batch_size=1, dtype=jnp.float32) -> params

    """
    flow = RealNVP(
        num_blocks=num_blocks,
        in_channels=in_channels,
        channels=channels,
    )

    def init_fn(rng: jax.random.PRNGKey,
                batch_size: int = 1,
                dtype=jnp.float32):
        x0 = jnp.zeros((batch_size, in_channels), dtype)
        variables = flow.init(rng, x0, reverse=False)
        return variables["params"]
    
    def apply_fn(params, mode: str,
               low: jnp.ndarray, high: jnp.ndarray,
                x: jnp.ndarray = None,
                rng: jax.random.PRNGKey = None,
                n_samples: int = None,
    ):
        prior = create_prior(in_channels)
        box_bij = BoxSigmoid(low=low, high=high)
        if mode == "log_prob":
            if x is None:
                raise ValueError("mode='log_prob' requires x")
            y, logdet_box = box_bij.inverse_and_log_det(x)
            z, logdet = flow.apply({'params':params}, y, reverse=False)
            return prior.log_prob(z) + logdet + logdet_box
        elif mode == "sample":
            if rng is None or n_samples is None:
                raise ValueError("mode='sample' requires rng and n_samples")
            z = prior.sample(seed=rng, sample_shape=(n_samples,))

            y, logdet = flow.apply({"params":params}, z, reverse=True)
            x, logdet_box = box_bij.forward_and_log_det(y)
            logp = prior.log_prob(z) - logdet - logdet_box
            return x, logp
        else:
            raise ValueError(f"Unknown mode: {mode}")

    
    return FeedForwardNetwork(
        init=init_fn,
        apply=apply_fn,
    )


import matplotlib as _mpl
try:
    _mpl.use("Agg")  # fast, non-interactive backend
except Exception:
    pass
_mpl.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 100,
    "axes.grid": False,
    "lines.antialiased": False,
    "path.simplify": True,
    "path.simplify_threshold": 1.0,
    "agg.path.chunksize": 20000,
})

import functools
import math
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

def render_flow_pdf_1d_subplots(
    flow_network, params, ndim, low, high,
    rng=None, resolution=512, other_samples=512,
    normalize=True, ncols=4,
    suptitle="Flow PDF — 1D marginals via MC",
    use_wandb=None, training_step=0, 
):
    """
    Faster version:
      - heavy compute is JIT-compiled per-dim (first call compiles once per dim)
      - avoids tight_layout; cheaper layout call
      - converts only small arrays host-side
    """
    # ---- prepare constants on device
    low  = jnp.asarray(low,  jnp.float32)
    high = jnp.asarray(high, jnp.float32)
    if rng is None:
        rng = jax.random.PRNGKey(0)
    rng, sub = jax.random.split(rng)

    # JIT-friendly log_prob that avoids dynamic 'mode' branching in the hot path
    def _log_prob(points):
        return flow_network.apply(
            params, mode="log_prob", x=points, low=low, high=high
        )

    # Monte-Carlo contexts for the other dims: (M, D)
    U = jax.random.uniform(sub, (other_samples, ndim), dtype=jnp.float32)
    others = low + (high - low) * U  # device-resident

    # JIT: compute one 1D marginal for a fixed d
    @functools.lru_cache(maxsize=None)
    def _compiled_for_dim(d_int, res_int):
        """Return a jitted function specialized for (d, resolution)."""
        d = int(d_int)
        R = int(res_int)

        @jax.jit  # compiles once per (d, R)
        def _compute(others_in):
            xs = jnp.linspace(low[d], high[d], R, dtype=jnp.float32)  # (R,)
            # vmap over contexts -> (M, R) logp slices along dim d
            def eval_for_context(ctx):
                X = jnp.broadcast_to(ctx, (R, ndim))
                X = X.at[:, d].set(xs)
                return _log_prob(X)  # (R,)

            lps = jax.vmap(eval_for_context)(others_in)      # (M, R)
            logp = jax.scipy.special.logsumexp(lps, axis=0)  # (R,)
            logp = logp - jnp.log(lps.shape[0])
            # stabilize + exp
            logp_shift = logp - jnp.max(logp)
            pdf = jnp.exp(logp_shift)                        # (R,)

            if normalize:
                dx = (high[d] - low[d]) / (R - 1)
                mass = jnp.sum(pdf) * dx
                pdf = jnp.where(mass > 0, pdf / mass, pdf)

            return xs, pdf

        return _compute

    # ---- plotting (cheap path)
    ncols = max(1, ncols)
    nrows = math.ceil(ndim / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4*ncols, 3*nrows), dpi=100)
    axes = axes.flatten().tolist() if isinstance(axes, np.ndarray) else [axes]

    for d in range(ndim):
        ax = axes[d]
        # compute marginal (device), then transfer only tiny arrays
        compute_d = _compiled_for_dim(d, resolution)
        xs_d, pdf_d = compute_d(others)
        xs_d = np.asarray(xs_d)   # triggers device sync just for small arrays
        pdf_d = np.asarray(pdf_d)

        ax.plot(xs_d, pdf_d, lw=1)
        ax.set_title(f"dim {d}", pad=2)
        ax.set_xlim(float(low[d]), float(high[d]))
        ax.set_xlabel(f"x[{d}]")
        ax.set_ylabel("p(x)" + (" (norm.)" if normalize else ""))

    # remove unused axes without tight_layout (faster)
    for k in range(ndim, nrows * ncols):
        fig.delaxes(axes[k])

    if suptitle:
        # cheaper than tight_layout + suptitle combo
        fig.suptitle(suptitle, y=0.99, fontsize=10)
    # cheaper layout than tight_layout for large grids
    fig.subplots_adjust(wspace=0.25, hspace=0.35, top=0.93)

    if use_wandb is not None:
        try:
            import wandb
            wandb.log(
                {f"all_dims_1d_pdf(step : {training_step})": wandb.Image(fig)},
                step=int(training_step),
            )
        except Exception as e:
            print(f"[render_flow_pdf_1d_subplots] W&B log failed: {e}")

    return fig, axes
import functools, math, numpy as np
import jax, jax.numpy as jnp
import matplotlib.pyplot as plt

def render_flow_pdf_2d_subplots(
    flow_network, params, ndim, low, high,
    pairs=None,                      # list of (i,j); defaults to all i<j
    rng=None, resolution=256, other_samples=256,
    normalize=True, ncols=3,
    suptitle="Flow PDF — 2D marginals via MC",
    use_wandb=None, training_step=0,
    add_colorbar=False,
):
    """
    Estimates 2D marginals p(x[i], x[j]) by Monte-Carlo integration over the other dims.
      - JIT compiled per (pair, resolution)
      - Works in log-space; robust normalization
      - Avoids tight_layout for speed
    Cost ~ O(other_samples * resolution^2 * num_pairs)
    """
    # ---- constants on device
    low  = jnp.asarray(low,  jnp.float32)
    high = jnp.asarray(high, jnp.float32)
    if rng is None:
        rng = jax.random.PRNGKey(0)
    rng, sub = jax.random.split(rng)

    def _log_prob(points):
        return flow_network.apply(params, mode="log_prob", x=points, low=low, high=high)

    # Monte-Carlo contexts for remaining dims
    U = jax.random.uniform(sub, (other_samples, ndim), dtype=jnp.float32)
    others = low + (high - low) * U  # (M, D)

    # default to all unique pairs i<j
    if pairs is None:
        pairs = [(i, j) for i in range(ndim) for j in range(i+1, ndim)]
    nplots = len(pairs)

    # JIT per (i, j, res)
    @functools.lru_cache(maxsize=None)
    def _compiled_for_pair(i_int, j_int, res_int):
        i, j = int(i_int), int(j_int)
        R = int(res_int)

        @jax.jit
        def _compute(others_in):
            xi = jnp.linspace(low[i], high[i], R, dtype=jnp.float32)  # (R,)
            xj = jnp.linspace(low[j], high[j], R, dtype=jnp.float32)  # (R,)
            XX, YY = jnp.meshgrid(xi, xj, indexing="xy")               # (R, R)

            grid_flat_i = XX.reshape(-1)  # (R*R,)
            grid_flat_j = YY.reshape(-1)  # (R*R,)

            def eval_for_context(ctx):
                # (R*R, D) all set to ctx, then overwrite dims i,j with grid
                X = jnp.broadcast_to(ctx, (R*R, ndim))
                X = X.at[:, i].set(grid_flat_i)
                X = X.at[:, j].set(grid_flat_j)
                return _log_prob(X).reshape(R, R)  # (R, R)

            lps = jax.vmap(eval_for_context)(others_in)  # (M, R, R)
            logp = jax.scipy.special.logsumexp(lps, axis=0) - jnp.log(lps.shape[0])  # (R,R)

            # stabilize + exp
            logp_shift = logp - jnp.max(logp)
            pdf = jnp.exp(logp_shift)  # unnormalized marginal on the box

            if normalize:
                dx = (high[i] - low[i]) / (R - 1)
                dy = (high[j] - low[j]) / (R - 1)
                mass = jnp.sum(pdf) * dx * dy
                pdf = jnp.where(mass > 0, pdf / mass, pdf)

            return xi, xj, pdf

        return _compute

    # ---- plotting
    ncols = max(1, ncols)
    nrows = math.ceil(nplots / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4.6*ncols, 4.0*nrows), dpi=100)
    axes = axes.flatten().tolist() if isinstance(axes, np.ndarray) else [axes]

    for k, (i, j) in enumerate(pairs):
        ax = axes[k]
        compute_ij = _compiled_for_pair(i, j, resolution)
        xi, xj, pdf = compute_ij(others)

        xi = np.asarray(xi); xj = np.asarray(xj); pdf = np.asarray(pdf)
        im = ax.pcolormesh(xi, xj, pdf, shading="auto")
        ax.set_title(f"dim ({i},{j})", pad=2)
        ax.set_xlabel(f"x[{i}]"); ax.set_ylabel(f"x[{j}]")
        if add_colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # remove unused axes
    for k in range(nplots, nrows * ncols):
        fig.delaxes(axes[k])

    if suptitle:
        fig.suptitle(suptitle, y=0.99, fontsize=10)
    fig.subplots_adjust(wspace=0.28, hspace=0.32, top=0.93)

    if use_wandb is not None:
        try:
            import wandb
            wandb.log(
                {f"all_pairs_2d_pdf(step : {training_step})": wandb.Image(fig)},
                step=int(training_step),
            )
        except Exception as e:
            print(f"[render_flow_pdf_2d_subplots] W&B log failed: {e}")

    return fig, axes
