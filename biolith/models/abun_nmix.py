import unittest
from typing import Optional, Type

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from biolith.regression import AbstractRegression, LinearRegression
from biolith.utils.spatial import sample_spatial_effects, simulate_spatial_effects
from biolith.utils.distributions import RightTruncatedPoisson


def n_mixture_abundance(
    site_covs: jnp.ndarray,
    obs_covs: jnp.ndarray,
    # coords: Optional[jnp.ndarray] = None,
    # ell: float = 1.0,
    # false_positives_constant: bool = False,
    # false_positives_unoccupied: bool = False,
    obs: Optional[jnp.ndarray] = None,
    prior_beta: dist.Distribution = dist.Normal(),
    prior_alpha: dist.Distribution = dist.Normal(),
    regressor_occ: Type[AbstractRegression] = LinearRegression,
    regressor_det: Type[AbstractRegression] = LinearRegression,
    # prior_prob_fp_constant: dist.Distribution = dist.Beta(2, 5),
    # prior_prob_fp_unoccupied: dist.Distribution = dist.Beta(2, 5),
    # prior_gp_sd: dist.Distribution = dist.HalfNormal(1.0),
    # prior_gp_length: dist.Distribution = dist.HalfNormal(1.0),
    poisson_truncation: int = 100,
) -> None:
    """
    Model animal abundance from unmarked count data (aka N-mixture model):

    State process:
    -----------
    N ~ Poisson(lambda)
    log(lambda) = beta_0 + beta_1 * site_covs_1 + ... + beta_n * site_covs_n
    where beta are the site-level regression coefficients.

    Detection process: each individual at a site has a probability of detection p
    p can be modeled with a regression on observation-level covariates:
    y ~ Binomial(N, p)
    logit(p) = alpha_0 + alpha_1 * obs_covs_1 + ... + alpha_n * obs_covs_n
    where alpha are the observation-level regression coefficients.

    References
    ----------
        - Royle, J. Andrew. "N-mixture models for estimating population size from spatially replicated counts." Biometrics 60.1 (2004): 108-115.

    Parameters
    ----------
    site_covs : jnp.ndarray
        An array of site-level covariates of shape (n_sites, n_site_covs).
    obs_covs : jnp.ndarray
        An array of observation-level covariates of shape (n_sites, n_revisits, n_obs_covs).
    obs : jnp.ndarray, optional
        Observation matrix of shape (n_sites, n_revisits) or None.
    prior_beta : numpyro.distributions.Distribution
        Prior distribution for the site-level regression coefficients.
    prior_alpha : numpyro.distributions.Distribution
        Prior distribution for the observation-level regression coefficients.
    regressor_occ : Type[AbstractRegression]
        Class for the occupancy regression model, defaults to LinearRegression.
    regressor_det : Type[AbstractRegression]
        Class for the detection regression model, defaults to LinearRegression.


    Examples
    --------
    >>> from biolith.models  abun_nmix
    >>> from biolith.utils import fit
    >>> data, _ = abun_nmix.simulate()
    >>> results = fit(abun_nmix.n_mixture_abundance, **data)
    >>> print(results.samples['lambda_'].mean())
    """

    # Check input data
    assert (
        obs is None or obs.ndim == 2
    ), "obs must be None or of shape (n_sites, time_periods)"
    assert site_covs.ndim == 2, "site_covs must be of shape (n_sites, n_site_covs)"
    assert (
        obs_covs.ndim == 3
    ), "obs_covs must be of shape (n_sites, time_periods, n_obs_covs)"
    # assert obs is None or (obs[np.isfinite(obs)] >= 0).all(), "observations must be non-negative"  # TODO: re-enable
    # assert obs is None or (obs[np.isfinite(obs)] <= 1).all(), "observations must be binary"  # TODO: re-enable

    n_sites = site_covs.shape[0]
    time_periods = obs_covs.shape[1]
    n_site_covs = site_covs.shape[1]
    n_obs_covs = obs_covs.shape[2]

    assert (
        n_sites == site_covs.shape[0] == obs_covs.shape[0]
    ), "site_covs and obs_covs must have the same number of sites"
    assert (
        time_periods == obs_covs.shape[1]
    ), "obs_covs must have the same number of time periods as obs"
    if obs is not None:
        assert n_sites == obs.shape[0], "obs must have n_sites rows"
        assert time_periods == obs.shape[1], "obs must have time_periods columns"

    # Mask observations where covariates are missing
    obs_mask = jnp.isnan(obs_covs).any(axis=-1) | jnp.tile(
        jnp.isnan(site_covs).any(axis=-1)[:, None], (1, time_periods)
    )
    obs = jnp.where(obs_mask, jnp.nan, obs) if obs is not None else None
    obs_covs = jnp.nan_to_num(obs_covs)
    site_covs = jnp.nan_to_num(site_covs)

    # Occupancy and detection regression models
    reg_occ = regressor_occ("beta", n_site_covs, prior=prior_beta)
    reg_det = regressor_det("alpha", n_obs_covs, prior=prior_alpha)

    # no spatial component implemented yet
    # w = jnp.zeros(n_sites)

    # Transpose in order to fit NumPyro's plate structure
    site_covs = site_covs.transpose((1, 0))
    obs_covs = obs_covs.transpose((2, 1, 0))
    obs = obs.transpose((1, 0)) if obs is not None else None

    with numpyro.plate("site", n_sites, dim=-1):

        # Number of individuals is a latent Poisson random variable
        # with a log-linear regression on site-level covariates.
        # lambda: expected Poisson rate of individuals at each site given site covariates
        lambda_ = numpyro.deterministic("lambda_", jnp.exp(reg_occ(site_covs)))  # + w

        # N: latent number of individuals at each site, right-truncated Poisson
        N = numpyro.sample(
            "N", RightTruncatedPoisson(rate=lambda_, max_cutoff=poisson_truncation)
        )

        with numpyro.plate("time_periods", time_periods, dim=-2):

            # Detection process
            prob_detection = numpyro.deterministic(
                f"prob_detection",
                jax.nn.sigmoid(reg_det(obs_covs)),
            )

            if obs is not None:
                with numpyro.handlers.mask(mask=jnp.isfinite(obs)):
                    # observed individuals is a binomial draw with probability p and N trials
                    numpyro.sample(
                        "y",
                        dist.Binomial(total_count=N, probs=prob_detection),
                        obs=jnp.nan_to_num(obs),
                    )
            else:  # what is happening here when no observations are provided?
                numpyro.sample(
                    f"y",
                    dist.Binomial(
                        total_count=N, probs=jax.nn.sigmoid(reg_det(obs_covs))
                    ),
                    infer={"enumerate": "parallel"},
                )


def simulate(
    n_site_covs: int = 1,
    n_obs_covs: int = 1,
    n_sites: int = 100,
    n_visits: int = 5,
    simulate_missing: bool = False,
    random_seed: int = 0,
    # spatial: bool = False,
    # gp_sd: float = 1.0,
    # gp_l: float = 0.2,
) -> tuple[dict, dict]:
    """Generate a synthetic dataset for the :func:`occu` model.

    Returns ``(data, true_params)`` suitable for :func:`fit`.

    Examples
    --------
    >>> from biolith.models import abun_nmix
    >>> data, params = abun_nmix.simulate()
    >>> list(data.keys())
    ['site_covs', 'obs_covs', 'obs', 'coords', 'ell']
    """

    # Initialize random number generator
    rng = np.random.default_rng(random_seed)
    # if spatial:
    #     coords = rng.uniform(0, 1, size=(n_sites, 2))
    # else:
    # coords = None

    # # Make sure detection not too close to 0 or 1
    # z = None
    # while (
    #     z is None
    #     # or z.mean() < min_occupancy
    #     # or z.mean() > max_occupancy
    #     or np.mean(obs[np.isfinite(obs)]) < min_observation_rate
    #     or np.mean(obs[np.isfinite(obs)]) > max_observation_rate
    # ):

    # Generate intercept and slopes
    beta = rng.normal(
        size=n_site_covs + 1
    )  # intercept and slopes for occupancy logistic regression
    alpha = rng.normal(
        size=n_obs_covs + 1
    )  # intercept and slopes for detection logistic regression

    # Generate occupancy and site-level covariates
    site_covs = rng.normal(size=(n_sites, n_site_covs))
    # if spatial:
    #     w, ell = simulate_spatial_effects(coords, gp_sd=gp_sd, gp_l=gp_l, rng=rng)
    # else:
    # w, ell = np.zeros(n_sites), 0.0
    log_lambda = (
        beta[0].repeat(n_sites)
        + np.sum(
            [beta[i + 1] * site_covs[..., i] for i in range(n_site_covs)],
            axis=0,
        )
        # + w
    )
    lambda_ = jnp.exp(log_lambda)
    # simulate abundance (N) from a Poisson
    N = rng.poisson(lambda_)

    # N = rng.binomial(
    #     n=1, p=psi, size=n_sites
    # )  # vector of latent occupancy status for each site

    # Create matrix of detection covariates
    obs_covs = rng.normal(size=(n_sites, n_visits, n_obs_covs))
    logit_prob_detection = alpha[0].repeat(n_sites)[:, None] + np.sum(
        [alpha[i + 1] * obs_covs[..., i] for i in range(n_obs_covs)],
        axis=0,
    )
    prob_detection = jax.nn.sigmoid(logit_prob_detection)

    # Create matrix of detections
    y = rng.binomial(n=N[:, None], p=prob_detection)

    if simulate_missing:
        # Simulate missing data:
        y[rng.choice([True, False], size=y.shape, p=[0.2, 0.8])] = np.nan
        obs_covs[rng.choice([True, False], size=obs_covs.shape, p=[0.05, 0.95])] = (
            np.nan
        )
        site_covs[rng.choice([True, False], size=site_covs.shape, p=[0.05, 0.95])] = (
            np.nan
        )

    print(f"True abundance: {np.mean(N):.4f}")
    print(f"Proportion of timesteps with observation: {np.mean(y[np.isfinite(y)]):.4f}")

    return dict(
        site_covs=site_covs,
        obs_covs=obs_covs,
        obs=y,
        # coords=coords,
        # ell=ell,
    ), dict(
        # lambda_=lambda_,
        beta=beta,
        alpha=alpha,
        # w=w,
        # gp_sd=gp_sd,
        # gp_l=gp_l,
    )
