import torch
import torch.nn as nn



def _log_sum_exp(X, dim, eps=1e-20):
    X = torch.log(torch.exp(X).sum(dim=dim)+eps)
    return X

def sinkhorn(log_alpha, n_iters=20):
    """Performs incomplete Sinkhorn normalization to log_alpha.
    By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix
    with positive entries can be turned into a doubly-stochastic matrix
    (i.e. its rows and columns add up to one) via the succesive row and column
    normalization.
    -To ensure positivity, the effective input to sinkhorn has to be
    exp(log_alpha) (elementwise).
    -However, for stability, sinkhorn works in the log-space. It is only at
    return time that entries are exponentiated.
    [1] Sinkhorn, Richard and Knopp, Paul.
    Concerning nonnegative matrices and doubly stochastic
    matrices. Pacific Journal of Mathematics, 1967
    Args:
        log_alpha: 2D tensor (a matrix of shape [N, N])
        or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
        n_iters: number of sinkhorn iterations (in practice, as little as 20
        iterations are needed to achieve decent convergence for N~100)
    Returns:
        A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are
        converted to 3D tensors with batch_size equals to 1)
    """
    # if len(log_alpha.shape)==2:
    #     n = log_alpha.shape[0]
    #     m = log_alpha.shape[1]
    #     log_alpha = log_alpha.reshape(-1, n, m)
    n = log_alpha.shape[1]
    log_alpha = log_alpha.reshape(-1, n, n)
    for i in range(n_iters):
        # log_alpha -= _log_sum_exp(log_alpha, dim=2)[:, :, None]
        log_alpha -= _log_sum_exp(log_alpha, dim=1)[:, None, :]
        log_alpha -= _log_sum_exp(log_alpha, dim=2)[:, :, None]
    return torch.exp(log_alpha) # become alpha


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)


    norm = - (ms + ns).log()
    log_mu = norm.expand(m)
    log_nu = norm.expand(n)
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(scores, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z