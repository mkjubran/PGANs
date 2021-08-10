import TruncatedNormal2 as TNorm
import torch
import numpy as np
import math
from scipy.stats import truncnorm

def parameterized_truncated_normal(uniform, mu, sigma, a, b):
    normal = torch.distributions.normal.Normal(0, 1)

    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    alpha_normal_cdf = normal.cdf(alpha)
    p = alpha_normal_cdf + (normal.cdf(beta) - alpha_normal_cdf) * uniform

    p = p.numpy()
    one = np.array(1, dtype=p.dtype)
    epsilon = np.array(np.finfo(p.dtype).eps, dtype=p.dtype)
    v = np.clip(2 * p - 1, -one + epsilon, one - epsilon)
    x = mu + sigma * np.sqrt(2) * torch.erfinv(torch.from_numpy(v))
    x = torch.clamp(x, a, b)

    return x


def truncated_normal(uniform):
    return parameterized_truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2)


loc = torch.tensor([0., 0.]) #.cuda()
scale = torch.tensor([1., 1.]) #.cuda()
a = torch.tensor([-10000.]) #.cuda()
b = torch.tensor([10000.]) #.cuda()
pt = TNorm.TruncatedNormal(loc, scale, a, b)

x = torch.tensor([0., 0.]) #.cuda()
log_prob_pt = pt.log_prob(x)
print(log_prob_pt)
print(torch.exp(log_prob_pt))
print((1/(math.sqrt(2*np.pi))))

big_phi_a = 0.5 * (1 + (a  / math.sqrt(2)).erf())
big_phi_b = 0.5 * (1 + (b  / math.sqrt(2)).erf())

print(big_phi_a)
print(big_phi_b)
eps = torch.finfo(a.dtype).eps

_Z = (big_phi_b - big_phi_a).clamp_min(eps)
print(_Z)

_log_Z = _Z.log()
print(_log_Z)

#print(truncnorm.pdf(x=0, a=-10000, b=10000, loc=0, scale=1))
print(truncnorm.pdf(x, a, b, loc, scale))

a, b = -2, 2
size = 1000000
r = truncated_normal_(x, a, b, mean=loc, std=scale)
print(r)
