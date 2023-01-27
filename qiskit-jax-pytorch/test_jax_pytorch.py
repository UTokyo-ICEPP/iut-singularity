#!/usr/bin/env python

import timeit

import jax
import jax.numpy as jnp

@jax.jit
def selu(x, alpha=1.67, lmbda=1.05):
    return jnp.mean(lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha))

dselu = jax.jit(jax.grad(lambda x: selu(x)))

key = jax.random.PRNGKey(0)

def run_selu():
    global key
    
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (1000000,))
    selu(x).block_until_ready()

def run_dselu():
    global key

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (1000000,))
    dselu(x).block_until_ready()

run_selu()
run_dselu()

number = 100

t_selu = timeit.timeit(run_selu, number=number)
print(f'{t_selu / number * 1.e+6} us to run selu')

t_dselu = timeit.timeit(run_dselu, number=number)
print(f'{t_dselu / number * 1.e+6} us to run dselu')

import torch

gpu = torch.device('cuda:0')

class MeanSELU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.selu = torch.nn.SELU()

    def forward(self, x):
        return torch.mean(self.selu(x))

selu = MeanSELU().to(gpu)

def run_selu():
    x = torch.randn(1000000, requires_grad=False, device=gpu)
    z = selu(x)
    return z

def run_dselu():
    x = torch.randn(1000000, requires_grad=True, device=gpu)
    z = selu(x)
    z.backward()
    return x.grad

run_selu()
run_dselu()

with torch.no_grad():
    t_selu = timeit.timeit(run_selu, number=number)
print(f'{t_selu / number * 1.e+6} us to run selu')

t_dselu = timeit.timeit(run_dselu, number=number)
print(f'{t_dselu / number * 1.e+6} us to run dselu')
