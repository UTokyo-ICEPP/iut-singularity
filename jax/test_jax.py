#!/usr/bin/env python

import timeit
import jax
import jax.numpy as jnp

@jax.jit
def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

dselu = jax.jit(jax.grad(lambda x: jnp.mean(selu(x))))

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1000000,))

selu(x)
dselu(x)

number = 100

t_selu = timeit.timeit(lambda: selu(x).block_until_ready(), number=number)
print(f'{t_selu / number * 1.e+6} us to run selu')

t_dselu = timeit.timeit(lambda: dselu(x).block_until_ready(), number=number)
print(f'{t_dselu / number * 1.e+6} us to run dselu')
