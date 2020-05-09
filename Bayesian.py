"""
MCMC sampler to estimate the value of the mathematical constant e.
This code uses Metropolis Hastings algorithm for sampling. 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform


def gaussian_likelihood(mean, random_variables):
    """
    Function to calculate the likelihood at a given point.
    
    Parameters
    ----------
    mean: float
        Floating point number that specifies the mean 
    
    random_variables: array
        Array of floats which are drawn from a normal distribution. 

    Returns
    -------
    likelihood: integer
        The likelihood evaluated at mean for the random_variables.
    """
    length = len(random_variables)
    normalization = np.sqrt(np.log(mean) / 2 / np.pi) ** length
    likelihood = normalization * mean ** (-0.5 * sum(random_variables ** 2))
    return likelihood


def uniform_prior(a):
    """
    Function to calculate the prior at a given point.
    
    Parameters
    ----------
    a: The value for which you want to evaluate the prior.
     

    Returns
    -------
    prior: float
        The likelihood evaluated at mean for the random_variables.
    """
    e_guess = 2.7
    scale = 1
    prior = uniform(e_guess - scale, e_guess + scale).pdf(a)
    return prior


def calc_posterior(mean, random_variables):
    """
    Function to calculate the posterior at a given point.
    
    Parameters
    ----------
    mean: float
        Floating point number that specifies the mean. 
    random_variables: Numpy array
        Array of floats which are drawn from a normal distribution. 

    Returns
    -------
    posterior: float
        The posterior evaluated at mean for the random_variables.
    """
    posterior = gaussian_likelihood(mean, random_variables) * uniform_prior(mean)
    return posterior


def sampler(random_variables, nsamples, mu_init, proposal_width=0.25):
    """
    Function to calculate the posterior at a given point.
    
    Parameters
    ----------
    random_variables: array
        Array of floats which are drawn from a normal distribution. 
    nsamples: integer
        Length of the chain    
    mu_init: float
        Floating point number that specifies the starting point. 
    proposal_width: float
        Floating point number that specifies the step size. 

    Returns
    -------
    chain: array
        The MCMC chain
    """
    mu_current = mu_init
    chain = [mu_current]
    for _ in range(nsamples):
        # suggest new position
        mu_proposal = np.random.normal(mu_current, proposal_width)

        posterior_current = calc_posterior(mu_current, random_variables)
        posterior_proposal = calc_posterior(mu_proposal, random_variables)

        # Accept proposal?
        p_accept = posterior_proposal / posterior_current

        accept = np.random.rand() < p_accept

        if accept:
            # Update position
            mu_current = mu_proposal

        chain.append(mu_current)

    return np.array(chain)


random_variables = np.random.randn(100)

plt.figure(1)
ax = plt.subplot()
a_values = np.linspace(1, 10, 1000)
posterior = calc_posterior(a_values, random_variables)
ax.plot(a_values, posterior)
ax.set(xlabel="a", ylabel="belief", title="Posterior")
max_y = max(posterior)
max_x = a_values[posterior.argmax()]
plt.show()

parameter = sampler(random_variables, nsamples=15000, mu_init=5)

plt.figure(2)
ax = plt.subplot()
ax.plot(parameter)
ax.set(xlabel="sample", ylabel="a", title="Trace plot")
plt.show()

plt.figure(3)
ax = plt.subplot()
ax.set(xlabel="a", ylabel="belief", title="Posterior distribution")
ax.hist(parameter, bins=30, density=True)
ax.plot(a_values, posterior / np.sqrt(np.sum(posterior ** 2)))
ax.axvline(x=np.mean(parameter), color="k", label="MCMC mean")
ax.axvline(x=a_values[posterior.argmax()], color="r", label="Analytic mean")
ax.axvline(x=np.e, color="b", label="True mean")
plt.legend()
plt.show()


print("The approximate value of e : " + str(np.mean(parameter)))
