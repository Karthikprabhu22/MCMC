import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import fmin


def gaussian_likelihood(mean, random_variables):
    """
    Function to calculate the likelihood at a given point.
    
    Parameters
    ----------
    mean: float
        Floating point number that specifies the mean 
    
    random_variables: Numpy array
        Array of floats which are drawn from a normal distribution. 

    Returns
    -------
    likelihood: int
        The likelihood evaluated at mean for the random_variables.
    """
    length = len(random_variables)
    normalization = np.sqrt(np.log(mean) / 2 / np.pi) ** length
    likelihood = normalization * mean ** (-0.5 * sum(random_variables ** 2))
    return likelihood


def uniform_prior(mean):
    """
    Function to calculate the prior at a given point.
    
    Parameters
    ----------
    mean: float
        Floating point number that specifies the mean. 

    Returns
    -------
    prior: float
        The likelihood evaluated at mean for the random_variables.
    """
    prior = np.random.uniform(1, 10)
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


random_variables = np.random.randn(100)

ax = plt.subplot()
a_values = np.linspace(1, 10, 1000)
posterior = calc_posterior(a_values, random_variables)
ax.plot(a_values, posterior)
ax.set(xlabel="a", ylabel="belief", title="Posterior")
max_y = max(posterior)
max_x = a_values[posterior.argmax()]
# plt.text(max_x, max_y, str((max_x, max_y)))
plt.show()


def sampler(random_variables, samples, mu_init=5, proposal_width=0.25):
    """
    Function to calculate the posterior at a given point.
    
    Parameters
    ----------
    random_variables: Numpy array
        Array of floats which are drawn from a normal distribution. 
    samples: integer
        Length of the chain    
    mu_init: float
        Floating point number that specifies the starting point. 
    proposal_width: float
        Floating point number that specifies the step size. 

    Returns
    -------
    chain: Numpy array
        The MCMC chain
    """
    mu_current = mu_init
    chain = [mu_current]
    for _ in range(samples):
        # suggest new position
        mu_proposal = norm(mu_current, proposal_width).rvs()

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


parameter = sampler(random_variables, samples=15000, mu_init=5)

plt.plot(parameter)
plt.title("Trace plot")
plt.show()

plt.hist(parameter, density=True)
plt.plot(a_values, posterior / np.sqrt(np.sum(posterior ** 2)))
plt.axvline(x=np.mean(parameter), color="k", label="MCMC mean")
plt.axvline(x=a_values[posterior.argmax()], color="r", label="Analytic mean")
plt.legend()
plt.show()

print("The approximate value of e : " + str(np.mean(parameter)))
