import numpy as np
from scipy.special import logsumexp
import itertools
np.seterr(divide='ignore')


def assert_dist_non_negative(log_p):
    assert np.all(log_p <= 0)


def assert_dist_sums_to_1(log_p, axis, check_idx=None):
    if check_idx is None: assert np.all(np.isclose(np.exp(logsumexp(log_p, axis=axis)), 1))
    else: assert np.all(np.isclose(np.exp(logsumexp(log_p, axis=axis)[check_idx]), 1))


def _log_forward(data_obs, log_p, log_t_mat, log_e_mat, val_X, T):
    """
    F[1, k] = p(X_1 = k, o_1) = p(o_1 | X_1 = k) * p(X_1 = k) = e[k -> o_1] * p[k]
    F[t, k] = p(X_t = k, o_{1:t}) = e[k -> o_t] * sum_l[ F[t-1, l] * tau[l -> k] ]
    shape = N x T x |Val(X)|
    """
    N = data_obs.shape[0]
    log_F = np.zeros((N, T, len(val_X))) - np.inf
    log_F[:, 0] = log_p + log_e_mat[:, data_obs[:, 0]].T
    for t in range(1, T):
        temp = log_F[:, t - 1, None] + log_t_mat.T[None, :]
        log_F[:, t] = logsumexp(temp + log_e_mat[:, data_obs[:, t]].T[:, :, None], axis=2)

    # tests
    assert_dist_non_negative(log_F)
    return log_F


class HMM:
    def __init__(self, T, val_X, val_O, prior, transition_mat, emission_mat):
        self.T = T
        self.val_X = val_X
        self.val_O = val_O
        self.log_prior = np.log(prior)
        self.log_transition_mat = np.log(transition_mat)
        self.log_emission_mat = np.log(emission_mat)

    def get_CPDs(self):
        return {'prior': np.exp(self.log_prior),
                'transition_mat': np.exp(self.log_transition_mat),
                'emission_mat': np.exp(self.log_emission_mat)}

    def print_CPDs(self):
        cpds = self.get_CPDs()
        k = 'prior'
        print(k)
        print(np.array([f'prior({x})={cpds[k][x]:.3f}' for x in self.val_X]))
        k = 'transition_mat'
        print(k)
        print(np.array([[f'tau({xt}->{xtp1})={cpds[k][xt][xtp1]:.3f}' for xt in self.val_X] for xtp1 in self.val_X]).T)
        k = 'emission_mat'
        print(k)
        print(np.array([[f'e({xt}->{ot})={cpds[k][xt][ot]:.3f}' for ot in self.val_O] for xt in self.val_X]))

    ########################################
    ##########      Sampling      ##########
    ########################################
    def sample(self, N=1):
        """
        Assumes that the HMM CPDs are defined.
        :param N: optional, default=1. Number of samples.
        :return: (hidden, obs) for N samples from the HMM. shape of hidden & obs = (N,hmm.T)
        """
        cpds = self.get_CPDs()

        hidden_samples = np.zeros((N, self.T), dtype=int)
        obs_samples = np.zeros((N, self.T), dtype=int)

        for n in range(N):
            hidden_samples[n, 0] = np.random.choice(self.val_X, p=cpds['prior'])
            obs_samples[n, 0] = np.random.choice(self.val_O, p=cpds['emission_mat'][hidden_samples[n, 0]])

            for t in range(1, self.T):
                hidden_samples[n, t] = np.random.choice(self.val_X, p=cpds['transition_mat'][hidden_samples[n, t - 1]])
                obs_samples[n, t] = np.random.choice(self.val_O, p=cpds['emission_mat'][hidden_samples[n, t]])

        return (hidden_samples, obs_samples)

    ########################################
    ##########     Calc Prob.     ##########
    ########################################
    def log_joint(self, hidden, obs):
        """
        :param hidden - N hidden sequences. shape = (N,T)
        :param obs - N observations. shape = (N,T)
        :return log-joint probability. shape = (N)

        c3 = log(p1 * p2) = log(p1) + log(p2) = c1 + c2
        """
        N, T = hidden.shape
        log_p = self.log_prior[hidden[:, 0]] + self.log_emission_mat[hidden[:, 0], obs[:, 0]]

        for t in range(1, T):
            log_p += self.log_transition_mat[hidden[:, t - 1], hidden[:, t]] + self.log_emission_mat[
                hidden[:, t], obs[:, t]]

        return log_p

    def naive_log_likelihood(self, obs):
        """
        Calculate the log likelihood of the observations for each sample p(o1:T[i]) in a naive way (going over all
        possibilities). This will take many resources for T>5.
        :param obs - N observations. shape = (N,T)
        :return log-likelihood. shape = (N)
        """
        assert self.T < 6
        X = set(itertools.permutations(np.array([[x] * self.T for x in self.val_X]).flatten(), self.T))
        p = []
        for x in X:
            p.append(self.log_joint(np.broadcast_to(x, obs.shape), obs))
        return logsumexp(p, axis=0)

    def log_likelihood(self, obs):
        """
        :param obs - N observations. shape = (N,T)
        :return log-likelihood. shape = (N)

        log(p(o_[1:T]) = log (    sum_{k in Val(X)} p(o_1, ... , o_T, k) )     )
                       = log( sum_{k in Val(X)} exp ( log_p(o_1, ... , o_T, k) ) )
                       = log( sum_{k in Val(X)} exp ( F[T,k] )    )
        """
        log_forward = _log_forward(obs, self.log_prior, self.log_transition_mat, self.log_emission_mat, self.val_X,
                                   self.T)
        return logsumexp(log_forward[:, self.T - 1, :], axis=1)

    def log_prior_Xt(self):
        """
        :return point-wise prior. shape = (T, |val(X)|)
        P'[t, k] = p(X_t = k) = sum_{l in Val(x)} p(X_t = k | X_{t-1}=l) * p(X_{t-1} = l)
                             = sum_{l in Val(x)} P(X_t = k | X_{t-1}=l) * P'[t-1, l]
                             = P(X_t = k | X_{t-1}=l_1) * P'[t-1, l_1] + ...  P(X_t = k | X_{t-1}=l_n) * P'[t-1, l_n]

        P[t, k] = logp(X_t = k) = logp( exp(   logp(X_t = k | X_{t-1}=l_1) + P[t, l_1])   ) + ... + exp(   logp(X_t = k | X_{t-1}=l_n) + P[t, l_n])   ) )
        """
        val_X_len = len(self.val_X)
        log_p_Xt = np.full((self.T, val_X_len), -np.inf)
        log_p_Xt[0] = self.log_prior
        for t in range(1, self.T):
            for val_X_index in range(val_X_len):
                log_p_Xt[t, val_X_index] = logsumexp(self.log_transition_mat[:, val_X_index] + log_p_Xt[t - 1, :])

        # tests
        assert_dist_non_negative(log_p_Xt)  # p(Xt = k) >= 0
        assert_dist_sums_to_1(log_p_Xt, axis=-1)  # sum_k[ p(Xt = k) ] == 1
        return log_p_Xt

    def log_naive_posterior_Xt(self, obs):
        """
        :param obs - N observations. shape = (N,T)
        :return point-wise posterior. shape = (N, T, |val(X)|)

        log( p(X_t = x_t | O_t = o_t) ) = log(     p(X_t = x_t) * p(O_t = o_t | X_t = x_t)    /    sum_{x in Val(X)}  p(X_t = x) * p(O_t = o_t | X_t = x)  )
                                      = log_p(X_t = x) + log_p(O = o_t| X = x) - log(sum_{x in Val(X)}  p(X_t = x) * p(O = o_t | X = x)  )
                                      = log_p(X_t = x) + log_p(O = o_t| X = x) - log(sum_{x in Val(X)}  exp( log_p(X_t = x) + log_p(O = o_t | X = x)  )

        """
        N = len(obs)
        val_X_len = len(self.val_X)

        log_prior_Xt = self.log_prior_Xt()
        log_posterior_Xt_given_Ot = np.full((N, self.T, val_X_len), -np.inf)

        for n in range(N):
            for t in range(self.T):
                log_p_Xt = log_prior_Xt[t]
                log_p_O_given_X = self.log_emission_mat[:, obs[n, t]]
                log_posterior_Xt_given_Ot[n, t, :] = log_p_Xt + log_p_O_given_X - logsumexp(log_p_Xt + log_p_O_given_X)

        # tests
        assert_dist_non_negative(log_posterior_Xt_given_Ot)  # p(Xt = k | ot) >= 0
        assert_dist_sums_to_1(log_posterior_Xt_given_Ot, axis=-1)  # sum_k[ p(Xt = k | ot) ] == 1
        return log_posterior_Xt_given_Ot

    ########################################
    ##########     Prediction     ##########
    ########################################
    def naive_predict_by_naive_posterior(self, obs):
        """
        :param obs - N observations. shape = (N,T)
        :return X_hat - N hidden sequences. shape = (N,T)
        """
        X_hat = np.argmax(self.log_naive_posterior_Xt(obs), axis=-1)
        return X_hat
