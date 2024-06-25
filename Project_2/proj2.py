from HMM_proj2 import *


def get_hmm():
    # define HMM
    val_X = np.arange(2)
    val_O = np.arange(2)
    hmm_params = {
        'prior': np.array([0.1, 0.9]),
        'transition_mat': np.array([[0.1, 0.9],  # p(X_{t+1}=x | X_t=0)
                                    [0.2, 0.8]]),  # p(X_{t+1}=x | X_t=1)
        'emission_mat': np.array([[0.8, 0.2],
                                  [0.1, 0.9]])
    }
    T = 10

    hmm = HMM(T=T, val_X=val_X, val_O=val_O, prior=hmm_params['prior'],
              transition_mat=hmm_params['transition_mat'], emission_mat=hmm_params['emission_mat'])
    print(f'Load HMM. CPDs:')
    hmm.print_CPDs()
    print()
    print()
    return hmm


########################################
##########         Q1         ##########
########################################

def Q1_prior_vs_posterior(hmm, obs):
    """
    1. Calculate the posterior p(Xt | obs[i]) for each obs i
    2. Plot the prior vs the mean posterior of Xt=1 (mean over the observations i=1,...,N)

    TODO: You need to implement the log_posterior_Xt method in the HMM class

    :param obs - N observations. shape = (N,T)
    """
    # Calculate the prior p(Xt = 1)
    priors = np.exp(hmm.log_prior_Xt())[:, 1]
    # Calculate the posterior p(Xt | obs[i]) for each obs i
    log_post_Xt = hmm.log_posterior_Xt(obs)
    mean_posteriors = np.exp(log_post_Xt).mean(axis=0)[:, 1]  # mean over observations i of p(Xt=1 | obs[i])

    # Plot the prior vs the mean posterior (mean over the observations i=1,...,N)
    plot_barplot(priors, mean_posteriors, labels=['prior', 'mean posterior'], xlabel='t', ylabel='prob. of Xt',
                 title='Probability of active promoter at location t', figname='q1_prior_vs_mean_posterior')
    return log_post_Xt


def Q1_predictions(hmm, hidden, obs):
    """
    1. Predict the hidden assignment for each observation using the rule:
                    X_hat[i][t] = argmax_x[ p(X_t=x | o=obs[i]) ]
    2. Calculate the accuracy of the predication

    TODO: You need to implement the naive_predict_by_posterior method in the HMM class

    :param hidden - N hidden sequences. shape = (N,T)
    :param obs - N observations. shape = (N,T)
    """
    # Predict the hidden assignment for each observation using the rule:
    #                     X_hat[i][t] = argmax_x[ p(X_t=x | o=obs[i]) ]
    pred = hmm.naive_predict_by_posterior(obs)
    # Calculate the accuracy of the predication
    res = accuracy(pred, hidden)
    print(f'Accuracy for exact posterior = {res:.3f}')


########################################
##########        Q2-3        ##########
########################################

def calc_posterior_for_M_arr(obs, hmm, M_arr, n_repeats, algo_name):
    log_post_Xt_est = [[] for k in range(n_repeats)]
    for k in tqdm(range(n_repeats)):
        for M in M_arr:
            # if algo_name == 'RS': log_post_Xt_est[k].append(hmm.rejection_sampling_posterior(obs, M))
            if algo_name == 'Gibbs': log_post_Xt_est[k].append(hmm.gibbs_sampling_posterior(obs, M, M_start=M // 2))
            elif algo_name == 'LW': log_post_Xt_est[k].append(hmm.likelihood_weighting_posterior(obs, M))

    return log_post_Xt_est


def Q23(obs, hmm, exact_log_post_Xt, M_arr, n_repeats, algo_name):
    """
    1. Estimates the log posterior of Xt using 'algo_name' (Gibbs \ LW) with M samples for each M in M_arr.
    2. Repeated n_repeats times
    3. Plot the posterior vs M

    TODO: You need to implement the gibbs_sampling_posterior and likelihood_weighting_posterior methods in the HMM class
    """
    est_log_post_Xt = calc_posterior_for_M_arr(obs, hmm, M_arr, n_repeats, algo_name)
    plot_posterior_vs_M(np.exp(est_log_post_Xt), np.exp(exact_log_post_Xt), algo_name, M_arr, t=5, x=1)
    return est_log_post_Xt


########################################
##########         Q4         ##########
########################################
def Q4_predictions(hmm, est_log_post_Xt_per_algo, exact_log_post_Xt, hidden, M_arr, n_repeats=5):
    """
    1. Predict the hidden sequence for each sample using the est_log_post_Xt_per_algo (for each algo) and using
            the exact posteriors
    3. Plot the accuracy vs M

    TODO: You need to implement the naive_predict_by_posterior methods in the HMM class
    """
    fig, ax = plt.subplots(1, 1)

    c = 0
    for algo_name, est_log_post_Xt in est_log_post_Xt_per_algo.items():
        # calculate accuracies for estimated posteriors
        res = []
        for M_, M in enumerate(M_arr):
            res.append([])
            for k in range(n_repeats):
                pred = hmm.naive_predict_by_posterior(obs, log_post_Xt=est_log_post_Xt[k][M_])
                res[M_].append(accuracy(hidden, pred))
        res = np.array(res)
        # print
        print(f'Accuracies for {algo_name}:')
        for M_, M in enumerate(M_arr): print(f'M={M}:\t{res[M_].mean():.2f} ± {res[M_].std():.3f}')
        # plot
        plot_with_errorbars(res.T, M_arr, ax, label=algo_name, c=plt.colormaps["Set1"](c),
                            title='Prediction using estimated posteriors', xlabel='M', ylabel='Accuracy')
        c += 1

    # calculate accuracies for exact posteriors
    pred = hmm.naive_predict_by_posterior(obs, log_post_Xt=exact_log_post_Xt)
    res = accuracy(hidden, pred)
    # print
    print(f'Accuracy for Exact Inference:', res)
    # plot
    ax.plot(M_arr, [res] * len(M_arr), label=f'Exact', marker='o', c='black')
    ax.legend()
    fig.tight_layout()
    plt.savefig(f'plots/est_vs_exact_accuracy.png')
    plt.show()



if __name__ == '__main__':
    # Load HMM
    hmm1 = get_hmm()

    # Load observations
    hidden = load_data('hidden_data')
    obs = load_data('obs_data')

    # Q1
    print('''
        ########################################
        ##########         Q1         ##########
        ########################################
    ''')
    exact_log_post_Xt = Q1_prior_vs_posterior(hmm1, obs)
    Q1_predictions(hmm1, hidden, obs)

    # Q2, Q3
    print('''
        ########################################
        ##########        Q2-3        ##########
        ########################################
    ''')
    M_arr = [10, 50, 70, 100, 200, 300, 500]
    n_repeats = 5
    est_log_post_Xt_per_algo = {}
    for algo_name in ['Gibbs', 'LW']:  # TODO: If you choose to not implement the bonus 'LW', remove it from this list.
        est_log_post_Xt_per_algo[algo_name] = Q23(obs, hmm1, exact_log_post_Xt, M_arr, n_repeats, algo_name)

    # Q4
    print('''
        ########################################
        ##########         Q4         ##########
        ########################################
    ''')
    Q4_predictions(hmm1, est_log_post_Xt_per_algo, exact_log_post_Xt, hidden, M_arr, n_repeats=n_repeats)

