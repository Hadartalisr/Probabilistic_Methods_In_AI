import os
from HMM_proj1 import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


########################################
##########        Utils       ##########
########################################
def load_data(file_name, data_path='data/'):
    return pd.read_csv(data_path + file_name + '.csv', index_col=0).values


def get_hmm(num_hmm):
    # define HMMs
    val_X = np.arange(2)
    val_O = np.arange(2)
    if num_hmm == 1:
        hmm_params = {
            'prior': np.array([0.849, 0.151]),
            'transition_mat': np.array([[0.75, 0.25],  # p(X_{t+1}=x | X_t=0)
                                        [0.25, 0.75]]),   # p(X_{t+1}=x | X_t=1)
            'emission_mat': np.array([[0.83, 0.17],
                                      [0.17, 0.83]])
        }
        T = 1000

    elif num_hmm == 2:
        hmm_params = {
            'prior': np.array([1, 0]),
            'transition_mat': np.array([[0.999, 0.001],
                                        [0.005, 0.995]]),
            'emission_mat': np.array([[0.99, 0.01],
                                      [0.01, 0.99]])
        }
        T = 1000
    elif num_hmm == 3:
        hmm_params = {
            'prior': np.array([1, 0]),
            'transition_mat': np.array([[0.999, 0.001],
                                        [0, 1]]),
            'emission_mat': np.array([[0.99, 0.01],
                                      [0.01, 0.99]])
        }
        T = 1000

    hmm = HMM(T=T, val_X=val_X, val_O=val_O, prior=hmm_params['prior'],
               transition_mat=hmm_params['transition_mat'], emission_mat=hmm_params['emission_mat'])
    print(f'Load HMM{num_hmm}. CPDs:')
    hmm.print_CPDs()
    print()
    print()
    return hmm


########################################
##########         Q2         ##########
########################################
def plot_sequence(sequences, seq_name=None, fig_name='', figsize=(20, 3)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(sequences)
    ax.set_title(f'Sequence of {seq_name} variables, {sequences.shape[1]} samples')
    ax.set_xlabel('T')
    ax.set_ylabel('modification status')
    fig.tight_layout()
    plt.savefig(f'plots/{fig_name}_sequence_{seq_name.lower()}', bbox_inches='tight')
    plt.show()


def plot_heatmap(table, table_name=None, fig_name='', figsize=(8, 4)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(table, cmap='YlOrRd', ax=ax)
    ax.set_title(f'{table_name} variables, {table.shape[0]} samples')
    ax.set_xlabel('T')
    ax.set_ylabel('samples')
    fig.tight_layout()
    plt.savefig(f'plots/{fig_name}_heatmap_{table_name.lower()}', bbox_inches='tight')
    plt.show()


def plot_histogram(scores, score_names=None, fig_name='', bins=20):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.hist(scores, bins=bins, density=True)
    ax.set_title(f'{score_names} histogram')
    ax.set_xlabel('log-joint probability')
    ax.set_ylabel('density')
    fig.tight_layout()
    plt.savefig(f'plots/{fig_name}_hist_{score_names.lower()}', bbox_inches='tight')
    plt.show()


def plot_samples(obs, hidden=None, N_plot=3, fig_name='hmm'):
    if hidden is not None: plot_sequence(hidden[:N_plot].T, seq_name=f'hidden', fig_name=fig_name)
    plot_sequence(obs[:N_plot].T, seq_name=f'observed', fig_name=fig_name)
    plot_heatmap(obs, table_name='Observed', fig_name=fig_name)


def Q2_sampling(hmm, fig_name='hmm', N=100):
    """
    Samples N samples from the HMM. Calculates the log joint probability of each sample.
    Plots:
        1. The hidden sequence of 3 samples.
        2. The observed sequence of 3 samples.
        3. A heatmap for all the observed samples.
        4. A histogram of the log joint probabilities.
    """
    hidden, obs = hmm.sample(N)
    log_joint = hmm.log_joint(hidden, obs)
    plot_samples(obs, hidden, fig_name=fig_name)
    plot_histogram(log_joint, score_names='Log-joint', bins=20, fig_name=fig_name)


########################################
##########         Q4         ##########
########################################
def Q4_identify_corrupt_data(hmm):
    """
    Loads the data and plots the histograms. Rest is TODO.
    Your job is to compute the validation_marginal_log_likelihood, real_marginal_log_likelihood and
    corrupt_marginal_log_likelihood below.
    """
    # get data
    validation_data = load_data('validation_data')
    test_data = load_data('test_data')
    validation_marginal_log_likelihood = hmm.log_likelihood(validation_data)
    test_marginal_log_likelihood = hmm.log_likelihood(test_data)

    validation_log_likelihood_expectation = np.average(validation_marginal_log_likelihood)
    validation_log_likelihood_std = np.std(validation_marginal_log_likelihood)

    is_corrupted = [log_p_O < validation_log_likelihood_expectation - 3 * validation_log_likelihood_std for log_p_O in
                    test_marginal_log_likelihood]

    real_marginal_log_likelihood = test_marginal_log_likelihood[~np.array(is_corrupted)]
    corrupt_marginal_log_likelihood = test_marginal_log_likelihood[is_corrupted]

    # plot histograms
    plt.title('Histogram of marginal log-likelihood')
    mi = np.min([corrupt_marginal_log_likelihood.min(), real_marginal_log_likelihood.min(),
                 validation_marginal_log_likelihood.min()])
    _, bins, _ = plt.hist(validation_marginal_log_likelihood, label='validation data', bins=np.arange(mi - 10, 0, 4))
    plt.hist(real_marginal_log_likelihood, label='real test data', bins=bins)
    plt.hist(corrupt_marginal_log_likelihood, label='corrupted test data', bins=bins)
    plt.xlabel('marginal log-likelihood')
    plt.ylabel('frequency')
    plt.legend()
    plt.savefig('plots/Q4_hist', bbox_inches='tight')
    plt.show()


########################################
##########         Q5         ##########
########################################
def accuracy(true, pred):
    return np.mean(true == pred)


def test_pred(hmm, data_obs, data_hidden, val_X, invalid_transitions=None):
    def check_invalid_transitions(X_hat):
        if invalid_transitions is not None:
            for (xt, xtp1) in invalid_transitions:
                print(f'# invalid transitions from {xt} to {xtp1}')
                print(np.sum([((X_hat[:, t] == 1) & (X_hat[:, t + 1] == 0)).sum() for t in range(hmm.T - 1)]))

    X_hat = hmm.naive_predict_by_naive_posterior(data_obs)
    print(f'Naive prediction accuracy = {accuracy(data_hidden.flatten(), X_hat.flatten())}, confusion mat:')
    print(confusion_matrix(data_hidden.flatten(), X_hat.flatten(), labels=val_X))
    check_invalid_transitions(X_hat)


def Q5_naive_predication(hmm, N=100):
    hidden, obs = hmm.sample(N)
    test_pred(hmm, obs, hidden, hmm.val_X, invalid_transitions=[(1, 0)])


if __name__ == '__main__':
    if not os.path.exists('plots/'): os.mkdir('plots/')

    # Q2
    print('''
        ########################################
        ##########         Q2         ##########
        ########################################
    ''')
    hmm1 = get_hmm(1)
    Q2_sampling(hmm1, fig_name='Q2_hmm1')
    hmm2 = get_hmm(2)
    Q2_sampling(hmm2, fig_name='Q2_hmm2')
    small_data = load_data('small_binary_data')
    plot_samples(small_data, fig_name='Q2_real_data')

    # Q4
    print('''
        ########################################
        ##########         Q4         ##########
        ########################################
    ''')
    hmm2 = get_hmm(2)
    Q4_identify_corrupt_data(hmm2)

    # Q5
    print('''
        ########################################
        ##########         Q5         ##########
        ########################################
    ''')
    hmm3 = get_hmm(3)
    Q5_naive_predication(hmm3)
