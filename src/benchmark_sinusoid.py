import matplotlib.pyplot as plt
import numpy as np


def plot_sin_alpaca(model, test_idx, data_test, x_gt_tile, x_gt, y_gt):
    context_num_sample = range(10)
    #plt.figure(figsize=(8, 12))
    fig, axs = plt.subplots(len(context_num_sample), 1, sharex=True)
    fig.set_size_inches(8, 18)
    fig.subplots_adjust(hspace=0.03)

    for i, num_context in enumerate(context_num_sample):
        data_test_context_x = []
        data_test_context_y = []
        for task in data_test:
            data_test_context_x.append(task[0][0:num_context])
            data_test_context_y.append(task[1][0:num_context])

        data_test_context_x = np.array(data_test_context_x)
        data_test_context_y = np.array(data_test_context_y)

        mu_alpaca, sigma_alpaca = model.test(data_test_context_x, data_test_context_y, x_gt_tile)

        #plt.subplot(len(context_num_sample), 1, i + 1, sharex=True)
        axs[i].plot(x_gt, y_gt, color='r', label='Ground Truth')
        axs[i].plot(x_gt, mu_alpaca[test_idx], label='Mean Prediction')
        axs[i].fill_between(x_gt, *compute_conf_bound_var(mu_alpaca[test_idx], sigma_alpaca[test_idx]), alpha=.5, label='95% Confidence Interval')
        axs[i].plot(data_test_context_x[test_idx], data_test_context_y[test_idx], '+', color='k', markersize=10, label='Context Data')
        axs[i].set_ylim(-1, 12)
        if i==0:
            axs[i].legend(loc=1)

    plt.savefig("alpaca_update.pdf", bbox_inches='tight')
    plt.show()


def plot_sin_comparison(alpaca, gpmeta, test_idx, data_test, x_gt, y_gt, model_path=None):
    context_num_sample = range(10)
    plt.figure(figsize=(8, 12))
    x_gt_tile = np.tile(x_gt, (100, 1)).reshape(100, 100, 1)

    for i, num_context in enumerate(context_num_sample):
        ## AlPACA
        data_test_context_x = []
        data_test_context_y = []
        for task in data_test:
            data_test_context_x.append(task[0][0:num_context + 1])
            data_test_context_y.append(task[1][0:num_context + 1])

        data_test_context_x = np.array(data_test_context_x)
        data_test_context_y = np.array(data_test_context_y)

        mu_alpaca, sigma_alpaca = alpaca.test(data_test_context_x, data_test_context_y, x_gt_tile)
        ## GPMeta
        mu_gpmeta, std_gpmeta = list(zip(
            *[gpmeta.predict(test_data_tuple[0][0:num_context + 1, :], test_data_tuple[1][0:num_context + 1, :], x_gt)
              for test_data_tuple in data_test]))

        plt.subplot(len(context_num_sample), 1, i + 1)

        plt.fill_between(x_gt, *compute_conf_bound_stddev(mu_gpmeta[test_idx], std_gpmeta[test_idx]), alpha=.5, color='b')
        plt.fill_between(x_gt, *compute_conf_bound_var(mu_alpaca[test_idx], sigma_alpaca[test_idx]), alpha=.5, color='g')
        plt.plot(x_gt, mu_gpmeta[test_idx])

        plt.plot(x_gt, y_gt, color='r')

        plt.plot(data_test[test_idx][0][0:num_context + 1, :], data_test[test_idx][1][0:num_context + 1, :], '+',
                 color='k', markersize=10)


    if model_path is not None:
        plt.savefig(model_path+'sinusoid_comparison.pdf')
    plt.show()


def compute_conf_bound_var(mean, conf_int):
    lower = np.squeeze(mean) - 1.96 * np.squeeze(np.sqrt(conf_int))
    upper = np.squeeze(mean) + 1.96 * np.squeeze(np.sqrt(conf_int))
    return lower, upper

def compute_conf_bound_stddev(mean, std_dev):
    lower = np.squeeze(mean) - 1.96 * np.squeeze(std_dev)
    upper = np.squeeze(mean) + 1.96 * np.squeeze(std_dev)
    return lower, upper