def alpaca_config(exp_config, opt_config, model_config):
    config = {'lr': opt_config['lr'], 'weight_decay': opt_config['weight_decay'], 'loss_type': model_config['loss_type'],
              'optimizer': opt_config['optimizer'],
              'x_dim': exp_config['x_dim'], 'y_dim': exp_config['y_dim'],
              'fixL': model_config['fixL'], 'impl_c': model_config['impl_c'], 'loss_full_cov': model_config['loss_full_cov'],
              'nn_layers': model_config['nn_layers'], 'sigma_eps': model_config['sigma_eps'], 'learn_sigma': model_config['learn_sigma'],
              'activation': model_config['activation'],
              'meta_batch_size': opt_config['meta_batch_size'], 'data_horizon': model_config['data_horizon'],
              'test_horizon': model_config['test_horizon']}
    return config


def metalgp_config(opt_config, model_config):
    config = {
        'learning_mode': model_config['learning_mode'], 'lr_params': opt_config['lr'],
        'weight_decay': opt_config['weight_decay'], 'feature_dim': model_config['feature_dim'],
        'num_iter_fit': 10000, 'covar_module': model_config['covar_module'],
        'mean_module': model_config['mean_module'], 'mean_nn_layers': model_config['mean_nn_layers'],
        'kernel_nn_layers': model_config['kernel_nn_layers'], 'task_batch_size': opt_config['meta_batch_size'],
        'normalize_data': model_config['normalize_data'], 'optimizer': opt_config['optimizer'],
        'lr_decay': opt_config['lr_decay'], 'random_seed': None}
    return config
