import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, cho_solve
import seaborn as sns

plt.rcParams['font.size'] = 11
sns.set_style('darkgrid')


class GP:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, noise_var: float = 1., lscale: float = 1.,
                 k_var: float = 1., prior_mean: float = 0, standardize=True):
        self.__lscale = lscale
        self.__k_var = k_var
        self.__noise_var = noise_var
        self.__x_train = x_train
        self.__y_train = y_train
        self.__n_train = y_train.shape[0]
        if standardize:
            self.__prior_mean = np.mean(y_train)
        else:
            self.__prior_mean = prior_mean
        self.__standardize = standardize

        self.__K_inv = None
        self.__set_k_inv()

    @property
    def n_train(self):
        return self.__n_train

    @property
    def lscale(self):
        return self.__lscale

    @property
    def k_var(self):
        return self.__k_var

    @property
    def noise_var(self):
        return self.__noise_var

    @property
    def x_train(self):
        return self.__x_train

    @property
    def y_train(self):
        return self.__y_train

    @property
    def K_inv(self):
        return self.__K_inv

    def gauss_kernel(self, x1, x2):
        assert x1.ndim == 2
        assert x2.ndim == 2
        r = np.linalg.norm(x1[:, None] - x2, axis=2)
        return self.k_var * np.exp(-0.5 * np.square(r) / np.square(self.lscale))

    def __set_k_inv(self):
        K = self.gauss_kernel(self.x_train, self.x_train)
        K += self.noise_var * np.eye(self.n_train)
        self.__K_inv = cho_solve((cholesky(K, True), True), np.eye(self.n_train))

    def predict(self, x, fullcov=False):
        assert x.ndim == 2
        kx = self.gauss_kernel(self.x_train, x)  # (n,m)
        kK = kx.T @ self.__K_inv
        mean = kK @ (self.y_train - self.__prior_mean)
        var = self.gauss_kernel(x, x) + self.noise_var * np.eye(x.shape[0]) - kK @ kx
        if not fullcov:
            var = np.diag(var)
        return mean.flatten() + self.__prior_mean, var

    def add_observation(self, x, y):
        x = np.array(x).reshape(1, -1)
        y = np.array(y).reshape(1, 1)
        self.__x_train = np.vstack([self.__x_train, x])
        self.__y_train = np.vstack([self.__y_train, y])
        self.__n_train = self.__y_train.shape[0]
        self.__set_k_inv()
        if self.__standardize:
            self.__prior_mean = np.mean(self.y_train)
        return


class RBFFourierBasis:
    def __init__(self, n_features: int, n_dim: int, rbf_ls: float = 1.,
                 rng: np.random.Generator = None):
        if rng is None:
            rng = np.random.default_rng()
        self.__n_features = n_features
        self.__n_dim = n_dim
        self.__rbf_ls = rbf_ls
        self.__rng = rng
        self.__weight = rng.normal(size=(n_dim, n_features)) / rbf_ls
        self.__offset = rng.uniform(low=0, high=2 * np.pi, size=n_features)
        return

    @property
    def n_features(self):
        return self.__n_features

    @property
    def n_dims(self):
        return self.__n_dim

    @property
    def rbf_ls(self):
        return self.__rbf_ls

    @property
    def weight(self):
        return self.__weight

    @property
    def offset(self):
        return self.__offset

    def transform(self, x):
        assert x.ndim == 2
        rff = np.sqrt(2 / self.n_features) * np.cos(x @ self.weight + self.offset)
        return rff


class BLR:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, noise_var: float = 1.,
                 w_mean: float = 0, w_var: float = 1.):
        self.__noise_var = noise_var
        self.__x_train = x_train
        self.__y_train = y_train
        self.__n_train = y_train.shape[0]
        self.__input_dim = x_train.shape[1]
        self.__w_mean = np.full(self.input_dim, w_mean)
        self.__w_var = np.eye(self.input_dim) * w_var
        self.__w_var_inv = np.eye(self.input_dim) / w_var
        self.__set_posterior()
        return

    @property
    def n_train(self):
        return self.__n_train

    @property
    def noise_var(self):
        return self.__noise_var

    @property
    def x_train(self):
        return self.__x_train

    @property
    def y_train(self):
        return self.__y_train

    @property
    def input_dim(self):
        return self.__input_dim

    @property
    def w_mean(self):
        return self.__w_mean

    @property
    def w_var(self):
        return self.__w_var

    @property
    def w_var_inv(self):
        return self.__w_var_inv

    def __set_posterior(self):
        pos_var_inv = self.w_var_inv + self.x_train.T @ self.x_train / self.noise_var
        pos_var = cho_solve((cholesky(pos_var_inv, True), True), np.eye(self.input_dim))
        pos_mean = pos_var @ (self.w_var_inv @ self.w_mean + self.x_train.T @ self.y_train / self.noise_var)
        self.__w_mean = pos_mean
        self.__w_var = pos_var
        self.__w_var_inv = pos_var_inv
        return

    def predict(self, x, fullcov=False):
        assert x.ndim == 2
        mean = x @ self.w_mean
        var = self.noise_var + x @ self.w_var @ x.T
        if not fullcov:
            var = np.diag(var)
        return mean.flatten(), var

    def add_observation(self, x, y):
        x = np.array(x).reshape(1, -1)
        y = np.array(y).reshape(1, 1)
        self.__x_train = np.vstack([self.__x_train, x])
        self.__y_train = np.vstack([self.__y_train, y])
        self.__n_train = self.__y_train.shape[0]
        self.__set_posterior()
        return


def main():
    rng = np.random.default_rng(0)
    n_trains = {3, 10, 50, 100}
    noise_var = 1e-6
    n_feature = 2000
    rff = RBFFourierBasis(n_feature, 1, 1, np.random.default_rng(0))

    f_n_feature = 2500
    f_rff = RBFFourierBasis(f_n_feature, 1, 1, np.random.default_rng(0))
    w = rng.normal(size=f_n_feature)

    def f(_x):
        return f_rff.transform(_x.reshape(-1, 1)) @ w

    n_test = 201
    x_test_plot = np.linspace(-10, 10, n_test)
    x_test = x_test_plot.reshape(-1, 1)
    # test_f = f(x_test_plot)
    n_sample = 25

    for n_train in n_trains:
        x_train = rng.uniform(-8, 2.5, n_train).reshape(-1, 1)
        y_train = f(x_train) + rng.normal(scale=np.sqrt(noise_var), size=n_train)

        # exact GP
        gp = GP(x_train, y_train, lscale=1, noise_var=noise_var, standardize=False)

        m, cov = gp.predict(x_test, fullcov=True)
        sample_path = rng.multivariate_normal(m, cov, size=n_sample)
        std = np.sqrt(np.diag(cov))
        true_m = m
        true_ci = (m + 1.96 * std, m - 1.96 * std)

        fig, ax = plt.subplots()
        # ax.plot(x_test_plot, test_f, ls='--', c='black', lw=2, label=r'$f(x)$')
        ax.plot(x_test_plot, true_m, ls='--', c='black', lw=1.2)
        ax.plot(x_test_plot, m, ls='-', c='tab:blue', lw=1, label='predict mean')
        ax.plot(x_test_plot, true_ci[0], ls='--', c='black', lw=0.8)
        ax.plot(x_test_plot, true_ci[1], ls='--', c='black', lw=0.8)
        # ax.fill_between(x_test_plot, m + 1.96 * std, m - 1.96 * std, fc='tab:blue', alpha=0.2, label='95% ci')
        ax.scatter(gp.x_train.flatten(), gp.y_train.flatten(), ec='black', c='black', s=30, marker='o',
                   label='observed', zorder=5)
        ax.plot(np.tile(x_test_plot, (n_sample, 1)).T, sample_path.T, ls='-', lw=0.5, c='tab:blue', alpha=0.5)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$f(x)$')
        ax.set_ylim(-5, 5)
        ax.set_title(f'GP', fontsize=14)
        # ax.legend()
        fig.tight_layout()
        fig.savefig(f'../image/GPsample_{n_train}.png')

        # RFF
        blr = BLR(rff.transform(x_train), y_train, noise_var=noise_var)
        m, v = blr.predict(rff.transform(x_test))
        w_sample = rng.multivariate_normal(blr.w_mean, blr.w_var, size=n_sample)
        sample_path = rff.transform(x_test) @ w_sample.T
        std = np.sqrt(v)

        fig, ax = plt.subplots()
        # ax.plot(x_test_plot, test_f, ls='--', c='black', lw=2, label=r'$f(x)$')
        ax.plot(x_test_plot, true_m, ls='--', c='black', lw=1.2)
        ax.plot(x_test_plot, m, ls='-', c='tab:orange', lw=1, label='predict mean')
        ax.plot(x_test_plot, true_ci[0], ls='--', c='black', lw=0.8)
        ax.plot(x_test_plot, true_ci[1], ls='--', c='black', lw=0.8)
        # ax.fill_between(x_test_plot, m + 1.96 * std, m - 1.96 * std, fc='tab:orange', alpha=0.2, label='95% ci')
        ax.scatter(gp.x_train.flatten(), gp.y_train.flatten(), ec='black', c='black', s=30, marker='o',
                   label='observed', zorder=5)
        ax.plot(np.tile(x_test_plot, (n_sample, 1)).T, sample_path, ls='-', lw=0.5, c='tab:orange', alpha=0.5)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$f(x)$')
        ax.set_ylim(-5, 5)
        ax.set_title(f'RFF-BLR', fontsize=14)
        # ax.legend()
        fig.tight_layout()
        fig.savefig(f'../image/RFFsample_{n_train}.png')

        # DSGP
        prior_w = rng.normal(size=(n_sample, n_feature))
        sample_path = rff.transform(x_test) @ prior_w.T
        update = gp.gauss_kernel(x_test, gp.x_train) @ gp.K_inv @ (
                    gp.y_train[:, None] - rff.transform(x_train) @ prior_w.T)
        sample_path += update

        fig, ax = plt.subplots()
        ax.plot(x_test_plot, true_m, ls='--', c='black', lw=1.2)
        ax.plot(x_test_plot, true_ci[0], ls='--', c='black', lw=0.8)
        ax.plot(x_test_plot, true_ci[1], ls='--', c='black', lw=0.8)
        ax.scatter(gp.x_train.flatten(), gp.y_train.flatten(), ec='black', c='black', s=30, marker='o',
                   label='observed', zorder=5)
        ax.plot(np.tile(x_test_plot, (n_sample, 1)).T, sample_path, ls='-', lw=0.5, c='tab:green', alpha=0.5)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$f(x)$')
        ax.set_ylim(-5, 5)
        ax.set_title(f'DSGP', fontsize=14)
        # ax.legend()
        fig.tight_layout()
        fig.savefig(f'../image/DSGPsample_{n_train}.png')

        plt.close('all')


if __name__ == '__main__':
    main()
