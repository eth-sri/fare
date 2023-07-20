from sklearn.cluster import k_means
from tqdm import tqdm
import numpy as np
from statsmodels.stats.proportion import proportion_confint as clop
#from rpy2.robjects.packages import importr 
#blaker = importr('BlakerCI')

class AlphaBetaAdversary:

    def _hash(self, cell):
        return hash(str(cell))

    # get a two-sided confidence interval for binomial {p} with N samples, x successes, and given error
    def _confint(self, N, x, err):
        if self.method == 'hoeffding':
            phat = x / N 
            p_lb = phat - np.sqrt(-np.log(err/2)/(2*N))
            p_ub = phat + np.sqrt(-np.log(err/2)/(2*N))
        elif self.method == 'cp':
            p_lb, p_ub = clop(x, N, alpha=err, method='beta')
        elif self.method == 'blaker':
            raise RuntimeError('Blaker is not really better than C-P')
            #ret = blaker.binom_blaker_limits(x=int(x), n=int(N), level=float((1-err)+1e-15), tol=1e-12)
            #p_lb, p_ub = ret[0], ret[1]
        return p_lb, p_ub 

    def _err_at_thresh(self, Ni, piNi, qiNi, ti, alpha_ub, beta_ub):
        if self.method == 'hoeffding':
            ti_alpha = ti / alpha_ub 
            e1 = np.exp(-2*Ni*(ti_alpha-piNi/Ni)*(ti_alpha-piNi/Ni))

            ti_beta = ti / beta_ub
            e2 = np.exp(-2*Ni*(ti_beta-qiNi/Ni)*(ti_beta-qiNi/Ni))
            return e1, e2
        else:
            # find err to get this thresh
            lo, hi = 0, 1
            while (hi - lo) > 1e-6:
                pivot_err = (lo + hi) / 2
                curr_ti = alpha_ub * self._confint(Ni, piNi, pivot_err)[1]
                if curr_ti > ti: 
                    lo = pivot_err 
                else:
                    hi = pivot_err
            err1 = hi 
            #print(f'{curr_ti} -> {err1}')

            lo, hi = 0, 1
            while (hi - lo) > 1e-6:
                pivot_err = (lo + hi) / 2
                curr_ti = beta_ub * self._confint(Ni, qiNi, pivot_err)[1]
                if curr_ti > ti: 
                    lo = pivot_err 
                else:
                    hi = pivot_err
            err2 = hi 
            #print(f'{curr_ti} -> {err2} -> {ti}')

            return err1, err2 

    def __init__(self, k, err_budget, eps_glob, eps_ab, method='cp', verbose=True):
        self.k = k 
        self.method = method
        assert self.method in ['hoeffding', 'cp', 'blaker']

        self.err_budget = err_budget
        self.eps_glob = eps_glob
        self.eps_ab = eps_ab

        self.verbose = verbose

    def ub_demographic_parity(self, embeddings):
        ret = self.ub_accuracy(embeddings, balanced=True)
        return [2*x-1 for x in ret]

    def ub_equal_opportunity(self, embeddings, favorable_y=1):
        embeddings_favorable_y = dict()
        for split in ['train', 'val', 'test']:
            mask = (embeddings[f'y_{split}'] == favorable_y).ravel()
            for d in ['y', 'z', 'c']:
                embeddings_favorable_y[f'{d}_{split}'] = embeddings[f'{d}_{split}'][mask]

        ret = self.ub_accuracy(embeddings_favorable_y, balanced=True)
        return [2*x-1 for x in ret]

    def ub_equalized_odds(self, embeddings):
        self.err_budget /= 2 # as we do it twice

        eopp_y1 = self.ub_equal_opportunity(embeddings, favorable_y=1)
        eopp_y0 = self.ub_equal_opportunity(embeddings, favorable_y=0)
        ret = [] 
        for b1, b0 in zip(eopp_y1, eopp_y0):
            ret.append((b1+b0)/2)
        return ret 

    """
        Given embeddings z (N x d) and sens. values (N x 1), compute an adversary accuracy upper bound 
        that holds with error probability <= self.err_budget (5%)
        Assumes: 1 sens. attribute with 2 possible values
    """
    def ub_accuracy(self, embeddings, balanced=False):
        # Extract data
        z_train, z_val, z_test = embeddings['z_train'], embeddings['z_val'], embeddings['z_test']
        s_train, s_val, s_test = embeddings['c_train'], embeddings['c_val'], embeddings['c_test']
        assert type(z_train) == np.ndarray
        assert s_train.shape[1] == 1

        alpha_ub, beta_ub = self.bound_train(z_train, s_train, self.eps_ab)
        tis = self.bound_val(z_val, s_val, self.err_budget - self.eps_ab - self.eps_glob, alpha_ub, beta_ub)
        T = self.bound_test(z_test, s_test, self.eps_glob, tis)

        print(f'--> Empirical on test set is: {self.empirical(z_test, s_test):.3f}')

        return [T] 

    # bound global p(s=0) and p(s=1) to get alpha_ub and beta_ub 
    def bound_train(self, z, s, budget):
        N = z.shape[0]

        unique_cells = np.unique(z, axis=0)
        assert len(unique_cells) == self.k 

        pN = (s==0).sum()

        p_lb, p_ub = self._confint(N, pN, budget)
        q_lb = 1 - p_ub 
        alpha_ub = 1 / (2*p_lb)
        beta_ub = 1 / (2*q_lb)

        alpha, beta = 1 / (2 * pN/N), 1 / (2 * (1 - pN/N))
        if self.verbose:
            print(f'[train] budget={budget:.4f} ||| (p={pN/N:.4f}, p_lb={p_lb:.4f}) ==> (alpha={alpha:.4f}, alpha_ub={alpha_ub:.4f}), (q={1 - pN/N:.4f}, q_lb={q_lb:.4f}) ==> (beta={beta:.4f}, beta_ub={beta_ub:.4f})')

        return alpha_ub, beta_ub

    # use alpha_ub and beta_ub to bound max(alpha * p(s=0|zi), beta * p(s=1|zi)) for every cell
    def bound_val(self, z, s, budget, alpha_ub, beta_ub, side='upper'):
        N = z.shape[0]
        unique_cells = np.unique(z, axis=0)
        assert len(unique_cells) == self.k 

        # Get cell data
        cell_data = [] # (hsh, Ni, piNi)
        for _, cell in enumerate(unique_cells):
            cell_members_flag = (z == cell.reshape(1, -1)).all(axis=1)
            cell_s = s[cell_members_flag]
            Ni = cell_s.shape[0]
            piNi = (cell_s == 0).sum()
            cell_data.append((self._hash(cell), Ni, piNi))
        cell_data = sorted(cell_data, key=lambda d: d[1], reverse=True)

        # Bound
        tis = {}
        for hsh, Ni, piNi, in cell_data:
            budget_cell = budget / self.k
            BSRCH = False
            if not BSRCH:
                # simple solution
                p_lb, p_ub = self._confint(Ni, piNi, budget_cell)
                q_lb, q_ub = 1 - p_ub, 1 - p_lb 
                if side == 'upper':
                    ti = max(alpha_ub * p_ub, beta_ub * q_ub)
                    if ti >= max(alpha_ub, beta_ub):
                        if self.verbose:
                            print(f' \n\n(!!!) ti >= max(alpha_ub, beta_ub), this cell should have been skipped to save budget, clipping')
                        ti = max(alpha_ub, beta_ub)
                else:
                    ti = max(alpha_ub * p_lb, beta_ub * q_lb) # only used in find_k.py
            else:
                pi = piNi / Ni 
                qi = 1 - pi 
                # binary search solution 
                lo, hi = max(pi * alpha_ub, qi * beta_ub), max(1 * alpha_ub, 1 * beta_ub)

                # Upper bound max(alpha_ub * pi, beta_ub * qi)
                while (hi - lo) > 1e-6:
                    pivot_ti = (lo + hi) / 2
                    # what is the error at this threshold? 
                    e1, e2 = self._err_at_thresh(Ni, piNi, Ni - piNi, pivot_ti, alpha_ub=alpha_ub, beta_ub=beta_ub) # for balanced
                    err = e1 + e2
                    if err < budget_cell:
                        hi = pivot_ti
                    else:
                        lo = pivot_ti
                ti = hi
                aa = self._confint(Ni, piNi, e1)[1]
                bb = self._confint(Ni, Ni-piNi, e2)[1]
                print(f'{aa} and {bb}')

            tis[hsh] = ti

            if self.verbose:
                ss = f'[val] cell (Ni={Ni}) bgt={budget_cell:.4f}, (pi,qi)=({piNi/Ni:.4f}, {1-piNi/Ni:.4f})->({alpha_ub*piNi/Ni:.4f}, {beta_ub*(1-piNi/Ni):.5f})'
                ss += f'||| ti={ti:.4f} [so] Contrib={(2*(Ni/N)*ti):.4f} and ContribIfSkipped:{2*(Ni/N):.4f}'
                print(ss)

        return tis

    # based on tis and b get the final bound A <= T
    def bound_test(self, z, s, budget, tis, side='upper'):
        a = min(list((tis.values())))
        b = max(list((tis.values())))
        print(f'[test] ti is in [{a:.2f}, {b:.2f}]')

        T = 0 

        N = z.shape[0]
        unique_cells = np.unique(z, axis=0)
        assert len(unique_cells) == self.k 
        for _, cell in enumerate(unique_cells):
            cell_members_flag = (z == cell.reshape(1, -1)).all(axis=1)
            cell_s = s[cell_members_flag]
            Ni = cell_s.shape[0]
            T += (Ni / N) * tis[self._hash(cell)]

        # not Bernoulli anymore, need to use Hoeffding
        # we pick N numbers from set {ti} where each is in [a, b]
        t0 = np.sqrt(-np.log(budget)*(b-a)*(b-a)/(2*N))
        if side == 'upper':
            T += t0
        else:
            T -= t0  # only used in find_k.py

        if self.verbose:
            print(f'[test] budget eps_glob={self.eps_glob:.4f} ||| t0={t0:.4f} (contrib={2*t0:.4f})')
        return T 

    def empirical(self, z, s):
        N = z.shape[0]
        pN = (s==0).sum()
        alpha, beta = 1 / (2 * pN/N), 1 / (2 * (1 - pN/N))
        unique_cells = np.unique(z, axis=0)
        assert len(unique_cells) == self.k 

        ret = 0
        for _, cell in enumerate(unique_cells):
            cell_members_flag = (z == cell.reshape(1, -1)).all(axis=1)
            cell_s = s[cell_members_flag]
            Ni = cell_s.shape[0]
            piNi = (cell_s == 0).sum()
            ret += (Ni / N) * max(alpha * piNi/Ni, beta * (1 - piNi/Ni))
        return 2*ret-1 
