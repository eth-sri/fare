import torch
import numpy as np
from numpy.random import beta
from sklearn.metrics import average_precision_score, balanced_accuracy_score


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def sample_batch_sen_idx(X, A, y, batch_size, s):
    batch_idx = np.random.choice(np.where(A == s)[0], size=batch_size, replace=False).tolist()
    batch_x = X[batch_idx]
    batch_y = y[batch_idx]
    batch_x = torch.tensor(batch_x).cuda().float()
    batch_y = torch.tensor(batch_y).cuda().float()

    return batch_x, batch_y


def sample_batch_sen_idx_y(X, A, y, batch_size, s):
    batch_idx = []
    for i in range(2):
        idx = list(set(np.where(A == s)[0]) & set(np.where(y == i)[0]))
        batch_idx += np.random.choice(idx, size=batch_size, replace=False).tolist()

    batch_x = X[batch_idx]
    batch_y = y[batch_idx]
    batch_x = torch.tensor(batch_x).cuda().float()
    batch_y = torch.tensor(batch_y).cuda().float()

    return batch_x, batch_y


# freeze and activate gradient w.r.t. parameters
def model_freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def model_activate(model):
    for param in model.parameters():
        param.requires_grad = True


def matrix_evaluator(loss, x, y, model):
    def evaluator(v):
        hvp = hessian_vector_prodct(loss, x, y, model, v)
        return hvp

    return evaluator

def hessian_vector_prodct(loss, x, y, model, vector_to_optimize):
    # given a gradient vector and parameter with the same size, compute its input for CG: AX
    # need to re-compute the gradient
    prediction_loss = loss(model(x), y)
    partial_grad = torch.autograd.grad(
        prediction_loss, model.parameters(), create_graph=True
    )  # need to compute hessian
    flat_grad = torch.cat([g.contiguous().view(-1) for g in partial_grad])
    h = torch.sum(flat_grad * vector_to_optimize)
    hvp = torch.autograd.grad(h, model.parameters(), retain_graph=True)
    hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
    return hvp_flat


def cg_solve(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10, x_init=None):
    """
    Goal: Solve Ax=b equivalent to minimizing f(x) = 1/2 x^T A x - x^T b
    The problem can be solved through CG solver
    Assumption: A is PSD, no damping term is used here (must be damped externally in f_Ax)
    Algorithm template from wikipedia
    Verbose mode works only with numpy
    """

    if type(b) == torch.Tensor:
        x = torch.zeros(b.shape[0]) if x_init is None else x_init
        x = x.to(b.device)
        if b.dtype == torch.float16:
            x = x.half()
        r = b - f_Ax(x)
        p = r.clone()
    elif type(b) == np.ndarray:
        x = np.zeros_like(b) if x_init is None else x_init
        r = b - f_Ax(x)
        p = r.copy()
    else:
        print("Type error in cg")

    fmtstr = "%10i %10.3g %10.3g %10.3g"
    titlestr = "%10s %10s %10s %10s"
    if verbose:
        print(titlestr % ("iter", "residual norm", "soln norm", "obj fn"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose:
            obj_fn = 0.5 * x.dot(f_Ax(x)) - 0.5 * b.dot(x)
            norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
            print(fmtstr % (i, r.dot(r), norm_x, obj_fn))

        rdotr = r.dot(r)
        Ap = f_Ax(p)
        alpha = rdotr / (p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        newrdotr = r.dot(r)
        beta = newrdotr / rdotr
        p = r + beta * p

        if newrdotr < residual_tol:
            # print("Early CG termination because the residual was small")
            break

    if callback is not None:
        callback(x)
    if verbose:
        obj_fn = 0.5 * x.dot(f_Ax(x)) - 0.5 * b.dot(x)
        norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
        print(fmtstr % (i, r.dot(r), norm_x, obj_fn))
    return x


def construct_b(loss_1, loss_2, model_1, model_2, kappa=1.0):
    # compute the b term without gradient
    partial_grad_1 = torch.autograd.grad(
        loss_1, model_1.parameters(), create_graph=False, retain_graph=False
    )
    partial_grad_2 = torch.autograd.grad(
        loss_2, model_2.parameters(), create_graph=False, retain_graph=False
    )

    flat_partial_grad_1 = torch.cat([g.contiguous().view(-1) for g in partial_grad_1])
    flat_partial_grad_2 = torch.cat([g.contiguous().view(-1) for g in partial_grad_2])

    flat_model_1 = torch.cat([g.contiguous().view(-1) for g in model_1.parameters()])
    flat_model_2 = torch.cat([g.contiguous().view(-1) for g in model_2.parameters()])

    gap = flat_model_1 - flat_model_2

    b_1 = flat_partial_grad_1 + kappa * gap
    b_2 = flat_partial_grad_2 - kappa * gap

    return b_1, b_2


# adapted from Implicit-MAML
def meta_grad_update(meta_grad, model, optimizer, flat_grad=False):
    """
    Given the gradient, step with the outer optimizer using the gradient.
    Assumed that the gradient is a tuple/list of size compatible with model.parameters()
    If flat_grad, then the gradient is a flattened vector
    """
    # check = 0
    # for p in model.parameters():
    #     check = check + 1 if type(p.grad) == type(None) else check
    # if check > 0:
    #     # initialize the grad fields properly
    #     dummy_loss = self.regularization_loss(self.get_params())
    #     dummy_loss.backward()  # this would initialize required variables
    if flat_grad:
        offset = 0
        # grad = utils.to_device(grad, self.use_gpu)
        for p in model.parameters():
            this_grad = meta_grad[offset : offset + p.nelement()].view(p.size())
            p.grad.copy_(this_grad)
            offset += p.nelement()
    else:
        for i, p in enumerate(model.parameters()):
            p.grad = meta_grad[i]
    optimizer.step()


def train_implicit(
    fea,
    clf_0,
    clf_1,
    criterion,
    optimizer_fea,
    optimizer_clf_0,
    optimizer_clf_1,
    X_train,
    A_train,
    y_train,
    kappa=1e-4,
    batch_size=128,
    n_epoch=300,
    max_inner=25,
    out_step = 10,
):

    fea.train()
    clf_0.train()
    clf_1.train()



    for it in range(n_epoch):
        ap_train, gap_train = evaluate_pp_implicit(fea, clf_0, clf_1, X_train, y_train, A_train)
        print(f'it={it+1}/{n_epoch} | [train] acc: {ap_train} prediction gap: {gap_train}', flush=True)

        # Gender dataset Split
        batch_x_0, batch_y_0 = sample_batch_sen_idx(X_train, A_train, y_train, batch_size, 0)
        batch_x_1, batch_y_1 = sample_batch_sen_idx(X_train, A_train, y_train, batch_size, 1)

        # Step 1: freeze feature representation, inner_optimization
        model_freeze(fea)

        z_0 = fea(batch_x_0)
        z_1 = fea(batch_x_1)

        # inner_loop for obtaining h_{\epsilon}
        for _ in range(max_inner):
            y_pred_0 = clf_0(z_0)
            y_pred_1 = clf_1(z_1)

            loss_0 = criterion(y_pred_0, batch_y_0)
            loss_1 = criterion(y_pred_1, batch_y_1)

            optimizer_clf_0.zero_grad()
            loss_0.backward()
            optimizer_clf_0.step()

            optimizer_clf_1.zero_grad()
            loss_1.backward()
            optimizer_clf_1.step()

        # Step2: computing P_1 and P_2 (does not require the gradient of lambda)
        # clear gradient
        optimizer_clf_0.zero_grad()
        optimizer_clf_1.zero_grad()


        y_pred_0 = clf_0(z_0)
        y_pred_1 = clf_1(z_1)

        loss_0 = criterion(y_pred_0, batch_y_0)
        loss_1 = criterion(y_pred_1, batch_y_1)

        AX_0 = matrix_evaluator(criterion, z_0, batch_y_0, clf_0)
        AX_1 = matrix_evaluator(criterion, z_1, batch_y_1, clf_1)

        b_0, b_1 = construct_b(loss_0, loss_1, clf_0, clf_1, kappa=kappa)

        P_0 = cg_solve(AX_0, b_0, cg_iters=out_step)
        P_1 = cg_solve(AX_1, b_1, cg_iters=out_step)

        P_0.detach()
        P_1.detach()

        # Step 3: compute meta-gradient (gradient of the representation)
        model_activate(fea)
        optimizer_clf_0.zero_grad()
        optimizer_clf_1.zero_grad()
        optimizer_fea.zero_grad()

        z_0 = fea(batch_x_0)
        z_1 = fea(batch_x_1)

        y_pred_0 = clf_0(z_0)
        y_pred_1 = clf_1(z_1)

        loss_0 = criterion(y_pred_0, batch_y_0)
        loss_1 = criterion(y_pred_1, batch_y_1)

        partial_lam_0 = torch.autograd.grad(loss_0, fea.parameters(), retain_graph=True)
        partial_lam_1 = torch.autograd.grad(loss_1, fea.parameters(), retain_graph=True)

        partial_h_0 = torch.autograd.grad(
            loss_0, clf_0.parameters(), create_graph=True, allow_unused=True
        )
        partial_h_1 = torch.autograd.grad(
            loss_1, clf_1.parameters(), create_graph=True, allow_unused=True
        )

        flat_grad_0 = torch.cat([g.contiguous().view(-1) for g in partial_h_0])
        hessian_vector_0 = torch.sum(flat_grad_0 * P_0)
        joint_hessian_0 = torch.autograd.grad(hessian_vector_0, fea.parameters())

        flat_grad_1 = torch.cat([g.contiguous().view(-1) for g in partial_h_1])
        hessian_vector_1 = torch.sum(flat_grad_1 * P_1)
        joint_hessian_1 = torch.autograd.grad(hessian_vector_1, fea.parameters())


        # the original gradients are in the form of tuple, we need additional function to make a new tuple
        meta_gradient = make_meta_grad(
            partial_lam_0, partial_lam_1, joint_hessian_0, joint_hessian_1
        )

        meta_grad_update(meta_gradient, fea, optimizer_fea)



def make_meta_grad(partial_lam_0, partial_lam_1, joint_hessian_0, joint_hessian_1):
    list_meta = []
    for i in range(len(partial_lam_0)):
        list_meta.append(
            partial_lam_0[i] + partial_lam_1[i] - joint_hessian_0[i] - joint_hessian_1[i]
        )
    return tuple(list_meta)



# baseline approach adapted from fair-mixup
def train_dp(
    model, criterion, optimizer, X_train, A_train, y_train, method, lam, batch_size=500, niter=100
):
    model.train()
    for it in range(niter):

        # Gender Split
        batch_x_0, batch_y_0 = sample_batch_sen_idx(X_train, A_train, y_train, batch_size, 0)
        batch_x_1, batch_y_1 = sample_batch_sen_idx(X_train, A_train, y_train, batch_size, 1)

        if method == "mixup":
            # Fair Mixup
            alpha = 1
            gamma = beta(alpha, alpha)

            batch_x_mix = batch_x_0 * gamma + batch_x_1 * (1 - gamma)
            batch_x_mix = batch_x_mix.requires_grad_(True)

            output = model(batch_x_mix)

            # gradient regularization
            gradx = torch.autograd.grad(output.sum(), batch_x_mix, create_graph=True)[0]

            batch_x_d = batch_x_1 - batch_x_0
            grad_inn = (gradx * batch_x_d).sum(1)
            E_grad = grad_inn.mean(0)
            loss_reg = torch.abs(E_grad)

        elif method == "GapReg":
            # Gap Regularizatioon
            output_0 = model(batch_x_0)
            output_1 = model(batch_x_1)
            loss_reg = torch.abs(output_0.mean() - output_1.mean())
        else:
            # ERM
            loss_reg = 0

        # ERM loss
        batch_x = torch.cat((batch_x_0, batch_x_1), 0)
        batch_y = torch.cat((batch_y_0, batch_y_1), 0)

        output = model(batch_x)
        loss_sup = criterion(output, batch_y)

        # final loss
        loss = loss_sup + lam * loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate_pp_implicit(fea, clf_0, clf_1, X_test, y_test, A_test):
    # evaluating the predictive parity in classification
    fea.eval()
    clf_0.eval()
    clf_1.eval()

    idx_0 = np.where(A_test == 0)[0]
    idx_1 = np.where(A_test == 1)[0]

    X_test_0 = X_test[idx_0]
    X_test_1 = X_test[idx_1]
    X_test_0 = torch.tensor(X_test_0).cuda().float()
    X_test_1 = torch.tensor(X_test_1).cuda().float()
    Y_test_0 = y_test[idx_0]
    Y_test_1 = y_test[idx_1]

    # compute the hat distribution (score, np form)
    y_hat_0 = clf_0(fea(X_test_0)).detach().cpu().numpy()
    y_hat_1 = clf_1(fea(X_test_1)).detach().cpu().numpy()

    if len(np.where(y_hat_0 < 0.5)[0]) == 0:
        pp_00 = 0
    else:
        pp_00 = len(list(set(np.where(y_hat_0 < 0.5)[0]) & set(np.where(Y_test_0 == 0)[0]))) / len(
            np.where(y_hat_0 < 0.5)[0]
        )
    if len(np.where(y_hat_1 < 0.5)[0]) == 0:
        pp_10 = 0
    else:
        pp_10 = len(list(set(np.where(y_hat_1 < 0.5)[0]) & set(np.where(Y_test_1 == 0)[0]))) / len(
            np.where(y_hat_1 < 0.5)[0]
        )

    gap_0 = np.abs(pp_00 - pp_10)

    if len(np.where(y_hat_0 >= 0.5)[0]) == 0:
        pp_01 = 0
    else:
        pp_01 = len(list(set(np.where(y_hat_0 >= 0.5)[0]) & set(np.where(Y_test_0 == 1)[0]))) / len(
            np.where(y_hat_0 >= 0.5)[0]
        )
    if len(np.where(y_hat_1 >= 0.5)[0]) == 0:
        pp_11 = 0
    else:
        pp_11 = len(list(set(np.where(y_hat_1 >= 0.5)[0]) & set(np.where(Y_test_1 == 1)[0]))) / len(
            np.where(y_hat_1 >= 0.5)[0]
        )

    gap_1 = np.abs(pp_01 - pp_11)

    gap = (gap_0 + gap_1) / 2.0

    # compute average accuracy
    ap = (
        average_precision_score(Y_test_0, y_hat_0) + average_precision_score(Y_test_1, y_hat_1)
    ) / 2.0
    return ap, gap



# evluate PP from feature + clf structure, analogous to evaluate implicit pp
def evaluate_pp_model_one_clf(fea, clf, X_test, y_test, A_test):

    fea.eval()
    clf.eval()

    idx_0 = np.where(A_test == 0)[0]
    idx_1 = np.where(A_test == 1)[0]

    X_test_0 = X_test[idx_0]
    X_test_1 = X_test[idx_1]
    X_test_0 = torch.tensor(X_test_0).cuda().float()
    X_test_1 = torch.tensor(X_test_1).cuda().float()
    Y_test_0 = y_test[idx_0]
    Y_test_1 = y_test[idx_1]

    # compute the predictive distribution
    y_hat_0 = clf(fea(X_test_0)).detach().cpu().numpy()
    y_hat_1 = clf(fea(X_test_1)).detach().cpu().numpy()

    if len(np.where(y_hat_0 < 0.5)[0]) == 0:
        pp_00 = 0
    else:
        pp_00 = len(list(set(np.where(y_hat_0 < 0.5)[0]) & set(np.where(Y_test_0 == 0)[0]))) / len(
            np.where(y_hat_0 < 0.5)[0]
        )
    if len(np.where(y_hat_1 < 0.5)[0]) == 0:
        pp_10 = 0
    else:
        pp_10 = len(list(set(np.where(y_hat_1 < 0.5)[0]) & set(np.where(Y_test_1 == 0)[0]))) / len(
            np.where(y_hat_1 < 0.5)[0]
        )

    gap_0 = np.abs(pp_00 - pp_10)

    if len(np.where(y_hat_0 >= 0.5)[0]) == 0:
        pp_01 = 0
    else:
        pp_01 = len(list(set(np.where(y_hat_0 >= 0.5)[0]) & set(np.where(Y_test_0 == 1)[0]))) / len(
            np.where(y_hat_0 >= 0.5)[0]
        )
    if len(np.where(y_hat_1 >= 0.5)[0]) == 0:
        pp_11 = 0
    else:
        pp_11 = len(list(set(np.where(y_hat_1 >= 0.5)[0]) & set(np.where(Y_test_1 == 1)[0]))) / len(
            np.where(y_hat_1 >= 0.5)[0]
        )

    gap_1 = np.abs(pp_01 - pp_11)

    gap = (gap_0 + gap_1) / 2.0

    # compute average accuracy
    ap = (
        average_precision_score(Y_test_0, y_hat_0) + average_precision_score(Y_test_1, y_hat_1)
    ) / 2.0

    return ap, gap


def evaluate_pp_model(model, X_test, y_test, A_test):
    # evaluate the predictive parity w.r.t. the black-box model

    model.eval()

    idx_0 = np.where(A_test == 0)[0]
    idx_1 = np.where(A_test == 1)[0]

    X_test_0 = X_test[idx_0]
    X_test_1 = X_test[idx_1]
    X_test_0 = torch.tensor(X_test_0).cuda().float()
    X_test_1 = torch.tensor(X_test_1).cuda().float()
    Y_test_0 = y_test[idx_0]
    Y_test_1 = y_test[idx_1]

    # compute the predictive distribution

    y_hat_0 = model(X_test_0).data.cpu().numpy()
    y_hat_1 = model(X_test_1).data.cpu().numpy()

    if len(np.where(y_hat_0 < 0.5)[0]) == 0:
        pp_00 = 0
    else:
        pp_00 = len(list(set(np.where(y_hat_0 < 0.5)[0]) & set(np.where(Y_test_0 == 0)[0]))) / len(
            np.where(y_hat_0 < 0.5)[0]
        )
    if len(np.where(y_hat_1 < 0.5)[0]) == 0:
        pp_10 = 0
    else:
        pp_10 = len(list(set(np.where(y_hat_1 < 0.5)[0]) & set(np.where(Y_test_1 == 0)[0]))) / len(
            np.where(y_hat_1 < 0.5)[0]
        )

    gap_0 = np.abs(pp_00 - pp_10)

    if len(np.where(y_hat_0 >= 0.5)[0]) == 0:
        pp_01 = 0
    else:
        pp_01 = len(list(set(np.where(y_hat_0 >= 0.5)[0]) & set(np.where(Y_test_0 == 1)[0]))) / len(
            np.where(y_hat_0 >= 0.5)[0]
        )
    if len(np.where(y_hat_1 >= 0.5)[0]) == 0:
        pp_11 = 0
    else:
        pp_11 = len(list(set(np.where(y_hat_1 >= 0.5)[0]) & set(np.where(Y_test_1 == 1)[0]))) / len(
            np.where(y_hat_1 >= 0.5)[0]
        )

    gap_1 = np.abs(pp_01 - pp_11)

    gap = (gap_0 + gap_1) / 2.0

    # compute average accuracy
    ap = (
        average_precision_score(Y_test_0, y_hat_0) + average_precision_score(Y_test_1, y_hat_1)
    ) / 2.0

    return ap, gap


