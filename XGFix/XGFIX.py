import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import seaborn as sns
import sys
from time import time

from XGFix.phasing import *
from XGFix.simple_switch import simple_switch
from XGFix.Calibration import calibrator_module, normalize_prob
from Utils.utils import fb2proba

def mask_base_prob(base_prob, d=0):
    """
    given base out with shape [H, W, A] where 
        - H is number of Haplotypes
        - W is number of Windows
        - A is number of Ancestry
    filter out all windows that are not more than d windows away from center
    """
    H, W, A = base_prob.shape
    c = int((W-1)/2)
    masked = np.copy(base_prob)
    masked[:,np.arange(c-d),:] = 0
    masked[:,np.arange(c+d+1,W),:] = 0
    return masked

def check(Y_m, Y_p, w, base, check_criterion):

    check = False

    if check_criterion == "all":
        check = True
    elif check_criterion == "disc_smooth":
        check = Y_m[w] != Y_m[w-1] or Y_p[w] != Y_p[w-1]
    elif check_criterion == "disc_base":
        base_Y_ms, base_Y_ps = np.argmax(base[:,w-1:w +1,:],axis=2)
        check = base_Y_ms[0] != base_Y_ms[1] or base_Y_ps[0] != base_Y_ps[1]
    elif check_criterion == "disc_either":
        base_Y_ms, base_Y_ps = np.argmax(base[:,w-1:w+1,:],axis=2)
        base_check = base_Y_ms[0] != base_Y_ms[1] or base_Y_ps[0] != base_Y_ps[1]
        smooth_check = Y_m[w] != Y_m[w-1] or Y_p[w] != Y_p[w-1]
        check = base_check or smooth_check
    else:
        print("Warning: check criteration not recognized. Checking all windows")
        check = True

    return check

def xgfix_predict(data, model):
    
    Y_m, Y_p = model.smooth.predict(data).reshape(2,len(model.base))
    
    if model.calibrate:
        n = 2
        proba = model.smooth.predict_proba(data).reshape(n,-1,model.num_anc)
        proba_flatten=proba.reshape(-1,model.num_anc)
        iso_prob=np.zeros((proba_flatten.shape[0],model.num_anc))
        for i in range(model.num_anc):    
            iso_prob[:,i] = model.calibrator[i].transform(proba_flatten[:,i])
        proba = normalize_prob(iso_prob, model.num_anc).reshape(n,-1,model.num_anc)
        Y_m, Y_p = np.argmax(proba, axis = 2)

    return Y_m, Y_p


def xgfix_predict_proba(data, model):

    outs = model.smooth.predict_proba(data).reshape(-1,2,model.num_anc)

    if model.calibrate:
        n = len(data)
        proba = model.smooth.predict_proba(data).reshape(n,-1,model.num_anc)
        proba_flatten=proba.reshape(-1,model.num_anc)
        iso_prob=np.zeros((proba_flatten.shape[0],model.num_anc))
        for i in range(model.num_anc):    
            iso_prob[:,i] = model.calibrator[i].transform(proba_flatten[:,i])
        proba = normalize_prob(iso_prob, model.num_anc).reshape(n,-1,model.num_anc)
        outs = proba.reshape(-1,2,model.num_anc)

    return outs

def XGFix(M, P, model, max_it=50, non_lin_s=0, check_criterion="disc_base", max_center_offset=0, prob_comp="max", d=None, prior_switch_prob = 0.5,
            naive_switch=None, end_naive_switch=None, padding=True, calibrate=False, base_prob=None, verbose=False):

    print("Using updated version")
    
    model.mode_filter_size = 0
    model.calibrate = calibrate

    if d is None:
        d = int((model.sws-1)/2)

    if verbose:
        # print configs
        print("max center offset:", max_center_offset)
        print("non_lin_s:", non_lin_s)
        print("Mask with:", d)
        print("including naive switch:", naive_switch)
        print("including end naive switch:", end_naive_switch)
        print("prior switch prob:", prior_switch_prob)
        print("check criterion:", check_criterion)
        print("probability comparison:", prob_comp)
        print("model calibration:", model.calibrate)
        print("padding:", padding)
        
    # initial position
    X_m, X_p = np.copy(M), np.copy(P)

    # initializing base probabilities
    if base_prob is None:
        base_prob = model._get_smooth_data(data=np.array([X_m, X_p]), return_base_out=True)
    else:
        N, W, A = base_prob.shape
        assert N == 2, "Can currently only phase one individual"
        assert W == model.num_windows, "Window size of base probabilites not compatible with smoother"
        assert A == model.num_anc, "Number of Ancestry of base probabilites not compatible with smoother"

    # inferred labels of initial position
    Y_m, Y_p = model.predict(np.array([X_m, X_p]))

    # record the progression
    history = np.array([Y_m, Y_p])

    # define windows to iterate through
    centers = (np.arange(model.num_windows-model.sws+1)+(model.sws-1)/2).astype(int)
    iter_windows = np.arange(1,model.num_windows) if padding else centers

    # Track progresss
    X_m_its = [] # monitor convergence
    XGFix_tracker = (np.zeros_like(Y_m), np.ones_like(Y_p)) # monitor progression

    # Fix
    st = time()
    for it in range(max_it):

        if verbose:
            sys.stdout.write("\riteration %i/%i" % (it+1, max_it))

        if naive_switch:
            # Naive switch: heuristic to catch obvious errors and save computations
            _, _, M_track, _, _ = simple_switch(Y_m,Y_p,slack=naive_switch,cont=False,verbose=False,animate=False)
            X_m, X_p = correct_phase_error(X_m, X_p, M_track, model.win)
            # base_prob = model._get_smooth_data(data=np.array([X_m, X_p]), return_base_out=True)
            base_prob = np.array(correct_phase_error(base_prob[0], base_prob[1], M_track, model.win))
            smooth_data, _ = model._get_smooth_data(base_out = np.copy(base_prob))
            Y_m, Y_p = xgfix_predict(smooth_data, model)
            history = np.dstack([history, [Y_m, Y_p]])

        # Stop if converged
        if np.any([np.all(X_m == X_m_it) for X_m_it in X_m_its]):
            if verbose:
                print(); print("converged, stopping..", end="")
            break
        else:
            X_m_its.append(X_m)

        # Iterate through windows
        for w in iter_windows:

            # Heuristic to save computation, only check if there's a nuance
            if check(Y_m, Y_p, w, base_prob, check_criterion):

                # Different permutations depending on window position
                if w in centers:
                    center = w
                    max_center_offset_w, non_lin_s_w = max_center_offset, non_lin_s
                else:
                    center = centers[0] if w < centers[0] else centers[-1]
                    max_center_offset_w, non_lin_s_w = 0, 0
                
                # defining scope
                scope_idxs = center + np.arange(model.sws) - int((model.sws-1)/2)

                # indices of pair-wise permutations
                switch_idxs = []
                switch_idxs += [np.array([j]) for j in range(w-max_center_offset_w, w+max_center_offset_w+1)]  # single switches: xxxxxxoooooo
                switch_idxs += [np.array([w-j,w]) for j in range(1,non_lin_s_w)] # double switches left of center: xxxoocxxx
                switch_idxs += [np.array([w,w+j+1]) for j in range(non_lin_s_w)] # double switches right of center: xxxcooxxx

                # init collection of permutations and add the original
                mps = [] 
                m_orig, p_orig = np.copy(base_prob[:,scope_idxs,:])
                mps.append(m_orig); mps.append(p_orig) 
                
                # adding more permutations
                for switch_idx in switch_idxs:
                    switch_idx = np.concatenate([[scope_idxs[0]], switch_idx.reshape(-1), [scope_idxs[-1]+1]])
                    m, p = [], []
                    for s in range(len(switch_idx)-1):
                        m_s, p_s = base_prob[:,np.arange(switch_idx[s],switch_idx[s+1]),:]
                        if s%2:
                            m_s, p_s = p_s, m_s
                        m.append(m_s); p.append(p_s)
                    m, p = np.copy(np.concatenate(m,axis=0)), np.copy(np.concatenate(p,axis=0))
                    mps.append(m); mps.append(p);

                # get 2D probabilities for permutations
                masked_inps = mask_base_prob(np.array(mps), d=d).reshape(len(mps),-1)
                outs = xgfix_predict_proba(masked_inps, model)
                
                # map permutation probabilities to a scalar (R^2 -> R) for comparison
                if prob_comp=="prod":
                    probs = np.prod(np.max(outs,axis=2),axis=1)
                if prob_comp=="max":
                    probs = np.max(np.max(outs,axis=2),axis=1)

                # select the most probable one
                original_prob, switch_probs = probs[0], probs[1:]
                best_switch_prob = np.max(switch_probs)
                best_switch = switch_idxs[np.argmax(switch_probs)].reshape(-1)

                # if more likely than the original, replace the output of the base
                if best_switch_prob*prior_switch_prob > original_prob*(1-prior_switch_prob):
                    switched = True
                    m, p = [], []
                    switch_idx = np.concatenate([[0], best_switch, [model.num_windows]])
                    for s in range(len(switch_idx)-1):
                        m_s, p_s = base_prob[:,np.arange(switch_idx[s],switch_idx[s+1]),:]
                        if s%2:
                            m_s, p_s = p_s, m_s
                        m.append(m_s); p.append(p_s)
                    m, p = np.copy(np.concatenate(m,axis=0)), np.copy(np.concatenate(p,axis=0))
                    base_prob = np.copy(np.array([m,p])) 

                    # track the change
                    for switch in best_switch:
                        M_track, P_track = track_switch(np.zeros_like(Y_m), np.ones_like(Y_p), switch)
                        XGFix_tracker = track_switch(XGFix_tracker[0], XGFix_tracker[1], switch)

                    # correct inferred error on SNP level and re-label
                    X_m, X_p = correct_phase_error(X_m, X_p, M_track, model.win)
                    smooth_data, _ = model._get_smooth_data(base_out = np.copy(base_prob))
                    Y_m, Y_p = xgfix_predict(smooth_data, model)
                    history = np.dstack([history, [Y_m, Y_p]])
    
    if naive_switch:
        end_naive_switch = naive_switch
    if end_naive_switch:
        _, _, M_track, _, _ = simple_switch(Y_m,Y_p,slack=end_naive_switch,cont=False,verbose=False,animate=False)
        X_m, X_p = correct_phase_error(X_m, X_p, M_track, model.win)
        base_prob = np.array(correct_phase_error(base_prob[0], base_prob[1], M_track, model.win))
        smooth_data, _ = model._get_smooth_data(base_out = np.copy(base_prob))
        Y_m, Y_p = xgfix_predict(smooth_data, model)
        history = np.dstack([history, [Y_m, Y_p]])

    history = np.dstack([history, [Y_m, Y_p]])
    
    if verbose:
        print(); print("runtime:", np.round(time()-st))
    
    return X_m, X_p, Y_m, Y_p, history, XGFix_tracker