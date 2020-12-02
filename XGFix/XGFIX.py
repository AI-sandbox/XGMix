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

def mask_base_out(base_out, d=0):
    """
    given base out with shape [H, W, A] where 
        - H is number of Haplotypes
        - W is number of Windows
        - A is number of Ancestry
    filter out all windows that are not more than d windows away from center
    """
    H, W, A = base_out.shape
    c = int((W-1)/2)
    masked = np.copy(base_out)
    masked[:,np.arange(c-d),:] = 0
    masked[:,np.arange(c+d+1,W),:] = 0
    return masked

def XGFix(M_scrambled, P_scrambled, model, max_it=50, non_lin_s=0, max_center_offset=0, prob_comp="max", d=None, prior_switch_prob = 0.5,
            naive_switch=None, end_naive_switch=None, check_criterion="disc_base", padding=True, calibrate=False, verbose=False):
    
    model.mode_filter_size = 0
    model.calibrate = calibrate

    if d is None:
        d = int((model.sws-1)/2)

    if verbose:
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
    X_m, X_p = np.copy(M_scrambled), np.copy(P_scrambled)
    Y_m, Y_p = model.predict(np.array([X_m, X_p]))
    history = np.array([Y_m, Y_p])
    
    # getting output of base models
    base_out = model._get_smooth_data(data=np.array([X_m, X_p]), return_base_out=True)

    # solve
    st = time()
    centers = (np.arange(model.num_windows-model.sws+1)+(model.sws-1)/2).astype(int)
    windows = np.arange(1,model.num_windows)
    X_m_its = []
    XGFix_tracker = (np.zeros_like(Y_m), np.ones_like(Y_p))
    for it in range(max_it):
        if verbose:
            sys.stdout.write("\riteration %i/%i" % (it+1, max_it))

        if naive_switch:
            # Naive switch 
            # One round of naive switch with 0 slack:
            history = np.dstack([history, [Y_m, Y_p]])
            Y_m_rephased, Y_p_rephased, M_track, P_track, ani = simple_switch(Y_m,Y_p,slack=naive_switch,cont=False,verbose=False,animate=False)
            X_m, X_p = correct_phase_error(X_m, X_p, M_track, model.win)
            Y_m, Y_p = model.predict(np.array([X_m, X_p]))
            history = np.dstack([history, [Y_m, Y_p]])
            base_out = model._get_smooth_data(data=np.array([X_m, X_p]), return_base_out=True)

        # Stop if converged
        if np.any([np.all(X_m == X_m_it) for X_m_it in X_m_its]):
            if verbose:
                print(); print("converged, stopping..", end="")
            break
        else:
            X_m_its.append(X_m)

        iter_windows = windows if padding else centers
        for w in iter_windows:

            if w in centers:
                center = w
                max_center_offset_w, non_lin_s_w = max_center_offset, non_lin_s
            else:
                center = centers[0] if w < centers[0] else centers[-1]
                max_center_offset_w, non_lin_s_w = 0, 0

            check = False
            if check_criterion == "all":
                check = True
            elif check_criterion == "disc_smooth":
                check = Y_m[w] != Y_m[w-1] or Y_p[w] != Y_p[w-1]
            elif check_criterion == "disc_base":
                base_Y_ms, base_Y_ps = np.argmax(base_out[:,w-1:w +1,:],axis=2)
                check = base_Y_ms[0] != base_Y_ms[1] or base_Y_ps[0] != base_Y_ps[1]
            elif check_criterion == "disc_either":
                base_Y_ms, base_Y_ps = np.argmax(base_out[:,w-1:w+1,:],axis=2)
                base_check = base_Y_ms[0] != base_Y_ms[1] or base_Y_ps[0] != base_Y_ps[1]
                smooth_check = Y_m[w] != Y_m[w-1] or Y_p[w] != Y_p[w-1]
                check = base_check or smooth_check
            else:
                print("Warning: check criteration not recognized. Checking all windows")
                check = True

            if check:

                M_track, P_track = np.zeros_like(Y_m), np.ones_like(Y_p) # init tracking
                scope_idxs = center + np.arange(model.sws) - int((model.sws-1)/2) # defining scopes
                
                # collecting permutations
                switch_idxs = []
                switch_idxs += [np.array([j]) for j in range(w-max_center_offset_w, w+max_center_offset_w+1)]  # single switches: xxxxxxoooooo
                switch_idxs += [np.array([w-j,w]) for j in range(1,non_lin_s_w)] # double switches left of center: xxxoocxxx
                switch_idxs += [np.array([w,w+j+1]) for j in range(non_lin_s_w)] # double switches right of center: xxxcooxxx

                # collecting permutations, starting with original
                mps = []
                m_orig, p_orig = np.copy(base_out[:,scope_idxs,:])
                mps.append(m_orig); mps.append(p_orig)
                
                # adding permutations
                for switch_idx in switch_idxs:
                    switch_idx = np.concatenate([[scope_idxs[0]], switch_idx.reshape(-1), [scope_idxs[-1]+1]])
                    m, p = [], []
                    for s in range(len(switch_idx)-1):
                        m_s, p_s = base_out[:,np.arange(switch_idx[s],switch_idx[s+1]),:]
                        if s%2:
                            m_s, p_s = p_s, m_s
                        m.append(m_s); p.append(p_s)
                    m, p = np.copy(np.concatenate(m,axis=0)), np.copy(np.concatenate(p,axis=0))
                    mps.append(m); mps.append(p);

                # get probabilities
                masked_inps = mask_base_out(np.array(mps), d=d).reshape(len(mps),-1)
                outs = model.smooth.predict_proba(masked_inps).reshape(-1,2,model.num_anc)

                # optional calibration of the probabilities
                if model.calibrate:
                    n = len(masked_inps)
                    proba = model.smooth.predict_proba(masked_inps).reshape(n,-1,model.num_anc)
                    proba_flatten=proba.reshape(-1,model.num_anc)
                    iso_prob=np.zeros((proba_flatten.shape[0],model.num_anc))
                    for i in range(model.num_anc):    
                        iso_prob[:,i] = model.calibrator[i].transform(proba_flatten[:,i])
                    proba = normalize_prob(iso_prob, model.num_anc).reshape(n,-1,model.num_anc)
                    outs = proba.reshape(-1,2,model.num_anc)
                
                # map permutation probabilities to a scalar (R^2 -> R) 
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
                        m_s, p_s = base_out[:,np.arange(switch_idx[s],switch_idx[s+1]),:]
                        if s%2:
                            m_s, p_s = p_s, m_s
                        m.append(m_s); p.append(p_s)
                    m, p = np.copy(np.concatenate(m,axis=0)), np.copy(np.concatenate(p,axis=0))
                    base_out = np.copy(np.array([m,p])) 

                    # track the switch
                    for switch in best_switch:
                        M_track, P_track = track_switch(M_track, P_track, switch)
                        XGFix_tracker = track_switch(XGFix_tracker[0], XGFix_tracker[1], switch)

                    # correct inferred error
                    X_m, X_p = correct_phase_error(X_m, X_p, M_track, model.win)
                    smooth_data, _ = model._get_smooth_data(base_out = np.copy(base_out))
                    Y_m, Y_p = model.smooth.predict(smooth_data).reshape(2,len(model.base))

                    # if calibrating
                    if model.calibrate:
                        n = 2
                        proba = model.smooth.predict_proba(smooth_data).reshape(n,-1,model.num_anc)
                        proba_flatten=proba.reshape(-1,model.num_anc)
                        iso_prob=np.zeros((proba_flatten.shape[0],model.num_anc))
                        for i in range(model.num_anc):    
                            iso_prob[:,i] = model.calibrator[i].transform(proba_flatten[:,i])
                        proba = normalize_prob(iso_prob, model.num_anc).reshape(n,-1,model.num_anc)
                        Y_m, Y_p = np.argmax(proba, axis = 2)

                    history = np.dstack([history, [Y_m, Y_p]])
    
    if naive_switch:
        end_naive_switch = naive_switch
    if end_naive_switch:
        # One round of naive switch:
        Y_m_rephased, Y_p_rephased, M_track, P_track, ani = simple_switch(Y_m,Y_p,slack=end_naive_switch,cont=False,verbose=False,animate=False)
        X_m, X_p = correct_phase_error(X_m, X_p, M_track, model.win)
        Y_m, Y_p = model.predict(np.array([X_m, X_p]))
        history = np.dstack([history, [Y_m, Y_p]])

    if verbose:
        print(); print("runtime:", np.round(time()-st))
    
    return X_m, X_p, Y_m, Y_p, history, XGFix_tracker