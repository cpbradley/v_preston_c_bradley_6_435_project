#!/usr/env/python

'''
An implementation of an adjusted inference algorithm in:
Inference in the Space of Topological Maps: An MCMC-based Approach
https://smartech.gatech.edu/bitstream/handle/1853/38451/Ranganathan04iros.pdf

Maintainer: vpreston-at-{whoi, mit}-dot-edu
'''

import itertools
import random
import copy
import operator
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def PTM(T, Z, posterior_func, niter, R=10, sig_o=0.1, sig_t=0.1, max_range=3., max_penalty=80., visualize=True, filename=None):
    ''' Function which yields a PTM as presented in work by Ranganathan et al.
    Input:
    - T (topolgy) initial topology
    - Z (list of tuples of floats) landmark observations/detections
    - posterior_func (function handle) the posterior on landmark observations
    - niter (int) number of samples to draw; includes burn-in samples
    - R (int) number of samples in the importance sampler to draw
    - sig_o (float) how much "error" in the odom measurement to tolerate
    - sig_t (float) how much "error" in the topology proposal to tolerate
    - max_range (float) how "far away" to penalize detections
    - max_penalty (float) amount to penalize "far away" detections
    - visualize (boolean) whether to show plots of the results
    - filename (string) if not None, will save a pickle of the samples for post-processing
    Output:
    - samples (list of topologies) samples drawn
    '''
    ptm = metropolis_hastings(T=T,
                              Z=Z,
                              Cov=None,
                              posterior_func=posterior_func,
                              niter=niter,
                              R=R,
                              sig_o=sig_o,
                              sig_t=sig_t,
                              max_range=max_range,
                              max_penalty=max_penalty,
                              likelihood_type=None)
    if filename is not None:
        # save samples to a pickle for processing later
        with open(filename, 'wb') as fp:
            pickle.dump(ptm, fp)

    if visualize:
        # show a figure of the detections from the original topology
        map_fig = plot_detections(T=T, Z=Z)

        # show the sampling results
        fig = plot_report(samples=ptm, Z=Z)

        plt.show()
        plt.close()

    return ptm

def covPTM(T, Z, Cov, posterior_func, niter, R=10, sig_o=0.1, sig_t=0.1, max_range=3., max_penalty=80., likelihood_type='Mahalanobis', visualize=True, filename=None):
    ''' Function which yields a PTM as presented in work by Ranganathan et al.
    Input:
    - T (topolgy) initial topology
    - Z (list of tuples of floats) landmark observations/detections
    - Cov (list of matrices) landmark observation covariances
    - posterior_func (function handle) the posterior on landmark observations
    - niter (int) number of samples to draw; includes burn-in samples
    - R (int) number of samples in the importance sampler to draw
    - sig_o (float) how much "error" in the odom measurement to tolerate
    - sig_t (float) how much "error" in the topology proposal to tolerate
    - max_range (float) how "far away" to penalize detections
    - max_penalty (float) amount to penalize "far away" detections
    - likelihood_type (string) one of 'Mahalanobis', 'Bhattacharyya', or 'Xodom' None will yield a normal PTM
    - visualize (boolean) whether to show plots of the results
    - filename (string) if not None, will save a pickle of the samples for post-processing
    Output:
    - samples (list of topologies) samples drawn
    '''
    cptm = metropolis_hastings(T=T,
                               Z=Z,
                               Cov=Cov,
                               posterior_func=posterior_func,
                               niter=niter,
                               R=R,
                               sig_o=sig_o,
                               sig_t=sig_t,
                               max_range=max_range,
                               max_penalty=max_penalty,
                               likelihood_type=likelihood_type)
    if filename is not None:
        # save samples to a pickle for processing later
        with open(filename, 'wb') as fp:
            pickle.dump(ptm, fp)

    if visualize:
        # show a figure of the detections from the original topology
        map_fig = plot_detections(T=T, Z=Z, Cov=Cov)

        # show the sampling results
        fig = plot_report(samples=cptm, Z=Z)

        plt.show()
        plt.close()

    return cptm

def metropolis_hastings(T, Z, Cov, posterior_func, niter, R=10, sig_o=0.1, sig_t=0.1, max_range=3., max_penalty=80., likelihood_type=None):
    ''' Function which proposes new topolgies from a proposal
    distribution, calculates the acceptance ratio, and returns
    the relevant sample. Algorithm 1.

    Input:
    - T (topolgy) initial topology
    - Z (list of tuples of floats) landmark observations/detections
    - Cov (list of matrices of flaots) landmark detection covariances
    - posterior_func (function handle) the posterior on landmark observations
    - niter (int) number of samples to draw; includes burn-in samples
    - R (int) number of samples in the importance sampler to draw
    - sig_o (float) how much "error" in the odom measurement to tolerate
    - sig_t (float) how much "error" in the topology proposal to tolerate
    - max_range (float) how "far away" to penalize detections
    - max_penalty (float) amount to penalize "far away" detections
    - likelihood_type (string) one of "Bhattacharyya, Mahalanobis, or Xodom", default is None
    Output:
    - samples (list of topologies) samples drawn
    '''
    samples = []
    move_mix = []
    accepted_moves = []

    #initialize the posterior distribution over the initial proposal.
    #we will keep track of every proposal to prevent the need to recalculate in each loop
    _, denum = posterior_func(T, Z, Cov, R, sig_o, sig_t, max_range, max_penalty, likelihood_type)

    for i in range(niter):
        if i % 10 == 0:
            print 'Samples so far: ', i #print a sample update

        # draw proposal
        T_proposal, r_prop, move = draw_sample(T)
        move_mix.append(move) #track moves that are proposed

        # calculate acceptance ratio
        _, num = posterior_func(T_proposal, Z, Cov, R, sig_o, sig_t, max_range, max_penalty, likelihood_type)

        if denum == 0.0: #this is a degenerate case that occurs due to floating point error
            r_post = 1.
        else:
            r_post = num/denum

        a = r_prop * r_post #acceptance ratio
        p = min(1, a) #probability of acceptance

        if np.random.uniform(0, 1, 1) <= p:
            tp = copy.deepcopy(T_proposal)
            T = tp
            samples.append(copy.deepcopy(tp))
            accepted_moves.append(copy.deepcopy(move)) #keep track of moves that are accepted
            denum = copy.deepcopy(num) #update the posterior of the current sample
        else:
            samples.append(copy.deepcopy(T))

    return samples

def draw_sample(topo):
    '''Given a starting topology, draw a proposal topology and return the proposal ratio.
    Done by randonly choosing a merge or split move among the partitions in the topology.
    Algorithm 2.
    Input:
     - T (list of tuples) topology
    Output:
     - prop (list of tuples) update topology
     - prop_ratio (float) proposal ratio
     - move_type (int 1 or 2) integer flag for move type proposed
    '''
    T = copy.copy(topo)
    if np.random.uniform(0, 1, 1) <= 0.5:
        # Perform a merge move, move_type flag == 1
        move_type = 1

        if len(T) == 1: # if there is only one set, cannot merge
            prop = T
            prop_ratio = 1.
        else:
            # find the total number of mergeable pairs of sets, Nm
            Nm = len(list(itertools.combinations(T, 2))) #simply enumerate all possible pairs

            #select P and Q, the sets to merge
            Pid = random.randrange(len(T)) #select a set at random
            P = T[Pid]
            T.pop(Pid)

            Qid = random.randrange(len(T)) #select another set at random
            Q = T[Qid]
            T.pop(Qid)

            # make the merge proposal
            merge = []
            for elem in P:
                merge.append(elem)
            for elem in Q:
                merge.append(elem)

            T.append(tuple(merge))

            # Find the total number of splits in proposal topology, Ns
            Ns = 0.
            for elem in T:
                if len(elem) > 1:
                    Ns += 1.

            # calculate the proposal ratio
            prop = T
            prop_ratio = Nm * 1./(Ns * stirling(len(merge), 2))
    else:
        # Perform a split move, move_type flag == 2
        move_type = 2

        # find the total number of splits Ns
        Ns = 0.
        for elem in T:
            if len(elem) > 1:
                Ns += 1.

        # select R and get P and Q
        options = []
        for i, r in enumerate(T):
            if len(r) <= 1:
                pass
            else:
                options.append(i)

        if Ns < 1:
            # there are only singleton sets, propose the same topo
            prop = T
            prop_ratio = 1.
        else:
            # there are at least options
            Rid = options[random.randrange(0, len(options))] #choose a random index from the valid options
            R = T[Rid]
            T.pop(Rid)
            splitid = random.randrange(len(R))
            P = R[:splitid]
            Q = R[splitid:]
            if len(P) > 0:
                T.append(tuple(P))
            if len(Q) > 0:
                T.append(tuple(Q))

            # find the total number of sets in proposal topology, Nm
            if len(T) > 1:
                Nm = len(list(itertools.combinations(T, 2)))
                prop_ratio = 1./Nm * Ns * stirling(len(R), 2)
            else:
                Nm = 1.
                prop_ratio = 0.0 #1./Nm * Ns * stirling(len(R), 2)
            prop = T

    return prop, prop_ratio, move_type

def posterior(T, data, cov, R=10, sig_o=0.1, sig_t=0.1, max_range=3., max_penalty=80., likelihood_type='Mahalanobis'):
    '''Given a topology and current observations,
    provide the posterior probability P(T|Z) = P(T)*P(Z|T)
    where P(Z|T) = int_X (P(Z|X,T)*P(X|T))
    Input:
     - T (topology)
     - data (nx2 array of x,y coordinates)
     - cov (nx2 array of x,y covariance)
     - R (int) number of samples in the importance sampler to draw
     - sig_o (float) standard deviations for odom constraints
     - sig_t (float) standard deviations for topo constraints
     - max_range (float) radius for penalty to apply
     - max_penalty (float) value for penalty to apply
     - likelihood_type (string) one of Bhattacharyya, Xodom, or Mahalanobis. Changes the optimization function.
    Output:
     - x_i (list of floats) samples drawn from the X* gaussian
     - approx (float) posterior probability for the topology given the data'''

    # calculate the prior P(T)
    prior = 1 #noninformative uniform prior on all topologies

    #data based likelihood ratios can be calculated outside of the approximation scheme
    diff = 0
    if likelihood_type == 'Xodom':
        #adds a term that inspects metric odom of grouped landmarks
        for partition in T:
            if len(partition) > 1:
                pairs = list(itertools.combinations(partition, 2))
                for p in pairs:
                    diff += (dist(data[p[0]], data[p[1]])/sig_t)**2
    elif likelihood_type == 'Bhattacharyya' or likelihood_type == 'Mahalanobis':
        #adds a term that inspects the distance between the distributions of grouped landmarks
        #uses Bhattacharyya distance
        for partition in T:
            if len(partition) > 1:
                pairs = list(itertools.combinations(partition, 2))
                for p in pairs:
                    diff += ((get_b_distance(data[p[0]], data[p[1]], cov[p[0]], cov[p[1]])/sig_t)**2)[0,0]

    # calculate the likelihood P(Z|T)
    def likelihood_approximation(x):
        '''approximates negative log likelihood for P(Z|T) for the optimization regime'''
        xstar_diff = 0
        temp = x.reshape(len(x)/2, 2)
        if likelihood_type is None:
            for partition in T:
                if len(partition) > 1:
                    if likelihood_type is None: #vanilla PTM
                        pairs = list(itertools.combinations(partition, 2))
                        for p in pairs:
                            xstar_diff += (np.linalg.norm(temp[p[0]]-temp[p[1]])/sig_t)**2
                    else: #covPTM, use Mahalanobis instead
                        pairs = list(itertools.permutations(partition, 2))
                        for p in pairs:
                            xstar_diff += ((get_m_distance(temp[p[0]], data[p[1]], cov[p[0]])/sig_t)**2)[0,0]
        if likelihood_type is None: #vanilla PTM
            return (np.linalg.norm((x - data.flatten()))/sig_o)**2 + xstar_diff + diff
        else: #covPTM, use Mahalanobis instead
            m_dist = []
            for i in xrange(len(x)/2):
                m_dist.append(get_m_distance(temp[i], data[i], cov[i]))
            return (np.linalg.norm(m_dist)/sig_o)**2 + xstar_diff + diff

    #apply the LM Algorithm to get X* (the optimized location of the landmarks given the topology)
    #and the covariance matrix from the optimization
    sol = minimize(likelihood_approximation, x0=data.flatten(), jac=False)
    x_star = sol.x
    covariance = sol.hess_inv

    #define the gaussian Q(X|Z,T) (which drops some uncertainty around the X* optimized landmarks)
    mu_q = x_star
    cov_q = covariance

    #draw N samples from the Gaussian
    x_i = []
    q = []
    mvn = multivariate_normal(mean=mu_q, cov=cov_q, allow_singular=True)
    for j in xrange(R):
        samp = mvn.rvs()
        x_i.append(samp)
        samp = np.matrix(samp)
        q.append(mvn.pdf(samp))

    #calculate the approximation. Equation following the Q function above.
    approx = 0.
    q = np.array(q) / np.linalg.norm(q)
    for j in xrange(R):
        obs_mod = likelihood_approximation(x_i[j])
        if likelihood_type == None: #vanilla PTM
            loc_prior = location_prior(x_i[j].reshape(len(x_i[j])/2, 2), T, max_range, max_penalty)
        else: #covPTM, use Mahalanobis instead
            loc_prior = location_prior(x_i[j].reshape(len(x_i[j])/2, 2), T, max_range, max_penalty, distance_type='Mahalanobis', cov=cov, data=data)
        q_prob = q[j]
        approx += np.exp(-obs_mod)*np.exp(-loc_prior) / q_prob
    return x_i, approx/float(R)

def location_prior(X, T, max_range, max_penalty, distance_type=None, cov=None, data=None):
    '''Gives the negative log likelihood of the location prior, P(X|T). Equation 4.
    Input:
     - X (list of tuples) list of landmark locations
     - T (list of tuples) list of landmark indices
     - max_range (float)
     - max_penalty (float)
     - distance_type (string) can specify whether to use Mahalanobis
     - cov (list of matrices) covariance matrix, needed if distance type is Mahalanobis
     - data (list of landmark locations) needed if distance type is Mahalanobis
    Output:
     - s (float) negative log likelihood'''
    summer = 0.
    for i in range(len(X)):
        for partition in T: #each set
            if i in partition: #if the query landmark is in the set, pass
                pass
            else:
                for j in partition:
                    if distance_type is None:
                        if i < j: #calculate the penalty between the two
                            summer += penalty(X[i], X[j], max_range, max_penalty)
                    else:
                        summer += m_penalty(X[i], data[j], cov[j], max_range, max_penalty)
    return summer

def penalty(x1, x2, D, max_penalty):
    ''' penalty function for two locations'''
    d = dist(x1, x2)
    if d < D:
        return -max_penalty/D**2 * (d**2-D**2) # quadratic function
    else:
        return 0.

def m_penalty(x1, x2, cov1, D, max_penalty):
    ''' penalty function for two locations with respect to Mahalanobis distance '''
    d = get_m_distance(x1, x2, cov1)
    if d < D:
        return -max_penalty/D**2 * (d**2-D**2) # quadratic function
    else:
        return 0.

def dist(x1, x2):
    '''Calculates the euclidean distance between two points'''
    return np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)

def get_b_distance(pos1, pos2, cov1, cov2):
    '''Returns the Bhattacharyya distance'''
    pos_1 = np.matrix([[pos1[0]], [pos1[1]]])
    pos_2 = np.matrix([[pos2[0]], [pos2[1]]])
    cov_1 = np.matrix(cov1)
    cov_2 = np.matrix(cov2)
    sig = (cov_1 + cov_2)/2
    mh = 1/8. * np.matmul((pos_1-pos_2).T, np.matmul(np.linalg.inv(sig), (pos_1-pos_2)))
    return mh + 0.5*np.log(np.linalg.det(sig)/(np.sqrt(np.linalg.det(cov1)*np.linalg.det(cov2))))

def get_m_distance(pos1, pos2, cov1):
    '''Returns the Mahalanobis distance'''
    pos_1 = np.matrix([[pos1[0]],[pos1[1]]])
    pos_2 = np.matrix([[pos2[0]],[pos2[1]]])
    M_dist = np.sqrt(np.fabs(np.matmul((pos_1 - pos_2).T,np.matmul(np.linalg.inv(cov1), (pos_1 - pos_2)))))
    return M_dist

def stirling(n, m):
    '''Recursive function that returns the Stirling number'''
    row = [1]+[0 for _ in xrange(m)]
    for i in xrange(1, n+1):
        new = [0]
        for j in xrange(1, m+1):
            sling = (i-1) * row[j] + row[j-1]
            new.append(sling)
        row = new
    return row[m]

def eigsorted(cov):
    '''Helper function for extracting eigenstuff from covariance matrix'''
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def plot_detections(T, Z, Cov=None):
    '''Helper function for plotting detections
    Input:
    - T (topology)
    - Z (odom points)
    - Cov (covariance matrix)
    Output:
    - fig object
    '''
    map_fig, map_ax = plt.subplots()
    for p in T:
        c = 'r'
        for v in p:
            map_ax.scatter(Z[v][0], Z[v][1], c=c, lw=0, s=100)
            map_ax.annotate(str(v), (Z[v][0], Z[v][1]))

            if Cov is not None:
                vals, vecs = eigsorted(Cov[v])
                theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                w, h = 2 * np.sqrt(np.fabs(vals))
                error = Ellipse(xy=(Z[v][0], Z[v][1]),
                                width=w, height=h,
                                angle=theta, color='gray',
                                lw=1, edgecolor='black', alpha=0.5)
                map_ax.add_patch(error)
    map_ax.axis('auto')
    return map_fig

def plot_report(samples, Z):
    '''Helper function to plot a report of sampler results
    Input:
    - samples (results of sampler)
    - Z (input data detections)
    Return:
    - fig object
    '''

    fig = plt.figure(figsize=(21, 3))
    fig.subplots_adjust(left=0.025, bottom=0.2, right=0.975, top=0.9, wspace=0.3, hspace=None)
    ax1, ax2, ax3, ax4, ax5 = fig.subplots(1, 5)
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax5.clear()
    ax1.set_title("Most Sampled Topology")
    ax2.set_title("2nd Most Sampled Topology")
    ax3.set_title("3rd Most Sampled Topology")
    ax4.set_title("Histogram")
    ax5.set_title("Trace of Dimensionality")

    dist = {}
    x = []
    y = []
    for ii, sample in enumerate(samples):
        key = []
        dim = len(sample)
        for p in sample:
            key.append(tuple(sorted(p)))
        if tuple(sorted(key)) not in dist:
            dist[tuple(sorted(key))] = 1
        else:
            dist[tuple(sorted(key))] += 1
        y.append(dim)
        x.append(ii)
    ax5.plot(x, y)
    ax5.set_xlabel("Sample")
    ax5.set_ylabel("Dimensionality")

    sorted_dist = sorted(dist.items(), key=operator.itemgetter(1), reverse=True)

    axs = [ax1, ax2, ax3]
    bns = []

    for m, ax in zip(sorted_dist[:len(axs)], axs):
        print m
        ids = m[0]
        bns.append(m[1])
        for p in ids:
            c = np.random.rand(3)
            for v in p:
                ax.scatter(Z[v][0], Z[v][1], c=c, lw=0, s=100)
                ax.annotate(str(v), (Z[v][0], Z[v][1]))
        ax.axis('square')
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

    ax4.bar([1, 2, 3], bns, align='center', alpha=0.5)
    ax4.set_xlabel("Topology")
    ax4.set_ylabel("Number of Samples")

    return fig

def simulation_example_1():
    ''' Returns data for a simulated trial'''
    odom_measurements = [(-10.467772208416127, -9.267101431830813),
                         (44.40439635357341, -7.849687577954248),
                         (24.533711628964546, 11.804860154542427),
                         (9.21445182203307, 11.13683525805399),
                         (9.349162990214449, 24.922904237398463),
                         (-10.837796686363788, 44.86831199155543),
                         (45.17713215735204, 47.47439396916004),
                         (24.90889854281831, 27.492346696265876),
                         (-9.243460447514865, 50.36502304370084),
                         (10.876652537175433, 29.526947651900084),
                         (-7.783159943642355, -2.666329647464408),
                         (12.181733088721634, 16.34468181722762),
                         (48.29658608531393, 0.9460573032092793),
                         (28.14631967433353, 18.58041124614543),
                         (26.151753597574928, 19.729054095978732),
                         (11.151753597574928, 19.729054095978732),
                         (-8.848246402425072, 54.72905409597873),
                         (26.642794184747178, 19.919371434106125),
                         (11.642794184747178, 19.91937143410612),
                         (-8.35720581525282, 54.919371434106125)]
    
    inv_cov = [np.matrix([[0.72616167, 0.09660794], [0.09660794, 0.38661148]]),
               np.matrix([[0.9468823, -0.08812443], [-0.08812443, 1.04969224]]),
               np.matrix([[1.56317985, -0.03002452], [-0.03002452, 1.65910769]]),
               np.matrix([[1.20431606, -0.01869779], [-0.01869779, 0.69160073]]),
               np.matrix([[0.08885969, 0.00998019], [0.00998019, 0.1373896]]),
               np.matrix([[0.06382072, -0.06212953], [-0.06212953, 0.19367018]]),
               np.matrix([[0.95987927, 0.25281245], [0.25281245, 1.04857155]]),
               np.matrix([[2.27983969, -0.04454075], [-0.04454075, 2.04492279]]),
               np.matrix([[0.93314424, -0.20902045], [-0.20902045, 1.00467577]]),
               np.matrix([[2.00390687, -0.01259922], [-0.01259922, 1.80917297]]),
               np.matrix([[1.15286761, -0.16409394], [-0.16409394, 1.55032107]]),
               np.matrix([[0.79755027, 0.00498154], [0.00498154, 1.04822743]]),
               np.matrix([[0.80252022, -0.07481858], [-0.07481858, 0.35199254]]),
               np.matrix([[0.3521588, 0.07641365], [0.07641365, 0.22908545]]),
               np.matrix([[0.01565774, 0.00321377], [0.00321377, 0.01762144]]),
               np.matrix([[2.05445127e-02, 1.09914160e-05], [1.09914160e-05, 2.00002219e-02]]),
               np.matrix([[0.0044985, -0.00577761], [-0.00577761, 0.01784661]]),
               np.matrix([[0.01571637, 0.00294462], [0.00294462, 0.01797584]]),
               np.matrix([[2.00754538e-02, 4.95164566e-06], [4.95164566e-06, 2.00003250e-02]]),
               np.matrix([[0.00462037, -0.00596687], [-0.00596687, 0.01768502]])]

    cov = [np.linalg.inv(mat) for mat in inv_cov]

    return odom_measurements, cov

def simulation_example_2():
    odom_measurements = [(-10.25011981781375, -10.867139937205843),
                         (44.82968801265914, -11.095751421390013),
                         (24.78673822329393, 9.041253510818791),
                         (9.831421196187408, 9.112354875759529),
                         (10.117741526255013, 24.191997809756263),
                         (-10.230064771616021, 44.25159603969159),
                         (45.860432635884806, 43.83852713057017),
                         (25.47636776956681, 23.825538964573397),
                         (-8.5659164427737, 43.29761571540823),
                         (11.72006819882726, 23.471804019952756),
                         (-6.79423055281713, -12.05975400200347),
                         (13.115045045664077, 7.9125677623997355),
                         (48.3398581731764, -12.959944305534496),
                         (28.724878857984717, 6.89895002031602),
                         (48.734965689761516, 41.116350892135834),
                         (28.843505379945057, 21.171937909492172)]

    inv_cov = [np.matrix([[ 1.52170309, -0.42762997],  [-0.42762997,  2.68961336]]),
               np.matrix([[1.19059905, 0.21017101],  [0.21017101, 1.02434673]]),
               np.matrix([[ 0.54091815, -0.00679198],  [-0.00679198,  0.55154569]]),
               np.matrix([[0.81533478, 0.01124323],  [0.01124323, 1.34444783]]),
               np.matrix([[10.07674711, -0.74871901],  [-0.74871901,  6.85936784]]),
               np.matrix([[22.22672337,  6.8928638 ],  [ 6.8928638,   7.26462667]]),
               np.matrix([[ 0.9171395,  -0.17726104],  [-0.17726104,  0.94551484]]),
               np.matrix([[0.56847522, 0.01699988],  [0.01699988, 0.53500156]]),
               np.matrix([[0.85929543, 0.09496993],  [0.09496993, 0.85258429]]),
               np.matrix([[ 0.60644461, -0.02594034],  [-0.02594034,  0.61768986]]),
               np.matrix([[ 1.15947536, -0.156794  ],  [-0.156794,    0.95458832]]),
               np.matrix([[ 0.47603398, -0.00202755],  [-0.00202755,  0.55813899]]),
               np.matrix([[1.74526947, 0.95612491],  [0.95612491, 3.73369404]]),
               np.matrix([[0.55229765, 0.07725214],  [0.07725214, 1.15927535]]),
               np.matrix([[ 31.27482976, -11.46322213],  [-11.46322213,  12.6029932]]),
               np.matrix([[12.5852041,   0.13769225],  [ 0.13769225, 11.12854484]])]

    cov = [np.linalg.inv(mat) for mat in inv_cov]

    return odom_measurements, cov

def adversarial_world(option):
    if option == 1:
        odom_measurements = [(5., 5.), (0., 0.), (2., 0.), (7., 5.), (4., 0.), (9., 5.)]
        cov = [np.matrix([[30., 29.9], [29.9, 30.]]),
               np.matrix([[30., 29.9], [29.9, 30.]]),
               np.matrix([[30., 29.9], [29.9, 30.]]),
               np.matrix([[30., 29.9], [29.9, 30.]]),
               np.matrix([[30., 29.9], [29.9, 30.]]),
               np.matrix([[30., 29.9], [29.9, 30.]])]

    elif option == 2:
        odom_measurements = [(10., 10.), (0., 0.), (10., 0.), (0., 10.), (2.5, 2.5), (7.5, 7.5), (2.5, 7.5), (7.5, 2.5)]
        cov = [np.matrix([[20., 18.], [18., 20.]]),
               np.matrix([[20., 18.], [18., 20.]]),
               np.matrix([[20., -18.], [-18., 20.]]),
               np.matrix([[20., -18.], [-18., 20.]]),
               np.matrix([[1., 0.], [0., 1.]]),
               np.matrix([[1., 0.], [0., 1.]]),
               np.matrix([[1., 0.], [0., 1.]]),
               np.matrix([[1., 0.], [0., 1.]])]

    else:
        odom_measurements = [(10., 10.), (0., 0.), (10., 0.), (0., 10.), (2.5, 2.5), (12.5, 7.5), (2.5, 7.5), (12.5, 2.5)]
        cov = [np.matrix([[1., 0.], [0., 1.]]),
               np.matrix([[1., 0.], [0., 1.]]),
               np.matrix([[1., 0.], [0., 1.]]),
               np.matrix([[1., 0.], [0., 1.]]),
               np.matrix([[10., 9.], [9., 10.]]),
               np.matrix([[10., -9.], [-9., 10.]]),
               np.matrix([[10., -9.], [-9., 10.]]),
               np.matrix([[10., 9.], [9., 10.]])]

    return odom_measurements, cov


if __name__ == '__main__':
    # simple PTM test cases for tuning
    # odom_measurements = [(10, 0), (10, 10), (0, 10), (0, 0), (9, 0), (9, 9)]
    # odom_measurements = [(10, 0), (10, 10)]
    # odom_measurements = [(10, 10),(9,9),(10,9),(9,10),(10,10)]


    ###################
    # Simulated Data Example 1
    ###################
    # odom_measurements, cov = simulation_example_1()

    ###################
    # Simulated Data Example 2
    ###################
    # odom_measurements, cov = simulation_example_2()

    ###################
    # Adversarial World
    ###################
    odom_measurements, cov = adversarial_world(1)


    # proposal_topology = [tuple([i for i in range(len(odom_measurements))])] #group everything together
    proposal_topology = [tuple([i]) for i in range(0, len(odom_measurements))] #group everything seperately

    # # test draw sample
    # print 'Testing Draw Sample'
    # prop, ratio, move_type = draw_sample(proposal_topology)
    # print prop, ratio, move_type

    # # test location prior
    # print 'Testing Location Prior'
    # print location_prior(odom_measurements, prop, 2, 10)

    # # test posterior
    # print 'Testing Posterior Function'
    # gaussian_sample, approx = posterior(prop, np.array(odom_measurements), cov, likelihood_type=None)
    # print approx

    ###################
    # Set Parameters
    ###################
    sig_o = 2.5
    sig_t = 1.5
    max_range = 2.0
    max_penalty = 50.0
    R = 30
    number_of_samples = 3000

    ###################
    # Sample!
    ###################

    #vanilla PTM
    ptm = PTM(T=proposal_topology,
              Z=np.array(odom_measurements),
              posterior_func=posterior,
              niter=number_of_samples,
              R=R,
              sig_o=sig_o,
              sig_t=sig_t,
              max_range=max_range,
              max_penalty=max_penalty,
              visualize=True,
              filename='ptm_samples')

    #covPTM
    extended_ptm = covPTM(T=proposal_topology,
                          Z=np.array(odom_measurements),
                          Cov=cov,
                          posterior_func=posterior,
                          niter=number_of_samples,
                          R=R,
                          sig_o=sig_o,
                          sig_t=sig_t,
                          max_range=max_range,
                          max_penalty=max_penalty,
                          likelihood_type='Mahalanobis',
                          visualize=True,
                          filename='cptm_samples')
