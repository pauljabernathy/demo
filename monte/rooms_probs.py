import pandas as pd
from constants import *


def compute_probs_original():
    print("\n\n\ncompute_probs3()")

    assert kp == 1
    assert pk + sk + dk == 1
    assert ks + fs == 1
    assert sf + ef == 1
    assert fe + de == 1
    assert kd + hd + ed == 1
    assert dh + rh == 1
    assert hr + tr == 1
    assert rt == 1
    '''

    Z = pk - 1
    W = (ks / Z) + (1 / sk)
    V = (-Z * W / (kd * fs)) * ( (sf * fs) / (sk * W ) + 1 )
    T = (1 - fe * (ed * V + ef) ) / fs
    U = (1 / (hd * V)) * (1 - rh*hr / (1 - tr * rt))

    print('Z', Z)
    print('W', W)
    print('V', V)
    print('T', T)
    print('U', U)


    print("\n\n")

    # p + k + s + f + e + d + h + r + t == 1

    p_term = pk * (T - sf) / sk
    k_term = (T - sf) / sk
    s_term = T
    f_term = 1
    e_term = (ed * V + ef)
    d_term = V
    h_term = U
    r_term = rh / (U * (1 - tr * rt))
    t_term = tr * rh / (U * (1 - tr * rt))

    print("p_term", p_term)
    print("k_term", k_term)
    print("s_term", s_term)
    print("f_term", f_term)
    print("e_term", e_term)
    print("d_term", d_term)
    print("h_term", h_term)
    print("r_term", r_term)
    print("t_term", t_term)


    #assert p_term == k_term * pk
    assert t_term == r_term * tr

    print("sum of terms =", p_term + k_term + s_term + f_term + e_term + d_term + h_term + r_term + t_term)
    one_over_f = p_term + k_term + s_term + f_term + e_term + d_term + h_term + r_term + t_term
    f = 1 / one_over_f
    print(f)

    k = k_term * f
    p = p_term * f
    s = s_term * f
    e = e_term * f
    d = d_term * f
    h = h_term * f
    r = r_term * f
    t = t_term * f

    print('p', p)
    print('k', k)
    print('s', s)
    print('f', f)
    print('e', e)
    print('d', d)
    print('h', h)
    print('r', r)
    print('t', t)

    print(p + k + s + f + e + d + h + r + t)
    '''


    # Try again
    print("\n\n--\nsecond try")
    A = 1 - rt * tr
    B = 1 - hr * rh / A
    C = 1 - hd * dh / B
    G = 1 - kp * pk

    J = (sk * ks - 1) / G
    L = (fs * sf / J) + 1
    M = fs * sk * kd / G + fe * ed
    V = (L - fe * ef) / M
    T = (1 - fe * (ef + ed * V)) / fs
    print('T', T)
    #T = .78

    print("A", A)
    print("B", B)
    print("C", C)
    print("G", G)
    print('J', J)
    print('M', M)
    print('V', V)
    print('T', T)

    #p_term = pk * (sf - T) / sk
    #k_term = (sf - T) / sk
    k_term = (T - sf) / sk
    p_term = pk * k_term
    s_term = T
    f_term = 1
    e_term = (ed * V + ef)
    d_term = V
    h_term = hd * V / B
    r_term = rh * hd * V / (A * B)
    t_term = tr * r_term

    print("p_term", p_term)
    print("k_term", k_term)
    print("s_term", s_term)
    print("f_term", f_term)
    print("e_term", e_term)
    print("d_term", d_term)
    print("h_term", h_term)
    print("r_term", r_term)
    print("t_term", t_term)

    print("sum of terms =", p_term + k_term + s_term + f_term + e_term + d_term + h_term + r_term + t_term)
    one_over_f = p_term + k_term + s_term + f_term + e_term + d_term + h_term + r_term + t_term
    f = 1 / one_over_f
    print(f)

    k = k_term * f
    p = p_term * f
    s = s_term * f
    e = e_term * f
    d = d_term * f
    h = h_term * f
    r = r_term * f
    t = t_term * f

    print("calculated values")
    print('p', p)
    print('k', k)
    print('s', s)
    print('f', f)
    print('e', e)
    print('d', d)
    print('h', h)
    print('r', r)
    print('t', t)

    print(p + k + s + f + e + d + h + r + t)

    print("from the original equations")
    p2 = pk * k
    k2 = ks * s + kd * d + kp * p
    s2 = sk * k + sf * f
    f2 = fs * s + fe * e
    e2 = ef * f + ed * d
    d2 = de * d + dh * h + dk * k
    h2 = hd * d + hr * r
    r2 = rh * h + rt * t
    t2 = tr * r
    print('p', p2)
    print('k', k2)
    print('s', s2)
    print('f', f2)
    print('e', d2)
    print('d', d2)
    print('h', h2)
    print('r', r2)
    print('t', t2)
    print(p2 + k2 + s2 + f2 + e2 + d2 + h2 + r2 + t2)


    rooms = ["pantry", "kitchen", "school", "office", "entry", "den", "hall", "bedroom", "bathroom"]
    prob_list = [p, k, s, f, e, d, h, r, t]
    prob_list2 = [p2, k2, s2, f2, e2, d2, h2, r2, t2]
    probs = pd.DataFrame({"room": rooms, "probability": prob_list, 'other_prob': prob_list2})
    print(probs)
    print(probs.sort_values('probability', ascending=False))
    print(probs.probability.sum())


    return probs


def compute_probs2():
    """
    fewer rooms; only kitchen, school room, office, entry hall, and den
    :return:
    """

    # First, we need to change some of the transition probabilities
    sk = .4
    ed = .5
    kd = .5
    assert sk + dk == 1
    assert ks + fs == 1
    assert sf + ef == 1
    assert fe + de == 1
    assert kd + ed == 1

    N = sk * ks - 1
    L = (sf * fs / N) + 1
    M = fe * ed - (fs * sk * kd / N )
    V = (L - fe * ef) / M
    T = (1 - fe * (ef + ed * V)) / fs

    print('N', N)
    print('L', L)
    print('M', M)
    print('V', V)
    print('T', T)

    k_term = (T - sf) / sk
    s_term = T
    e_term = ef + ed * V
    d_term = V

    print()
    print('k_term', k_term)
    print('s_term', s_term)
    print('e_term', e_term)
    print('d_term', d_term)

    one_over_f = k_term + s_term + 1 + e_term + d_term
    f = 1 / one_over_f

    k = k_term * f
    s = s_term * f
    e = e_term * f
    d = d_term * f

    print()
    print('k', k)
    print('s', s)
    print('f', f)
    print('e', e)
    print('d', d)
    print('sum', k + s + f + e + d)

    k2 = ks * s + kd * d
    s2 = sk * k + sf * f
    f2 = fs * s + fe * e
    e2 = ef * f + ed * d
    d2 = de * e + dk * k

    print()
    print('k2', k2)
    print('s2', s2)
    print('f2', f2)
    print('e2', e2)
    print('d2', d2)
    print('sum', k2 + s2 + f2 + e2 + d2)

    print()
    print(k == k2)
    print(s == s2)
    print(f == f2)
    print(e == e2)
    print(d == d2)

    print()
    print(diff_ratio(k, k2))
    print(diff_ratio(s, s2))
    print(diff_ratio(f, f2))
    print(diff_ratio(e, e2))
    print(diff_ratio(d, d2))

    k3 = ks * s2 + kd * d2
    s3 = sk * k2 + sf * f2
    f3 = fs * s2 + fe * e2
    e3 = ef * f2 + ed * d2
    d3 = de * e2 + dk * k2

    print()
    print('k3', k3)
    print('s3', s3)
    print('f3', f3)
    print('e3', e3)
    print('d3', d3)
    print('sum', k3 + s3 + f3 + e3 + d3)

    k4 = ks * s3 + kd * d3
    s4 = sk * k3 + sf * f3
    f4 = fs * s3 + fe * e3
    e4 = ef * f3 + ed * d3
    d4 = de * e3 + dk * k3

    print()
    print('k4', k4)
    print('s4', s4)
    print('f4', f4)
    print('e4', e4)
    print('d4', d4)
    print('sum', k4 + s4 + f4 + e4 + d4)

    rooms = ["kitchen", "school", "office", "entry", "den"]
    prob_list = [k, s, f, e, d]
    prob_list2 = [k2, s2, f2, e2, d2]
    probs = pd.DataFrame({"room": rooms, "probability": prob_list, 'other_prob': prob_list2})
    print(probs)
    print(probs.sort_values('probability', ascending=False))
    print(probs.probability.sum())

    return probs


def diff_ratio(num1, num2):
    return (num1 - num2) / num1


def compute_probs():
    #print("\n\n\ncompute_probs()")
    A = 1 - rt * tr
    B = 1 - hr * rh / A
    C = 1 - hd * dh / B
    G = 1 - kp * pk

    J = (sk * ks / G) - 1
    L = fs * sf / J + 1
    M = fe * ed - (fs * sk * kd) / (J * G)
    V = (L - fe * ef) / M
    N = ef + ed * V
    T = (1 - fe * N) / fs

    '''print("A", A)
    print("B", B)
    print("C", C)
    print("G", G)
    print('J', J)
    print('M', M)
    print('V', V)
    print('T', T)'''

    # p_term = pk * (sf - T) / sk
    # k_term = (sf - T) / sk
    k_term = (T - sf) / sk
    p_term = pk * k_term
    s_term = T
    f_term = 1
    e_term = N
    d_term = V
    h_term = hd * V / B
    r_term = rh * hd * V / (A * B)
    t_term = tr * r_term

    '''print("p_term", p_term)
    print("k_term", k_term)
    print("s_term", s_term)
    print("f_term", f_term)
    print("e_term", e_term)
    print("d_term", d_term)
    print("h_term", h_term)
    print("r_term", r_term)
    print("t_term", t_term)'''

    #print("sum of terms =", p_term + k_term + s_term + f_term + e_term + d_term + h_term + r_term + t_term)
    one_over_f = p_term + k_term + s_term + f_term + e_term + d_term + h_term + r_term + t_term
    f = 1 / one_over_f
    #print(f)

    k = k_term * f
    p = p_term * f
    s = s_term * f
    e = e_term * f
    d = d_term * f
    h = h_term * f
    r = r_term * f
    t = t_term * f

    '''print("calculated values")
    print('p', p)
    print('k', k)
    print('s', s)
    print('f', f)
    print('e', e)
    print('d', d)
    print('h', h)
    print('r', r)
    print('t', t)'''

    #print(p + k + s + f + e + d + h + r + t)

    #print("from the original equations")
    p2 = pk * k
    k2 = ks * s + kd * d + kp * p
    s2 = sk * k + sf * f
    f2 = fs * s + fe * e
    e2 = ef * f + ed * d
    d2 = de * e + dh * h + dk * k
    h2 = hd * d + hr * r
    r2 = rh * h + rt * t
    t2 = tr * r
    '''print('p', p2)
    print('k', k2)
    print('s', s2)
    print('f', f2)
    print('e', d2)
    print('d', d2)
    print('h', h2)
    print('r', r2)
    print('t', t2)
    print(p2 + k2 + s2 + f2 + e2 + d2 + h2 + r2 + t2)

    print("diff ratios:")
    print('p', diff_ratio(p, p2))
    print('k', diff_ratio(k, k2))
    print('s', diff_ratio(s, s2))
    print('f', diff_ratio(f, f2))
    print('e', diff_ratio(e, e2))
    print('d', diff_ratio(d, d2))
    print('h', diff_ratio(h, h2))
    print('r', diff_ratio(r, r2))
    print('t', diff_ratio(t, t2))'''


    rooms = ["pantry", "kitchen", "school", "office", "entry", "den", "hall", "bedroom", "bathroom"]
    prob_list = [p, k, s, f, e, d, h, r, t]
    prob_list2 = [p2, k2, s2, f2, e2, d2, h2, r2, t2]
    #probs = pd.DataFrame({"room": rooms, "probability": prob_list, 'other_prob': prob_list2})
    probs = pd.DataFrame({"room": rooms, "probability": prob_list})
    #print(probs)
    probs = probs.sort_values('probability', ascending=False)
    #print(probs.probability.sum())

    return probs


#compute_probs()
#compute_probs2()
#compute_probs3()
