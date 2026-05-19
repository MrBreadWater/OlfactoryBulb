TITLE fitted sum-of-decays kainate receptor synapse

COMMENT
Event-driven glutamate-gated kainate receptor current.

This file was generated from notebooks/kar_kernel_approximation.ipynb.
The kernel is a conductance-objective fit to the Frerking and Ohliger-Frerking
KAR EPSP after passive analytical inversion:

    g_K(t) / g_L ~= sum_i amp_i * exp(-t / tau_i)

Use kd = 0 for the calibrated linear single-event kernel. Positive kd applies
a saturation nonlinearity for sensitivity tests.
ENDCOMMENT

NEURON {
    POINT_PROCESS KainateSyn
    RANGE tau1, tau2, tau3, amp1, amp2, amp3, e, i, g, x, gmax, kd, block
    NONSPECIFIC_CURRENT i
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

PARAMETER {
    tau1 = 6.728726245 (ms) <1e-9,1e9>
    tau2 = 81.75126152 (ms) <1e-9,1e9>
    tau3 = 468.7337682 (ms) <1e-9,1e9>
    amp1 = 0.06942183802 (1)
    amp2 = 0.008503803144 (1)
    amp3 = 0.01280596195 (1)
    gmax = 0 (uS)
    kd = 0 (1)
    block = 1 (1)
    e = 0 (mV)
}

ASSIGNED {
    v (mV)
    i (nA)
    g (uS)
    x (1)
}

STATE {
    D1 D2 D3
}

INITIAL {
    D1 = 0
    D2 = 0
    D3 = 0
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    x = D1 + D2 + D3
    if (x < 0) {
        x = 0
    }
    if (kd > 0) {
        g = gmax * block * x / (kd + x)
    } else {
        g = gmax * block * x
    }
    i = g * (v - e)
}

DERIVATIVE state {
    D1' = -D1/tau1
    D2' = -D2/tau2
    D3' = -D3/tau3
}

NET_RECEIVE(weight) {
    D1 = D1 + weight*amp1
    D2 = D2 + weight*amp2
    D3 = D3 + weight*amp3
}
