import funcs
vL: int = 1024
EmDim: int = 16
ContextSize: int = 48
KQDim: int = 8
L: int = 4
npL: list[int] = [16, 24, 24, 16]
DA: str = 'C:/Users/tatar/TTPuKoJIbI/projU/T/'

m = open('C:/Users/tatar/TTPuKoJIbI/Texts/FullT.txt').read().split(sep='\n')
Tokens: list[int] = [int(i) for i in m[:-1]]
KQR: float = KQDim ** 0.5
drop_prt: float = 0.0

for CreatingPars in range(1):
    Em: list[list[float]] = [[0 for em1 in range(EmDim)] for em0 in range(vL)]
    UEm: list[list[float]] = [[0 for uem1 in range(EmDim)] for uem0 in range(vL)]
    K: list[list[list[list[float]]]] = [[[[0 for k3 in range(EmDim)] for k2 in range(KQDim)] for k1 in range(ContextSize)] for k0 in range(L)]
    Q: list[list[list[list[float]]]] = [[[[0 for q3 in range(EmDim)] for q2 in range(KQDim)] for q1 in range(ContextSize)] for q0 in range(L)]
    V: list[list[list[list[float]]]] = [[[[0 for v3 in range(EmDim)] for v2 in range(KQDim)] for v1 in range(ContextSize)] for v0 in range(L)]
    Vo: list[list[list[list[float]]]] = [[[[0 for vo3 in range(KQDim)] for vo2 in range(EmDim)] for vo1 in range(ContextSize)] for vo0 in range(L)]
    W: list[list[list[list[list[float]]]]] = [[[[[0 for w4 in range(npL[w2])] for w3 in range(npL[w2 + 1])] for w2 in range(len(npL) - 1)] for w1 in range(ContextSize)] for w0 in range(L - 1)]
    B: list[list[list[list[float]]]] = [[[[0 for b3 in range(npL[b2])] for b2 in range(len(npL))] for b1 in range(ContextSize)] for b0 in range(L - 1)]

    n: list[list[list[list[float]]]] = [[[[0 for n3 in range(npL[n2])] for n2 in range(len(npL))] for n1 in range(ContextSize)] for n0 in range(L - 1)]
    O: list[float] = []
    EmnA: list[list[list[float]]] = [[[0 for emnA2 in range(EmDim)] for emnA1 in range(ContextSize)] for emnA0 in range(L)]
    EmnM: list[list[list[float]]] = [[[0 for emnM2 in range(EmDim)] for emnM1 in range(ContextSize)] for emnM0 in range(L)]
    EmnO: list[list[float]] = [[0 for emnO1 in range(EmDim)] for emnO0 in range(ContextSize)]
    P: list[list[list[float]]] = [[[0 for p2 in range(ContextSize)] for p1 in range(ContextSize)] for p0 in range(L)]
    DEmn: list[list[list[float]]] = [[[0 for Dems2 in range(EmDim)] for Dems1 in range(ContextSize)] for Dems0 in range(L)]
    Vn: list[list[list[float]]] = [[[0 for vn2 in range(EmDim)] for vn1 in range(ContextSize)] for vn0 in range(L)]
    Kn: list[list[list[float]]] = [[[0 for kn2 in range(KQDim)] for kn1 in range(ContextSize)] for kn0 in range(L)]
    Qn: list[list[list[float]]] = [[[0 for qn2 in range(KQDim)] for qn1 in range(ContextSize)] for qn0 in range(L)]

    drops: list[list[list[list[bool]]]] = [[[funcs.dropping(npL[Dn2], drop_prt) for Dn2 in range(len(npL))] for Dn1 in range(ContextSize)] for Dn0 in range(L - 1)]

    dEm: list[list[float]] = [[0 for dem1 in range(EmDim)] for dem0 in range(vL)]
    dUEm: list[list[float]] = [[0 for due1 in range(EmDim)] for due0 in range(vL)]
    dK: list[list[list[list[float]]]] = [[[[0 for dk3 in range(EmDim)] for dk2 in range(KQDim)] for dk1 in range(ContextSize)] for dk0 in range(L)]
    dQ: list[list[list[list[float]]]] = [[[[0 for dq3 in range(EmDim)] for dq2 in range(KQDim)] for dq1 in range(ContextSize)] for dq0 in range(L)]
    dV: list[list[list[list[float]]]] = [[[[0 for dv3 in range(EmDim)] for dv2 in range(KQDim)] for dv1 in range(ContextSize)] for dv0 in range(L)]
    dVo: list[list[list[list[float]]]] = [[[[0 for dvo3 in range(KQDim)] for dvo2 in range(EmDim)] for dvo1 in range(ContextSize)] for dvo0 in range(L)]
    dW: list[list[list[list[list[float]]]]] = [[[[[0 for dw4 in range(npL[dw2])] for dw3 in range(npL[dw2 + 1])] for dw2 in range(len(npL) - 1)] for dw1 in range(ContextSize)] for dw0 in range(L - 1)]
    dB: list[list[list[list[float]]]] = [[[[0 for db3 in range(npL[db2])] for db2 in range(len(npL))] for db1 in range(ContextSize)] for db0 in range(L - 1)]


def load(ext: str) -> None:
    for i in range(L):
        for o in range(ContextSize):
            for p in range(KQDim):
                mK = open(f'{DA}{ext}/K/{i}_{o}_{p}.txt').read().split(sep='\n')
                mQ = open(f'{DA}{ext}/Q/{i}_{o}_{p}.txt').read().split(sep='\n')
                mV = open(f'{DA}{ext}/V/{i}_{o}_{p}.txt').read().split(sep='\n')
                K[i][o][p] = [float(k) for k in mK[:-1]]
                Q[i][o][p] = [float(k) for k in mQ[:-1]]
                V[i][o][p] = [float(k) for k in mV[:-1]]
            for p in range(EmDim):
                m = open(f'{DA}{ext}/Vo/{i}_{o}_{p}.txt').read().split(sep='\n')
                Vo[i][o][p] = [float(k) for k in m[:-1]]
    for i in range(L - 1):
        for o in range(ContextSize):
            for p in range(len(npL)):
                m = open(f'{DA}{ext}/B/{i}_{o}_{p}.txt').read().split(sep='\n')
                B[i][o][p] = [float(k) for k in m[:-1]]
            for p in range(len(npL) - 1):
                for k in range(npL[p + 1]):
                    m = open(f'{DA}{ext}/W/{i}_{o}_{p}_{k}.txt').read().split(sep='\n')
                    W[i][o][p][k] = [float(l) for l in m[:-1]]
    for i in range(vL):
        m = open(f'{DA}{ext}/Em/{i}.txt').read().split(sep='\n')
        Em[i] = [float(o) for o in m[:-1]]
        m = open(f'{DA}{ext}/UEm/{i}.txt').read().split(sep='\n')
        UEm[i] = [float(o) for o in m[:-1]]
    print('pars loaded')


def upload(ext: str) -> None:
    for i in range(L):
        for o in range(ContextSize):
            for p in range(KQDim):
                open(f'{DA}{ext}/K/{i}_{o}_{p}.txt', 'w').write('\n'.join([str(k) for k in K[i][o][p]]) + '\n')
                open(f'{DA}{ext}/Q/{i}_{o}_{p}.txt', 'w').write('\n'.join([str(k) for k in Q[i][o][p]]) + '\n')
                open(f'{DA}{ext}/V/{i}_{o}_{p}.txt', 'w').write('\n'.join([str(k) for k in V[i][o][p]]) + '\n')
            for p in range(EmDim):
                open(f'{DA}{ext}/Vo/{i}_{o}_{p}.txt', 'w').write('\n'.join([str(k) for k in Vo[i][o][p]]) + '\n')
    for i in range(L - 1):
        for o in range(ContextSize):
            for p in range(len(npL)):
                open(f'{DA}{ext}/B/{i}_{o}_{p}.txt', 'w').write('\n'.join([str(k) for k in B[i][o][p]]) + '\n')
            for p in range(len(npL) - 1):
                for k in range(npL[p + 1]):
                    open(f'{DA}{ext}/W/{i}_{o}_{p}_{k}.txt', 'w').write('\n'.join([str(l) for l in W[i][o][p][k]]) + '\n')
    for i in range(vL):
        open(f'{DA}{ext}/Em/{i}.txt', 'w').write('\n'.join([str(o) for o in Em[i]]) + '\n')
        open(f'{DA}{ext}/UEm/{i}.txt', 'w').write('\n'.join([str(o) for o in UEm[i]]) + '\n')
    print('pars uploaded')


def fpr(stt: int) -> None:
    global O
    for i in range(ContextSize):
        EmnA[0][i] = Em[Tokens[stt + i]]
    for i in range(L - 1):
        for o in range(ContextSize):
            Qn[i][o] = funcs.vm_mpx_wa(EmnA[i][o], Q[i][o])
            for p in range(ContextSize):
                Kn[i][p] = funcs.vm_mpx_wa(EmnA[i][o], K[i][p])
                P[i][o][p] = funcs.vv_mpx(Kn[i][p], Qn[i][o]) / KQR

        for o in range(ContextSize):
            Vn[i][o] = funcs.vm_mpx_wa(funcs.vm_mpx_wa(EmnA[i][o], V[i][o]), Vo[i][o])
            for p in range(ContextSize):
                DEmn[i][p] = funcs.vv_sum(DEmn[i][p], funcs.nv_mpx(P[i][o][p], Vn[i][o]), None)
        for o in range(ContextSize):
            EmnM[i][o] = funcs.vv_sum(EmnA[i][o], DEmn[i][o], None)
            n[i][o][0] = funcs.act_v(funcs.vv_sum(EmnM[i][o], B[i][o][0], drops[i][o][0]))

        for o in range(ContextSize):
            for p in range(len(npL) - 1):
                n[i][o][p + 1] = funcs.act_v(funcs.vv_sum(funcs.vm_mpx(n[i][o][p], W[i][o][p], drops[i][o][p + 1]), B[i][o][p + 1], drops[i][o][p + 1]))
            EmnA[i + 1][o] = n[i][o][-1]

    for i in range(ContextSize):
        Qn[-1][i] = funcs.vm_mpx_wa(EmnA[-1][i], Q[-1][i])
        for o in range(ContextSize):
            Kn[-1][o] = funcs.vm_mpx_wa(EmnA[-1][i], K[-1][o])
            P[-1][i][o] = funcs.vv_mpx(Kn[-1][o], Qn[-1][i]) / KQR
    for i in range(ContextSize):
        Vn[-1][i] = funcs.vm_mpx_wa(funcs.vm_mpx_wa(EmnA[-1][i], V[-1][i]), Vo[-1][i])
        for o in range(ContextSize):
            DEmn[-1][o] = funcs.vv_sum(DEmn[-1][o], funcs.nv_mpx(P[-1][i][o], Vn[-1][i]), None)
    for i in range(ContextSize):
        EmnO[i] = funcs.vv_sum(EmnA[-1][i], DEmn[-1][i], None)

    O = funcs.vm_mpx_wa(EmnO[-1], UEm)


def cost(stt: int) -> float:
    result: float = 0
    for i in range(vL):
        result += abs(O[i])
    result += -abs(O[Tokens[stt + ContextSize]]) + abs(1 - O[Tokens[stt + ContextSize]])
    return result


def bpr(stt: int, c: float) -> None:
    dn: list[list[list[list[float]]]] = [[[[0 for dn3 in range(npL[dn2])] for dn2 in range(len(npL))] for dn1 in range(ContextSize)] for dn0 in range(L - 1)]
    dO: list[float] = [0 for do in range(vL)]
    dEmnA: list[list[list[float]]] = [[[0 for demA2 in range(EmDim)] for demA1 in range(ContextSize)] for demA0 in range(L)]
    dEmnM: list[list[list[float]]] = [[[0 for demM2 in range(EmDim)] for demM1 in range(ContextSize)] for demM0 in range(L)]
    dEmnO: list[list[float]] = [[0 for demO1 in range(EmDim)] for demO0 in range(ContextSize)]
    dP: list[list[list[float]]] = [[[0 for dp2 in range(ContextSize)] for dp1 in range(ContextSize)] for dp0 in range(L)]
    dVn: list[list[list[float]]] = [[[0 for dvn2 in range(EmDim)] for dvn1 in range(ContextSize)] for dvn0 in range(L)]
    dKn: list[list[list[float]]] = [[[0 for dkn2 in range(KQDim)] for dkn1 in range(ContextSize)] for dkn0 in range(L)]
    dQn: list[list[list[float]]] = [[[0 for dqn2 in range(KQDim)] for dqn1 in range(ContextSize)] for dqn0 in range(L)]

    for i in range(vL):
        dO[i] = O[i]
    dO[Tokens[stt + ContextSize]] = 2 - O[Tokens[stt + ContextSize]]
    dEmnO[-1] = funcs.vm_div_wa(dO, UEm)

    for i in range(ContextSize):
        for o in range(EmDim):
            dP[-1][i][-1] += dEmnO[-1][o] * Vn[-1][i][o]
            dVn[-1][i][o] = dEmnO[-1][o] * P[-1][i][-1]
        for o in range(KQDim):
            dKn[-1][-1][o] += dP[-1][i][-1] * Qn[-1][i][o]
            dQn[-1][i][o] = dP[-1][i][-1] * Kn[-1][-1][o]
    for i in range(ContextSize):
        dEmnA[-1][i] = funcs.vv_sum(funcs.vv_sum(dEmnA[-1][i], funcs.vm_div_wa(dQn[-1][i], Q[-1][i]), None), funcs.vm_div_wa(funcs.vm_div_wa(Vn[-1][i], Vo[-1][i]), V[-1][i]), None)
    dEmnA[-1][-1] = funcs.vv_sum(funcs.vv_sum(dEmnA[-1][-1], funcs.vm_div_wa(dKn[-1][-1], K[-1][-1]), None), dEmnO[-1], None)

    for i in range(L - 2, -1, -1):
        for o in range(ContextSize):
            dn[i][o][-1] = dEmnA[i + 1][o]
            for p in range(len(npL) - 2, -1, -1):
                dn[i][o][p] = funcs.vm_div(dn[i][o][p + 1], W[i][o][p], n[i][o][p])
            dEmnM[i][o] = dn[i][o][0]

        for o in range(ContextSize):
            for p in range(ContextSize):
                for k in range(EmDim):
                    dP[i][o][p] += dEmnM[i][p][k] * Vn[i][o][k]
                    dVn[i][o][k] += dEmnM[i][p][k] * P[i][o][p]
                for k in range(KQDim):
                    dKn[i][p][k] += dP[i][o][p] * Qn[i][o][k]
                    dQn[i][o][k] += dP[i][o][p] * Kn[i][p][k]
        for o in range(ContextSize):
            dEmnA[i][o] = funcs.vv_sum(funcs.vv_sum(funcs.vm_div_wa(dKn[i][o], K[i][o]), funcs.vm_div_wa(dQn[i][o], Q[i][o]), None),
                                       funcs.vv_sum(funcs.vm_div_wa(funcs.vm_div_wa(Vn[i][o], Vo[i][o]), V[i][o]), dEmnM[i][o], None), None)

    for i in range(L):
        for o in range(ContextSize):
            dK[i][o] = funcs.mm_sum(dK[i][o], funcs.nm_mpx(c, funcs.vv_div(dKn[i][o], EmnA[i][o])), None)
            dQ[i][o] = funcs.mm_sum(dQ[i][o], funcs.nm_mpx(c, funcs.vv_div(dQn[i][o], EmnA[i][o])), None)
            dVo[i][o] = funcs.mm_sum(dVo[i][o], funcs.nm_mpx(c, funcs.vv_div(dVn[i][o], funcs.vm_mpx_wa(EmnA[i][o], V[i][o]))), None)
            dV[i][o] = funcs.mm_sum(dV[i][o], funcs.nm_mpx(c, funcs.vv_div(funcs.vm_div_wa(dVn[i][o], Vo[i][o]), EmnA[i][o])), None)
    for i in range(L - 1):
        for o in range(ContextSize):
            for p in range(len(npL) - 1):
                dW[i][o][p] = funcs.mm_sum(dW[i][o][p], funcs.nm_mpx(c, funcs.vv_div(dn[i][o][p + 1], dn[i][o][p])), None)
            dB[i][o] = funcs.mm_sum(dB[i][o], funcs.nm_mpx(c, dn[i][o]), None)
    for i in range(ContextSize):
        for o in range(EmDim):
            dEm[Tokens[stt + i]][o] += dEmnA[0][i][o] * c
    global dUEm
    dUEm = funcs.mm_sum(dUEm, funcs.nm_mpx(c, funcs.vv_div(dO, EmnO[-1])), None)


def cng(cof: float) -> None:
    global Em
    global UEm
    for i in range(L):
        for o in range(ContextSize):
            K[i][o] = funcs.mm_sum(K[i][o], funcs.nm_mpx(cof, dK[i][o]), None)
            Q[i][o] = funcs.mm_sum(Q[i][o], funcs.nm_mpx(cof, dQ[i][o]), None)
            Vo[i][o] = funcs.mm_sum(Vo[i][o], funcs.nm_mpx(cof, dVo[i][o]), None)
            V[i][o] = funcs.mm_sum(V[i][o], funcs.nm_mpx(cof, dV[i][o]), None)
    for i in range(2, L - 1):
        for o in range(ContextSize):
            for p in range(len(npL) - 1):
                W[i][o][p] = funcs.mm_sum(W[i][o][p], funcs.nm_mpx(cof, dW[i][o][p]), None)
            B[i][o] = funcs.mm_sum(B[i][o], funcs.nm_mpx(cof, dB[i][o]), None)
    Em = funcs.mm_sum(Em, funcs.nm_mpx(cof, dEm), None)
    UEm = funcs.mm_sum(UEm, funcs.nm_mpx(cof, dUEm), None)


n_trial = len(Tokens) - ContextSize + 1
step = 500
load('3')

C = 0
for run in range(0, n_trial, step):
    for ReCreatingPars in range(1):
        n: list[list[list[list[float]]]] = [[[[0 for n3 in range(npL[n2])] for n2 in range(len(npL))] for n1 in range(ContextSize)] for n0 in range(L - 1)]
        O: list[float] = []
        EmnA: list[list[list[float]]] = [[[0 for emnA2 in range(EmDim)] for emnA1 in range(ContextSize)] for emnA0 in range(L)]
        EmnM: list[list[list[float]]] = [[[0 for emnM2 in range(EmDim)] for emnM1 in range(ContextSize)] for emnM0 in range(L)]
        EmnO: list[list[float]] = [[0 for emnO1 in range(EmDim)] for emnO0 in range(ContextSize)]
        P: list[list[list[float]]] = [[[0 for p2 in range(ContextSize)] for p1 in range(ContextSize)] for p0 in range(L)]
        DEmn: list[list[list[float]]] = [[[0 for Dems2 in range(EmDim)] for Dems1 in range(ContextSize)] for Dems0 in range(L)]
        Vn: list[list[list[float]]] = [[[0 for vn2 in range(EmDim)] for vn1 in range(ContextSize)] for vn0 in range(L)]
        Kn: list[list[list[float]]] = [[[0 for kn2 in range(EmDim)] for kn1 in range(ContextSize)] for kn0 in range(L)]
        Qn: list[list[list[float]]] = [[[0 for qn2 in range(EmDim)] for qn1 in range(ContextSize)] for qn0 in range(L)]
    fpr(run)
    c = cost(run)
    C += cost(run)
    bpr(run, c)
print(f'-1:\nLoss: {C}, per run: {C / (n_trial // step + 1)}, per tok: {C / (n_trial // step + 1) / vL}\ntop tok: {O.index(max(O))}, wanted tok: {Tokens[run + ContextSize]}, top`: {max(O)}, wanted`: {O[Tokens[run + ContextSize]]}')

for epoch in range(50):
    cng(-1 / 2 ** 14 / (1 + n_trial // step) / (1 - drop_prt))
    for ReCreatingPars in range(1):
        n: list[list[list[list[float]]]] = [[[[0 for n3 in range(npL[n2])] for n2 in range(len(npL))] for n1 in range(ContextSize)] for n0 in range(L - 1)]
        O: list[float] = []
        EmnA: list[list[list[float]]] = [[[0 for emnA2 in range(EmDim)] for emnA1 in range(ContextSize)] for emnA0 in range(L)]
        EmnM: list[list[list[float]]] = [[[0 for emnM2 in range(EmDim)] for emnM1 in range(ContextSize)] for emnM0 in range(L)]
        EmnO: list[list[float]] = [[0 for emnO1 in range(EmDim)] for emnO0 in range(ContextSize)]
        P: list[list[list[float]]] = [[[0 for p2 in range(ContextSize)] for p1 in range(ContextSize)] for p0 in range(L)]
        DEmn: list[list[list[float]]] = [[[0 for Dems2 in range(EmDim)] for Dems1 in range(ContextSize)] for Dems0 in range(L)]
        Vn: list[list[list[float]]] = [[[0 for vn2 in range(EmDim)] for vn1 in range(ContextSize)] for vn0 in range(L)]
        Kn: list[list[list[float]]] = [[[0 for kn2 in range(EmDim)] for kn1 in range(ContextSize)] for kn0 in range(L)]
        Qn: list[list[list[float]]] = [[[0 for qn2 in range(EmDim)] for qn1 in range(ContextSize)] for qn0 in range(L)]

        drops: list[list[list[list[bool]]]] = [[[funcs.dropping(npL[Dn2], drop_prt) for Dn2 in range(len(npL))] for Dn1 in range(ContextSize)] for Dn0 in range(L - 1)]

        dEm: list[list[float]] = [[0 for dem1 in range(EmDim)] for dem0 in range(vL)]
        dUEm: list[list[float]] = [[0 for due1 in range(EmDim)] for due0 in range(vL)]
        dK: list[list[list[list[float]]]] = [[[[0 for dk3 in range(EmDim)] for dk2 in range(KQDim)] for dk1 in range(ContextSize)] for dk0 in range(L)]
        dQ: list[list[list[list[float]]]] = [[[[0 for dq3 in range(EmDim)] for dq2 in range(KQDim)] for dq1 in range(ContextSize)] for dq0 in range(L)]
        dV: list[list[list[list[float]]]] = [[[[0 for dv3 in range(EmDim)] for dv2 in range(KQDim)] for dv1 in range(ContextSize)] for dv0 in range(L)]
        dVo: list[list[list[list[float]]]] = [[[[0 for dvo3 in range(KQDim)] for dvo2 in range(EmDim)] for dvo1 in range(ContextSize)] for dvo0 in range(L)]
        dW: list[list[list[list[list[float]]]]] = [[[[[0 for dw4 in range(npL[dw2])] for dw3 in range(npL[dw2 + 1])] for dw2 in range(len(npL) - 1)] for dw1 in range(ContextSize)] for dw0 in range(L - 1)]
        dB: list[list[list[list[float]]]] = [[[[0 for db3 in range(npL[db2])] for db2 in range(len(npL))] for db1 in range(ContextSize)] for db0 in range(L - 1)]
    C0 = float(C)
    C = 0
    for run in range(0, n_trial, step):
        for ReCreatingPars in range(1):
            n: list[list[list[list[float]]]] = [[[[0 for n3 in range(npL[n2])] for n2 in range(len(npL))] for n1 in range(ContextSize)] for n0 in range(L - 1)]
            O: list[float] = []
            EmnA: list[list[list[float]]] = [[[0 for emnA2 in range(EmDim)] for emnA1 in range(ContextSize)] for emnA0 in range(L)]
            EmnM: list[list[list[float]]] = [[[0 for emnM2 in range(EmDim)] for emnM1 in range(ContextSize)] for emnM0 in range(L)]
            EmnO: list[list[float]] = [[0 for emnO1 in range(EmDim)] for emnO0 in range(ContextSize)]
            P: list[list[list[float]]] = [[[0 for p2 in range(ContextSize)] for p1 in range(ContextSize)] for p0 in range(L)]
            DEmn: list[list[list[float]]] = [[[0 for Dems2 in range(EmDim)] for Dems1 in range(ContextSize)] for Dems0 in range(L)]
            Vn: list[list[list[float]]] = [[[0 for vn2 in range(EmDim)] for vn1 in range(ContextSize)] for vn0 in range(L)]
            Kn: list[list[list[float]]] = [[[0 for kn2 in range(EmDim)] for kn1 in range(ContextSize)] for kn0 in range(L)]
            Qn: list[list[list[float]]] = [[[0 for qn2 in range(EmDim)] for qn1 in range(ContextSize)] for qn0 in range(L)]
        fpr(run)
        c = cost(run)
        C += cost(run)
        bpr(run, c)
    print(f'{epoch}:\nLoss: {C}, per run: {C / (n_trial // step + 1)}, per tok: {C / (n_trial // step + 1) / vL}\ntop tok: {O.index(max(O))}, wanted tok: {Tokens[run + ContextSize]}, top`: {max(O)}, wanted`: {O[Tokens[run + ContextSize]]}, dC: {C - C0}')
    if C - C0 > 0.001:
        break
ans: str = input('Allow uploading (1 or any) ')
if ans == '1':
    upload('3')
    N = int(open(f'{DA}3/tr.txt').read())
    with open(f'{DA}3/tr.txt', 'w') as m:
        m.write(str(N + epoch))
