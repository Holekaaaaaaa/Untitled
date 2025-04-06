import funcs

vL: int = 1024
EmDim: int = 16
ContextSize: int = 48
DA: str = 'C:/Users/tatar/TTPuKoJIbI/projU/M/'
m = open('C:/Users/tatar/TTPuKoJIbI/Texts/FullT.txt').read().split(sep='\n')
Tokens: list[int] = [int(i) for i in m[:-1]]
npL: list[int] = [ContextSize, 64, 64, 64, 16, 32, 32, 32, 8, 32, 32, 8, 16, 16, 8, 1]
drop_prt: float = 0.0

for CreatingPars in range(1):
    n: list[list[list[float]]] = [[[0 for n2 in range(npL[n1])] for n1 in range(len(npL))] for n0 in range(EmDim)]
    O: list[float] = []
    o1: list[float] = []
    B: list[list[list[float]]] = [[[0 for b2 in range(npL[b1])] for b1 in range(len(npL))] for b0 in range(EmDim)]
    W: list[list[list[list[float]]]] = [[[[0 for w3 in range(npL[w1])] for w2 in range(npL[w1 + 1])] for w1 in range(len(npL) - 1)] for w0 in range(EmDim)]
    Em: list[list[float]] = [[0 for em1 in range(EmDim)] for em0 in range(vL)]
    UEm: list[list[float]] = [[0 for uem1 in range(EmDim)] for uem0 in range(vL)]

    dB: list[list[list[float]]] = [[[0 for db2 in range(npL[db1])] for db1 in range(len(npL))] for db0 in range(EmDim)]
    dW: list[list[list[list[float]]]] = [[[[0 for dw3 in range(npL[dw1])] for dw2 in range(npL[dw1 + 1])] for dw1 in range(len(npL) - 1)] for dw0 in range(EmDim)]
    dEm: list[list[float]] = [[0 for dem1 in range(EmDim)] for dem0 in range(vL)]
    dUEm: list[list[float]] = [[0 for dum1 in range(EmDim)] for dum0 in range(vL)]

    drops: list[list[list[bool]]] = [[funcs.dropping(npL[d1], drop_prt) for d1 in range(len(npL))] for d0 in range(EmDim)]
    dropsO: list[bool] = funcs.dropping(vL, drop_prt)


def load(ext: str) -> None:
    for i in range(EmDim):
        for o in range(len(npL)):
            m = open(f'{DA}{ext}/B/{i}_{o}.txt').read().split(sep='\n')
            B[i][o] = [float(p) for p in m[:-1]]
    for i in range(EmDim):
        for o in range(len(npL) - 1):
            for p in range(npL[o + 1]):
                m = open(f'{DA}{ext}/W/{i}_{o}_{p}.txt').read().split(sep='\n')
                W[i][o][p] = [float(k) for k in m[:-1]]
    for i in range(vL):
        m = open(f'{DA}{ext}/Em/{i}.txt').read().split(sep='\n')
        Em[i] = [float(o) for o in m[:-1]]
    for i in range(vL):
        m = open(f'{DA}{ext}/UEm/{i}.txt').read().split(sep='\n')
        UEm[i] = [float(o) for o in m[:-1]]
    print('pars loaded')


def upload(ext: str) -> None:
    for i in range(EmDim):
        for o in range(len(npL)):
            open(f'{DA}{ext}/B/{i}_{o}.txt', 'w').write('\n'.join([str(p) for p in B[i][o]]) + '\n')
    for i in range(EmDim):
        for o in range(len(npL) - 1):
            for p in range(npL[o + 1]):
                open(f'{DA}{ext}/W/{i}_{o}_{p}.txt', 'w').write('\n'.join([str(k) for k in W[i][o][p]]) + '\n')
    for i in range(vL):
        open(f'{DA}{ext}/Em/{i}.txt', 'w').write('\n'.join([str(o) for o in Em[i]]) + '\n')
    for i in range(vL):
        open(f'{DA}{ext}/UEm/{i}.txt', 'w').write('\n'.join([str(o) for o in UEm[i]]) + '\n')
    print('pars uploaded')


def fpr(stt: int) -> None:
    global O
    global o1
    for i in range(EmDim):
        for o in range(ContextSize):
            if not drops[i][0][o]:
                n[i][0][o] = funcs.act_n(Em[Tokens[stt + o]][i] + B[i][0][o])

    for i in range(EmDim):
        for o in range(len(npL) - 1):
            n[i][o + 1] = funcs.act_v(funcs.vv_sum(funcs.vm_mpx(n[i][o], W[i][o], drops[i][o + 1]), B[i][o + 1], drops[i][o + 1]))

    o1 = []
    for i in range(EmDim):
        o1 += [n[i][i][-1]]
    O = funcs.vm_mpx(o1, UEm, dropsO)


def cost(stt: int) -> float:
    c: float = 0
    for i in range(vL):
        c += abs(O[i])
    c += abs(1 - O[Tokens[stt + ContextSize]]) - abs(O[Tokens[stt + ContextSize]])
    return c ** 0.5


def bpr(stt: int, c: float) -> None:
    global dUEm
    dn: list[list[list[float]]] = [[[0 for dn2 in range(npL[dn1])] for dn1 in range(len(npL))] for dn0 in range(EmDim)]
    dO: list[float] = [O[i] for i in range(vL)]
    dO[Tokens[stt + ContextSize]] = 1 - O[Tokens[stt + ContextSize]]

    dUEm = funcs.mm_sum(dUEm, funcs.nm_mpx(c, funcs.vv_div(dO, o1)), None)
    do1: list[float] = funcs.vm_div(dO, UEm, o1)
    for i in range(EmDim):
        dn[i][-1][0] = do1[i]
        for o in range(len(npL) - 2, -1, -1):
            dn[i][o] = funcs.vm_div(dn[i][o + 1], W[i][o], n[i][o])
        dB[i] = funcs.mm_sum(dB[i], funcs.nm_mpx(c, dn[i]), None)
        for o in range(len(npL) - 1):
            dW[i][o] = funcs.mm_sum(dW[i][o], funcs.nm_mpx(c, funcs.vv_div(dn[i][o + 1], n[i][o])), None)
    for i in range(ContextSize):
        for o in range(EmDim):
            dEm[Tokens[stt + i]][o] += dn[o][0][i] * c


def cng(cof: float) -> None:
    global Em
    global UEm

    Em = funcs.mm_sum(Em, funcs.nm_mpx(cof, dEm), None)
    UEm = funcs.mm_sum(UEm, funcs.nm_mpx(cof, dUEm), None)

    for i in range(EmDim):
        B[i] = funcs.mm_sum(B[i], funcs.nm_mpx(cof, dB[i]), None)
        for o in range(len(npL) - 1):
            W[i][o] = funcs.mm_sum(W[i][o], funcs.nm_mpx(cof, dW[i][o]), None)


n_trial: int = len(Tokens) - ContextSize + 1
step: int = 400

load('0')
C = 0
for run in range(0, n_trial, step):
    fpr(run)
    c0 = cost(run)
    C += c0
    bpr(run, c0)
print(f'-1:\nLoss: {C}, per run: {C / (n_trial // step + 1)}, per tok: {C / (n_trial // step + 1) / vL}\ntop tok: {O.index(max(O))}, wanted tok: {Tokens[run + ContextSize]}, top`: {max(O)}, wanted`: {O[Tokens[run + ContextSize]]}')

for epoch in range(400):
    cng(-1 / 2 ** 16 / (n_trial // step + 1) / (1 - drop_prt))

    dB: list[list[list[float]]] = [[[0 for db2 in range(npL[db1])] for db1 in range(len(npL))] for db0 in range(EmDim)]
    dW: list[list[list[list[float]]]] = [[[[0 for dw3 in range(npL[dw1])] for dw2 in range(npL[dw1 + 1])] for dw1 in range(len(npL) - 1)] for dw0 in range(EmDim)]
    dEm: list[list[float]] = [[0 for dem1 in range(EmDim)] for dem0 in range(vL)]
    dUEm: list[list[float]] = [[0 for dum1 in range(EmDim)] for dum0 in range(vL)]

    drops: list[list[list[bool]]] = [[funcs.dropping(npL[d1], drop_prt) for d1 in range(len(npL))] for d0 in range(EmDim)]
    dropsO: list[bool] = funcs.dropping(vL, drop_prt)
    C0 = float(C)
    C = 0
    for run in range(0, n_trial, step):
        fpr(run)
        c0 = cost(run)
        C += c0
        bpr(run, c0)
    print(f'{epoch}:\nLoss: {C}, per run: {C / (n_trial // step + 1)}, per tok: {C / (n_trial // step + 1) / vL}\ntop tok: {O.index(max(O))}, wanted tok: {Tokens[run + ContextSize]}, top`: {max(O)}, wanted`: {O[Tokens[run + ContextSize]]}, dC: {(C - C0)}')
    #if C - C0 > 200:
        #break
ans: str = input('Allow uploading (1 or any) ')
if ans == '1':
    out: str = '4ae'
    upload(out)
    N = int(open(f'{DA}{out}/epo.txt').read())
    with open(f'{DA}{out}/epo.txt', 'w') as m:
        m.write(str(N + epoch + 1))
