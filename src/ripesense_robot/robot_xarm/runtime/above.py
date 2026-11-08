"""Joint angles for a point straight above a taught pose, keeping the taught arm configuration (no wrist flip):
start from the taught joints and adjust only J2, J3, J5 (shoulder, elbow, wrist tilt) by Newton steps on
forward kinematics so that radius and pitch stay the same and z rises by `up` mm."""
import math


def above_joints(arm, j0, up, iters=25):
    fk = lambda q: arm.get_forward_kinematics(list(q), input_is_radian=False, return_is_radian=False)
    c, p0 = fk(j0)
    if c != 0: return None
    tgt = [math.hypot(p0[0], p0[1]), p0[2] + up, p0[4]]
    def f(q):
        c, p = fk(q)
        return None if c != 0 else [math.hypot(p[0], p[1]) - tgt[0], p[2] - tgt[1], (p[4] - tgt[2] + 180) % 360 - 180]
    q = list(j0)[:6]; idx = (1, 2, 4)
    for _ in range(iters):
        e = f(q)
        if e is None: return None
        if abs(e[0]) < 0.5 and abs(e[1]) < 0.5 and abs(e[2]) < 0.2: return q
        J = []
        for i in idx:
            qq = list(q); qq[i] += 0.5; ee = f(qq)
            if ee is None: return None
            J.append([(ee[k] - e[k]) / 0.5 for k in range(3)])
        # solve J^T dq = -e  (J rows are columns of the Jacobian)
        A = [[J[c][r] for c in range(3)] for r in range(3)]; b = [-x for x in e]
        try:
            for col in range(3):                      # Gaussian elimination with partial pivoting
                piv = max(range(col, 3), key=lambda r: abs(A[r][col])); A[col], A[piv] = A[piv], A[col]; b[col], b[piv] = b[piv], b[col]
                for r in range(col + 1, 3):
                    m = A[r][col] / A[col][col]
                    A[r] = [A[r][k] - m * A[col][k] for k in range(3)]; b[r] -= m * b[col]
            dq = [0.0] * 3
            for r in (2, 1, 0):
                dq[r] = (b[r] - sum(A[r][k] * dq[k] for k in range(r + 1, 3))) / A[r][r]
        except ZeroDivisionError:
            return None
        for k, i in enumerate(idx):
            q[i] += max(-8.0, min(8.0, dq[k]))
    return None
