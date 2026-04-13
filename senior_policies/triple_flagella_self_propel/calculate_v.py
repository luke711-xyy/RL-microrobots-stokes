import math
import os
from pathlib import Path

import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent

# 保持与单机器人/双机器人分支相同的离散化依赖和历史平面映射约定。
Xf_match_q_fila_template = torch.load(BASE_DIR / "Xf_match_q_fila.pt").to(torch.double)
Yf_match_q_fila_template = torch.load(BASE_DIR / "Yf_match_q_fila.pt").to(torch.double)
Zf_match_q_fila_template = torch.load(BASE_DIR / "Zf_match_q_fila.pt").to(torch.double)
Xf_all_fila_template = torch.load(BASE_DIR / "Xf_all_fila.pt").to(torch.double)
Yf_all_fila_template = torch.load(BASE_DIR / "Yf_all_fila.pt").to(torch.double)
Zf_all_fila_template = torch.load(BASE_DIR / "Zf_all_fila.pt").to(torch.double)
Label_Matrix_fila = torch.load(BASE_DIR / "Min_Distance_Label_Fila.pt").to(torch.int64)
Min_Distance_num_fila = torch.load(BASE_DIR / "Min_Distance_num_fila.pt").view(-1).to(torch.int64)
Min_Distance_Label_fila = torch.load(BASE_DIR / "Correponding_label_fila.pt").to(torch.double)


device = torch.device("cpu")

NL = 10
N_dense = int(NL * 8)
N = int(NL * 4)
FORCE_POINT_NUM = int(Xf_all_fila_template.shape[0])
MATCH_POINT_NUM = int(Xf_match_q_fila_template.shape[1])
MU = 1.0

torch.set_num_threads(int(os.environ.get("STOKES_NUM_THREADS", "5")))


def MatrixQ(L, theta, Qu, Q1, Ql, Q2):
    q_up = torch.cat((Qu, -Q2), dim=1)
    q_down = torch.cat((Ql, Q1), dim=1)
    return torch.cat((q_up, q_down), dim=0)


def MatrixQp(L, theta):
    Qu = torch.cat(
        (
            torch.ones((N + 1), dtype=torch.double, device=device).reshape(-1, 1),
            torch.zeros((N + 1), dtype=torch.double, device=device).reshape(-1, 1),
        ),
        dim=1,
    )
    Ql = torch.cat(
        (
            torch.zeros((N + 1), dtype=torch.double, device=device).reshape(-1, 1),
            torch.ones((N + 1), dtype=torch.double, device=device).reshape(-1, 1),
        ),
        dim=1,
    )
    q1 = L * torch.cos(theta[2:])
    q2 = L * torch.sin(theta[2:])

    Q1 = q1.reshape(1, -1).repeat(N + 1, 1)
    Q1 = torch.tril(Q1, -1)

    Q2 = q2.reshape(1, -1).repeat(N + 1, 1)
    Q2 = torch.tril(Q2, -1)

    Q = torch.cat((Qu, Q1, Ql, Q2), dim=1)
    Q = Q.reshape(2 * (N + 1), -1)
    return Q, Qu, Q1, Ql, Q2


def MatrixQp_dense(L, theta):
    Qu = torch.cat(
        (
            torch.ones((N_dense + 1), dtype=torch.double, device=device).reshape(-1, 1),
            torch.zeros((N_dense + 1), dtype=torch.double, device=device).reshape(-1, 1),
        ),
        dim=1,
    )
    Ql = torch.cat(
        (
            torch.zeros((N_dense + 1), dtype=torch.double, device=device).reshape(-1, 1),
            torch.ones((N_dense + 1), dtype=torch.double, device=device).reshape(-1, 1),
        ),
        dim=1,
    )
    q1 = L * torch.cos(theta[2:])
    q2 = L * torch.sin(theta[2:])

    Q1 = q1.reshape(1, -1).repeat(N_dense + 1, 1)
    Q1 = torch.tril(Q1, -1)

    Q2 = q2.reshape(1, -1).repeat(N_dense + 1, 1)
    Q2 = torch.tril(Q2, -1)

    Q = torch.cat((Qu, Q1, Ql, Q2), dim=1)
    Q = Q.reshape(2 * (N_dense + 1), -1)
    return Q, Qu, Q1, Ql, Q2


def MatrixB(L, theta, Y):
    B1 = 0.5 * torch.cat(
        (
            2 * torch.ones((N + 1), dtype=torch.double, device=device).reshape(-1, 1),
            torch.zeros((N + 1), dtype=torch.double, device=device).reshape(-1, 1),
        ),
        dim=1,
    )
    B1[0, 0] = 0.5
    B1[-1, 0] = 0.5
    B1 = B1.reshape(1, -1)

    B2 = 0.5 * torch.cat(
        (
            torch.zeros((N + 1), dtype=torch.double, device=device).reshape(-1, 1),
            2 * torch.ones((N + 1), dtype=torch.double, device=device).reshape(-1, 1),
        ),
        dim=1,
    )
    B2[0, 1] = 0.5
    B2[-1, 1] = 0.5
    B2 = B2.reshape(1, -1)

    B3 = torch.cat((-(Y[:, 1] - Y[0, 1]).view(1, -1), (Y[:, 0] - Y[0, 0]).view(1, -1)), dim=1)
    B3 = B3.reshape(1, -1)

    B = torch.cat((B1, B2), dim=0)

    Bx = B[:, 0::2]
    Bx = torch.cat((Bx, B3[:, : Bx.shape[1]]), dim=0)
    By = B[:, 1::2]
    By = torch.cat((By, B3[:, Bx.shape[1] :]), dim=0)

    Min_Distance_num_fila_copy = Min_Distance_num_fila.clone().reshape(1, -1).repeat(3, 1).to(torch.double)
    Bx = Bx * Min_Distance_num_fila_copy
    By = By * Min_Distance_num_fila_copy
    return torch.cat((Bx, By), dim=1)


def MatrixC(action_absolute):
    C1 = torch.zeros((N + 2, 3), dtype=torch.double, device=device)
    C1[0, 0] = 1
    C1[1, 1] = 1
    C1[2:, 2] = 1

    C2 = torch.zeros((N + 2, 1), dtype=torch.double, device=device)
    C2[3:, :] = action_absolute.view(-1, 1)
    return C1, C2


def MatrixD_sum(beta_ini, absU):
    D1 = torch.zeros((NL * 2 + 2, 3), dtype=torch.double, device=device)
    D2 = torch.zeros((NL * 2 + 2, 1), dtype=torch.double, device=device)
    for i in range(NL + 1):
        if i == 0:
            D1[i * 2, 0] = 1
            D1[i * 2 + 1, 1] = 1
        elif i == 1:
            D1[i * 2, 0] = 1
            D1[i * 2, 2] = D1[(i - 1) * 2, 2] - math.sin(beta_ini[i - 1]) * 1.0 / NL
            D1[i * 2 + 1, 1] = 1
            D1[i * 2 + 1, 2] = D1[(i - 1) * 2 + 1, 2] + math.cos(beta_ini[i - 1]) * 1.0 / NL
            D2[i * 2, :] = D2[(i - 1) * 2, :]
            D2[i * 2 + 1, :] = D2[(i - 1) * 2 + 1, :]
        else:
            D1[i * 2, 0] = 1
            D1[i * 2, 2] = D1[(i - 1) * 2, 2] - math.sin(beta_ini[i - 1]) * 1.0 / NL
            D1[i * 2 + 1, 1] = 1
            D1[i * 2 + 1, 2] = D1[(i - 1) * 2 + 1, 2] + math.cos(beta_ini[i - 1]) * 1.0 / NL
            D2[i * 2, :] = D2[(i - 1) * 2, :] - absU[i - 2] * math.sin(beta_ini[i - 1]) * 1.0 / NL
            D2[i * 2 + 1, :] = D2[(i - 1) * 2 + 1, :] + absU[i - 2] * math.cos(beta_ini[i - 1]) * 1.0 / NL
    return D1, D2


def _block_diag(mats):
    total_rows = sum(mat.shape[0] for mat in mats)
    total_cols = sum(mat.shape[1] for mat in mats)
    output = torch.zeros((total_rows, total_cols), dtype=torch.double, device=device)
    row_offset = 0
    col_offset = 0
    for mat in mats:
        output[row_offset : row_offset + mat.shape[0], col_offset : col_offset + mat.shape[1]] = mat
        row_offset += mat.shape[0]
        col_offset += mat.shape[1]
    return output


def build_triple_Q_total(q_mats):
    point_num = int(q_mats[0].shape[0] // 2)
    body_num = len(q_mats)
    total_rows = point_num * body_num * 2
    total_cols = sum(q_mat.shape[1] for q_mat in q_mats)
    output = torch.zeros((total_rows, total_cols), dtype=torch.double, device=device)

    col_offset = 0
    for body_idx, q_mat in enumerate(q_mats):
        state_dim = int(q_mat.shape[1])
        x_row_start = body_idx * point_num
        z_row_start = body_num * point_num + body_idx * point_num
        output[x_row_start : x_row_start + point_num, col_offset : col_offset + state_dim] = q_mat[0:point_num, :]
        output[z_row_start : z_row_start + point_num, col_offset : col_offset + state_dim] = q_mat[point_num:, :]
        col_offset += state_dim
    return output


def _initial_single(x, w, x_first, n_segments, n_links):
    segment_length = 1.0 / n_segments
    e = segment_length * 0.1

    Xini = float(x_first[0])
    Yini = float(x_first[1])

    beta_ini = torch.tensor(x[2:].copy(), dtype=torch.double, device=device)
    NI = np.zeros(n_links, dtype=np.double)
    for i in range(n_links):
        NI[i] = int(int(n_segments / n_links) * (i + 1) - 1)
        if i > 0:
            beta_ini[i] += beta_ini[i - 1]

    theta = torch.zeros((n_segments + 2), dtype=torch.double, device=device)
    for_qp = torch.ones((n_segments + 2), dtype=torch.double, device=device)
    for_qp[0] = Xini
    for_qp[1] = Yini
    theta[0] = Xini
    theta[1] = Yini

    ratio = int(n_segments / n_links)
    for i in range(n_segments):
        theta[i + 2] = beta_ini[int(i / ratio)]

    if n_segments == N:
        Q, Qu, Q1, Ql, Q2 = MatrixQp(segment_length, theta)
    else:
        Q, Qu, Q1, Ql, Q2 = MatrixQp_dense(segment_length, theta)

    positions = torch.matmul(Q, for_qp).reshape(-1, 2)

    absU = w.copy()
    action = absU.copy()
    for i in range(n_links):
        if 0 < i < n_links - 1:
            absU[i] += absU[i - 1]

    action_absolute = torch.zeros((n_segments - 1), dtype=torch.double, device=device)
    for i in range(n_links - 1):
        a = int(NI[i])
        if i == n_links - 2:
            action_absolute[a:] = absU[i]
        else:
            b = int(NI[i + 1])
            action_absolute[a:b] = absU[i]

    return {
        "L": segment_length,
        "e": e,
        "positions": positions,
        "theta": theta,
        "action_absolute": action_absolute,
        "Qu": Qu,
        "Q1": Q1,
        "Ql": Ql,
        "Q2": Q2,
        "action": action,
        "beta_ini": beta_ini,
        "absU": absU,
    }


def build_match_points(dense_positions):
    x_match = torch.zeros((FORCE_POINT_NUM, MATCH_POINT_NUM), dtype=torch.double, device=device)
    z_match = torch.zeros((FORCE_POINT_NUM, MATCH_POINT_NUM), dtype=torch.double, device=device)
    valid_mask = torch.zeros((FORCE_POINT_NUM, MATCH_POINT_NUM), dtype=torch.double, device=device)

    dense_x = dense_positions[:, 0]
    dense_z = dense_positions[:, 1]

    for sparse_idx in range(FORCE_POINT_NUM):
        selected = Label_Matrix_fila[:, sparse_idx].to(torch.bool)
        count = int(Min_Distance_num_fila[sparse_idx].item())
        if count <= 0:
            continue
        x_match[sparse_idx, :count] = dense_x[selected][:count]
        z_match[sparse_idx, :count] = dense_z[selected][:count]
        valid_mask[sparse_idx, :count] = Min_Distance_Label_fila[sparse_idx, :count]

    return x_match, z_match, valid_mask


def build_joint_stokeslet_matrix(force_points, x_match, z_match, valid_mask, e):
    point_num = force_points.shape[0]

    force_x = force_points[:, 0].view(-1, 1, 1)
    force_z = force_points[:, 1].view(-1, 1, 1)
    target_x = x_match.view(1, point_num, MATCH_POINT_NUM)
    target_z = z_match.view(1, point_num, MATCH_POINT_NUM)
    valid = valid_mask.view(1, point_num, MATCH_POINT_NUM)

    delta_x = force_x - target_x
    delta_z = force_z - target_z
    delta_y = torch.zeros_like(delta_x)

    R = torch.sqrt(delta_x**2 + delta_y**2 + delta_z**2 + e**2)
    RD = 1.0 / R
    RD3 = RD**3
    e2 = e**2

    s00 = (RD + e2 * RD3 + delta_x * delta_x * RD3) * valid
    sxz = (delta_x * delta_z * RD3) * valid
    syy = (RD + e2 * RD3) * valid
    szz = (RD + e2 * RD3 + delta_z * delta_z * RD3) * valid

    s00 = torch.sum(s00, dim=2)
    sxz = torch.sum(sxz, dim=2)
    syy = torch.sum(syy, dim=2)
    szz = torch.sum(szz, dim=2)

    A = torch.zeros((point_num * 3, point_num * 3), dtype=torch.double, device=device)
    A[0:point_num, 0:point_num] = s00
    A[0:point_num, point_num : point_num * 2] = sxz
    A[point_num : point_num * 2, 0:point_num] = sxz
    A[point_num : point_num * 2, point_num : point_num * 2] = szz
    A[point_num * 2 : point_num * 3, point_num * 2 : point_num * 3] = syy
    return A / (8 * math.pi * MU)


def build_triple_B_all(b_mats):
    per_body_points = int(b_mats[0].shape[1] // 2)
    body_num = len(b_mats)
    total_points = per_body_points * body_num
    B_planar = torch.zeros((body_num * 3, total_points * 2), dtype=torch.double, device=device)

    for body_idx, b_mat in enumerate(b_mats):
        rows = slice(body_idx * 3, (body_idx + 1) * 3)
        x_cols = slice(body_idx * per_body_points, (body_idx + 1) * per_body_points)
        z_cols = slice(total_points + body_idx * per_body_points, total_points + (body_idx + 1) * per_body_points)
        B_planar[rows, x_cols] = b_mat[:, :per_body_points]
        B_planar[rows, z_cols] = b_mat[:, per_body_points:]

    B_supply = torch.zeros((body_num * 3, total_points), dtype=torch.double, device=device)
    return torch.cat((B_planar, B_supply), dim=1)


def Calculate_velocity_triple(x1, w1, x_first1, x2, w2, x_first2, x3, w3, x_first3):
    inputs = ((x1, w1, x_first1), (x2, w2, x_first2), (x3, w3, x_first3))
    bodies = [_initial_single(x, w, x_first, N, NL) for x, w, x_first in inputs]
    dense_bodies = [_initial_single(x, w, x_first, N_dense, NL) for x, w, x_first in inputs]

    force_points_all = torch.cat([body["positions"] for body in bodies], dim=0)

    x_match_all = []
    z_match_all = []
    valid_all = []
    for dense_body in dense_bodies:
        x_match, z_match, valid = build_match_points(dense_body["positions"])
        x_match_all.append(x_match)
        z_match_all.append(z_match)
        valid_all.append(valid)

    x_match_all = torch.cat(x_match_all, dim=0)
    z_match_all = torch.cat(z_match_all, dim=0)
    valid_all = torch.cat(valid_all, dim=0)

    A = build_joint_stokeslet_matrix(force_points_all, x_match_all, z_match_all, valid_all, bodies[0]["e"])
    B_all = build_triple_B_all([MatrixB(body["L"], body["theta"], body["positions"]) for body in bodies])
    q_mats = [MatrixQ(body["L"], body["theta"], body["Qu"], body["Q1"], body["Ql"], body["Q2"]) for body in bodies]
    Q_total = build_triple_Q_total(q_mats)

    c_mats = [MatrixC(body["action_absolute"]) for body in bodies]
    C1_total = _block_diag([c_mat[0] for c_mat in c_mats])
    C2_total = torch.cat([c_mat[1] for c_mat in c_mats], dim=0)

    AB = torch.linalg.solve(A.T, B_all.T).T
    AB = AB[:, : Q_total.shape[0]]
    MT = torch.matmul(AB, Q_total)
    M = torch.matmul(MT, C1_total)
    M_rank = int(torch.linalg.matrix_rank(M).item())
    if M_rank < M.shape[0]:
        raise RuntimeError(f"Triple rigid matrix M is rank-deficient: rank={M_rank}, expected={M.shape[0]}")

    R = -torch.matmul(MT, C2_total)
    rigid_vel = torch.linalg.solve(M, R).view(-1)

    outputs = []
    rigid_vel_np = rigid_vel.detach().cpu().numpy()
    states = (x1, x2, x3)
    for body_idx, body in enumerate(bodies):
        rigid_slice = rigid_vel[body_idx * 3 : (body_idx + 1) * 3].view(-1, 1)
        D1, D2 = MatrixD_sum(body["beta_ini"], body["absU"])
        veloall = torch.matmul(D1, rigid_slice) + D2
        veloall_np = np.squeeze(veloall.detach().cpu().numpy())

        velon = np.zeros_like(states[body_idx], dtype=np.float64)
        velon[0] = np.mean(veloall_np[::2]) * 1.1
        velon[1] = np.mean(veloall_np[1::2]) * 1.1
        velon[2] = rigid_vel_np[body_idx * 3 + 2]
        velon[3:] = body["action"]

        Xp = np.squeeze(body["positions"][:, 0].detach().cpu().numpy())
        Yp = np.squeeze(body["positions"][:, 1].detach().cpu().numpy())
        outputs.extend((velon, rigid_vel_np[body_idx * 3 : (body_idx + 1) * 3], Xp, Yp))

    pressure_all = np.zeros((force_points_all.shape[0],), dtype=np.float64)
    outputs.extend((0.0, 0.0, pressure_all))
    return tuple(outputs)


def RK_triple(x1, w1, x_first1, x2, w2, x_first2, x3, w3, x_first3):
    Xn1 = 0.0
    Xn2 = 0.0
    Xn3 = 0.0
    Yn1 = 0.0
    Yn2 = 0.0
    Yn3 = 0.0
    r1 = 0.0
    r2 = 0.0
    r3 = 0.0

    xc1 = x1.copy()
    xc2 = x2.copy()
    xc3 = x3.copy()
    x_first_delta1 = np.zeros((2,), dtype=np.float64)
    x_first_delta2 = np.zeros((2,), dtype=np.float64)
    x_first_delta3 = np.zeros((2,), dtype=np.float64)
    x_fc1 = x_first1.copy()
    x_fc2 = x_first2.copy()
    x_fc3 = x_first3.copy()

    Ntime = 10
    whole_time = 0.2
    part_time = whole_time / Ntime

    Xp1 = np.zeros((NL + 1,), dtype=np.float64)
    Yp1 = np.zeros((NL + 1,), dtype=np.float64)
    Xp2 = np.zeros((NL + 1,), dtype=np.float64)
    Yp2 = np.zeros((NL + 1,), dtype=np.float64)
    Xp3 = np.zeros((NL + 1,), dtype=np.float64)
    Yp3 = np.zeros((NL + 1,), dtype=np.float64)

    for _ in range(Ntime):
        (
            V1,
            Vo1,
            Xp1,
            Yp1,
            V2,
            Vo2,
            Xp2,
            Yp2,
            V3,
            Vo3,
            Xp3,
            Yp3,
            pressure_diff,
            pressure_end,
            pressure_all,
        ) = Calculate_velocity_triple(xc1, w1, x_fc1, xc2, w2, x_fc2, xc3, w3, x_fc3)

        k1_1 = part_time * V1
        k1_2 = part_time * V2
        k1_3 = part_time * V3

        (
            V1,
            Vo1,
            Xp1,
            Yp1,
            V2,
            Vo2,
            Xp2,
            Yp2,
            V3,
            Vo3,
            Xp3,
            Yp3,
            pressure_diff,
            pressure_end,
            pressure_all,
        ) = Calculate_velocity_triple(
            xc1 + 0.5 * k1_1,
            w1,
            x_fc1 + 0.5 * part_time * Vo1[:2],
            xc2 + 0.5 * k1_2,
            w2,
            x_fc2 + 0.5 * part_time * Vo2[:2],
            xc3 + 0.5 * k1_3,
            w3,
            x_fc3 + 0.5 * part_time * Vo3[:2],
        )

        k2_1 = part_time * V1
        k2_2 = part_time * V2
        k2_3 = part_time * V3

        xc1 += k2_1
        xc2 += k2_2
        xc3 += k2_3
        xc1[2] = (xc1[2] + math.pi) % (2 * math.pi) - math.pi
        xc2[2] = (xc2[2] + math.pi) % (2 * math.pi) - math.pi
        xc3[2] = (xc3[2] + math.pi) % (2 * math.pi) - math.pi

        Xn1 += k2_1[0]
        Xn2 += k2_2[0]
        Xn3 += k2_3[0]
        Yn1 += k2_1[1]
        Yn2 += k2_2[1]
        Yn3 += k2_3[1]
        r1 += k2_1[0] / whole_time
        r2 += k2_2[0] / whole_time
        r3 += k2_3[0] / whole_time

        x_first_delta1 += part_time * Vo1[:2]
        x_first_delta2 += part_time * Vo2[:2]
        x_first_delta3 += part_time * Vo3[:2]
        x_fc1 += part_time * Vo1[:2]
        x_fc2 += part_time * Vo2[:2]
        x_fc3 += part_time * Vo3[:2]

    return (
        xc1,
        Xn1,
        Yn1,
        r1,
        x_first_delta1,
        Xp1,
        Yp1,
        xc2,
        Xn2,
        Yn2,
        r2,
        x_first_delta2,
        Xp2,
        Yp2,
        xc3,
        Xn3,
        Yn3,
        r3,
        x_first_delta3,
        Xp3,
        Yp3,
        0.0,
        0.0,
        np.zeros((FORCE_POINT_NUM * 3,), dtype=np.float64),
    )
