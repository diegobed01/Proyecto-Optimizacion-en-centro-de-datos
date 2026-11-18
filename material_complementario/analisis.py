# -*- coding: utf-8 -*-
"""
analisis.py — Análisis numérico y escenarios del proyecto.
Genera: figuras/*.png, tablas/*.csv y resultados.json con TODOS los
números que se citan en el documento (ningún valor del doc es inventado).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from modelo import (Params, P_IT, COP, dTdt, analitica_lazo_abierto,
                    analitica_lazo_cerrado, euler, rk4, simular,
                    perfil_Ta, perfil_carga, trapezoid)

os.makedirs("figuras", exist_ok=True)
os.makedirs("tablas", exist_ok=True)

# Paleta validada (dataviz, modo claro) --------------------------------
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRIS = "#52514e"; CRIT = "#b3261e"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
    "axes.grid": True, "grid.color": "#e5e4e0", "grid.linewidth": 0.7,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#0b0b0b",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})
R = {}   # resultados.json

p0 = Params()
R["parametros"] = {k: getattr(p0, k) for k in
                   ["N_serv", "P_nodo_max", "k_idle", "f_aux", "C", "hA",
                    "T_set", "Kp", "f_dim", "T_alerta", "T_critica"]}
R["parametros"]["P_IT_max"] = p0.P_IT_max
R["parametros"]["Q_max"] = p0.Q_max

# ======================================================================
# A. SOLUCIÓN ANALÍTICA — constantes de tiempo y estados estacionarios
# ======================================================================
u_pico = 1.0
Ps_pico = P_IT(u_pico, p0)                      # 20.4 kW
Ta_cte = 20.0
_, tau_oa, Tss_sin_ref = analitica_lazo_abierto(0, 22, Ps_pico, Ta_cte, 0.0, p0)
_, tau_cl, Tss_cl = analitica_lazo_cerrado(0, 22, Ps_pico, Ta_cte, p0)
# Q_ref estacionario del lazo cerrado y verificación de no saturación
Qr_ss = p0.Kp * (Tss_cl - p0.T_set)
R["analitico"] = dict(
    Ps_pico=Ps_pico, tau_lazo_abierto_s=tau_oa, tau_lazo_abierto_h=tau_oa/3600,
    Tss_sin_refrigeracion=Tss_sin_ref, dT_sin_ref=Tss_sin_ref - Ta_cte,
    tau_lazo_cerrado_s=tau_cl, Tss_lazo_cerrado=Tss_cl, Qr_ss_W=Qr_ss,
    Qmax_W=p0.Q_max, saturado=bool(Qr_ss > p0.Q_max))
print("A) τ_oa=%.0f s (%.2f h)  T_ss sin ref=%.1f °C | τ_cl=%.0f s  T_ss=%.2f °C  Qr_ss=%.0f W (Qmax=%.0f)"
      % (tau_oa, tau_oa/3600, Tss_sin_ref, tau_cl, Tss_cl, Qr_ss, p0.Q_max))

# ======================================================================
# B. VALIDACIÓN EULER/RK4 CONTRA LA SOLUCIÓN ANALÍTICA (lazo cerrado)
#    Caso lineal: u=1, T_a=20 °C ctes, control no saturado.
# ======================================================================
u_fun_c = lambda t: 1.0
Ta_fun_c = lambda t: Ta_cte
# T0 > T_set para que 0 < Kp(T-T_set) < Q_max en toda la trayectoria
# (control no saturado => la EDO es exactamente la lineal de la analítica)
T0, tf_val = 25.0, 2.0 * 3600.0          # 2 h ≈ 10·τ_cl

T_exacta = lambda t: analitica_lazo_cerrado(t, T0, Ps_pico, Ta_cte, p0)[0]

pasos = [600.0, 300.0, 120.0, 60.0, 30.0, 10.0, 5.0]
filas = []
for dt in pasos:
    te, Te = euler(dTdt, 0, tf_val, T0, dt, p0, u_fun_c, Ta_fun_c)
    tr, Tr = rk4 (dTdt, 0, tf_val, T0, dt, p0, u_fun_c, Ta_fun_c)
    ref_e, ref_r = T_exacta(te), T_exacta(tr)
    eaE = np.max(np.abs(Te - ref_e)); erE = np.max(np.abs(Te - ref_e) / np.abs(ref_e))
    eaR = np.max(np.abs(Tr - ref_r)); erR = np.max(np.abs(Tr - ref_r) / np.abs(ref_r))
    filas.append([dt, eaE, erE, eaR, erR])
    print(f"B) dt={dt:6.0f} s | Euler: ea={eaE:.3e} °C er={erE:.3e} | RK4: ea={eaR:.3e} °C er={erR:.3e}")

filas = np.array(filas)
# Orden de convergencia observado (pendiente log-log entre pasos sucesivos)
ordE = np.polyfit(np.log(filas[:, 0]), np.log(filas[:, 1]), 1)[0]
ordR = np.polyfit(np.log(filas[:, 0]), np.log(filas[:, 3]), 1)[0]
R["validacion"] = dict(pasos_s=filas[:, 0].tolist(),
                       err_abs_euler=filas[:, 1].tolist(),
                       err_rel_euler=filas[:, 2].tolist(),
                       err_abs_rk4=filas[:, 3].tolist(),
                       err_rel_rk4=filas[:, 4].tolist(),
                       orden_euler=ordE, orden_rk4=ordR,
                       dt_estabilidad_euler_s=2 * tau_cl)
print(f"B) orden observado: Euler={ordE:.2f}  RK4={ordR:.2f}  (límite estab. Euler ≈ {2*tau_cl:.0f} s)")
np.savetxt("tablas/tabla_errores.csv", filas, delimiter=",",
           header="dt_s,err_abs_euler_C,err_rel_euler,err_abs_rk4_C,err_rel_rk4",
           comments="", fmt="%.6e")

# --- Figura 2: T(t) analítica vs Euler vs RK4 con paso grueso ---------
dt_demo = 300.0
te, Te = euler(dTdt, 0, tf_val, T0, dt_demo, p0, u_fun_c, Ta_fun_c)
tr, Tr = rk4 (dTdt, 0, tf_val, T0, dt_demo, p0, u_fun_c, Ta_fun_c)
tfino = np.linspace(0, tf_val, 600)
fig, ax = plt.subplots(figsize=(7.0, 3.8))
ax.plot(tfino/60, T_exacta(tfino), color=GRIS, lw=2.4, label="Analítica")
ax.plot(te/60, Te, "o--", color=C2, lw=1.6, ms=5, mfc="white", label=f"Euler (Δt = {dt_demo:.0f} s)")
ax.plot(tr/60, Tr, "s-",  color=C1, lw=1.6, ms=4.5, label=f"RK4 (Δt = {dt_demo:.0f} s)")
ax.axhline(Tss_cl, color=C3, lw=1.1, ls=":",)
ax.annotate(f"T_ss = {Tss_cl:.2f} °C", xy=(0.99, Tss_cl), xycoords=("axes fraction", "data"),
            ha="right", va="bottom", fontsize=9, color=GRIS)
ax.set_xlabel("Tiempo [min]"); ax.set_ylabel("Temperatura de la sala T [°C]")
ax.set_title("Validación: solución analítica vs. Euler y RK4 (lazo cerrado lineal)", fontsize=10.5)
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figuras/fig2_validacion.png"); plt.close(fig)

# --- Figura 3: convergencia log-log -----------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.loglog(filas[:, 0], filas[:, 1], "o-", color=C2, lw=1.8, ms=5.5, label=f"Euler (orden obs. {ordE:.2f})")
ax.loglog(filas[:, 0], filas[:, 3], "s-", color=C1, lw=1.8, ms=5, label=f"RK4 (orden obs. {ordR:.2f})")
dts = np.array([pasos[-1], pasos[0]])
ax.loglog(dts, filas[-1, 1] * (dts/pasos[-1])**1, ":", color=C2, lw=1, alpha=0.7)
ax.loglog(dts, filas[-1, 3] * (dts/pasos[-1])**4, ":", color=C1, lw=1, alpha=0.7)
ax.set_xlabel("Tamaño de paso Δt [s]"); ax.set_ylabel("Error absoluto máximo [°C]")
ax.set_title("Convergencia frente a la solución analítica (guías: pendientes 1 y 4)", fontsize=10.5)
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figuras/fig3_convergencia.png"); plt.close(fig)

# ======================================================================
# C. ESCENARIOS DE SIMULACIÓN (24 h, RK4, Δt = 10 s)
# ======================================================================
escenarios = {
    "E1": dict(nombre="Carga base · clima templado · aire",
               u=0.40, Ta=20.0, A=5.0, estrategia="aire"),
    "E2": dict(nombre="Pico de tokens · clima templado · aire",
               u=0.85, Ta=20.0, A=5.0, estrategia="aire"),
    "E3": dict(nombre="Pico de tokens · clima cálido · aire",
               u=0.85, Ta=32.0, A=4.0, estrategia="aire"),
    "E4": dict(nombre="Pico de tokens · clima cálido · líquida D2C",
               u=0.85, Ta=32.0, A=4.0, estrategia="liquida"),
    "E5": dict(nombre="Calibración óptima: D2C + Kp=6000 W/K, T_set=22 °C",
               u=0.85, Ta=32.0, A=4.0, estrategia="liquida",
               Kp=6000.0, T_set=22.0),
}
res = {}
for k, e in escenarios.items():
    p = Params(estrategia=e["estrategia"],
               Kp=e.get("Kp", 2500.0), T_set=e.get("T_set", 24.0))
    s = simular(p, perfil_carga(e["u"]), perfil_Ta(e["Ta"], e["A"]),
                T0=22.0, tf=24*3600, dt=10.0, metodo="rk4")
    res[k] = s
    kWh = 1/3.6e6
    R.setdefault("escenarios", {})[k] = dict(
        nombre=e["nombre"], u_base=e["u"], Ta_media=e["Ta"],
        estrategia=e["estrategia"],
        T_max=float(s["T"].max()), T_media=float(s["T"].mean()),
        PUE_medio=float(trapezoid(s["P_tot"], s["t"]) / trapezoid(s["Ps"], s["t"])),
        PUE_max=float(s["PUE"].max()),
        E_IT_kWh=float(s["E_IT"][-1]*kWh), E_cool_kWh=float(s["E_cool"][-1]*kWh),
        E_tot_kWh=float(s["E_tot"][-1]*kWh),
        frac_T_sobre_27=s["frac_alerta"], frac_T_sobre_32=s["frac_critica"],
        balance_residuo_rel=s["balance"]["residuo_rel"],
        COP_medio=float(s["COP"].mean()))
    r = R["escenarios"][k]
    print(f"C) {k}: Tmax={r['T_max']:.2f} °C  PUE={r['PUE_medio']:.3f}  "
          f"E_tot={r['E_tot_kWh']:.1f} kWh  resid={r['balance_residuo_rel']:.2e}")

# Ahorro E4 vs E3
R["ahorro_E4_vs_E3"] = dict(
    kWh=R["escenarios"]["E3"]["E_tot_kWh"] - R["escenarios"]["E4"]["E_tot_kWh"],
    pct=100*(1 - R["escenarios"]["E4"]["E_tot_kWh"]/R["escenarios"]["E3"]["E_tot_kWh"]),
    pct_cool=100*(1 - R["escenarios"]["E4"]["E_cool_kWh"]/R["escenarios"]["E3"]["E_cool_kWh"]))

# Verificación adicional: RK4 Δt=10 s vs referencia Δt=0.5 s (caso no lineal E3)
p3 = Params(estrategia="aire")
sref = simular(p3, perfil_carga(0.85), perfil_Ta(32.0, 4.0), 22.0, 24*3600, 0.5, "rk4")
idx = np.searchsorted(sref["t"], res["E3"]["t"])
err_ref = float(np.max(np.abs(res["E3"]["T"] - sref["T"][idx])))
R["error_vs_referencia_E3_C"] = err_ref
print(f"C) error RK4(Δt=10 s) vs referencia(Δt=0.5 s) en E3: {err_ref:.2e} °C")

# Tabla CSV de escenarios
with open("tablas/tabla_escenarios.csv", "w", encoding="utf-8") as f:
    f.write("escenario,nombre,T_max_C,T_media_C,PUE_medio,PUE_max,E_IT_kWh,E_cool_kWh,E_tot_kWh,pct_t_sobre_27C,pct_t_sobre_32C,COP_medio,residuo_rel\n")
    for k, r in R["escenarios"].items():
        f.write(f"{k},{r['nombre']},{r['T_max']:.2f},{r['T_media']:.2f},{r['PUE_medio']:.3f},"
                f"{r['PUE_max']:.3f},{r['E_IT_kWh']:.1f},{r['E_cool_kWh']:.1f},{r['E_tot_kWh']:.1f},"
                f"{100*r['frac_T_sobre_27']:.1f},{100*r['frac_T_sobre_32']:.1f},"
                f"{r['COP_medio']:.2f},{r['balance_residuo_rel']:.2e}\n")

# --- Figura 4: evolución térmica de los 4 escenarios -------------------
cols = {"E1": C3, "E2": C1, "E3": C2, "E4": C4, "E5": C5}
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for k in escenarios:
    s = res[k]
    ls = "--" if k == "E4" else "-"   # E4 comparte trayectoria térmica con E3
    ax.plot(s["t"]/3600, s["T"], color=cols[k], lw=1.8, ls=ls,
            label=f"{k}: {escenarios[k]['nombre']}")
ax.axhline(27, color=GRIS, ls="--", lw=1)
ax.axhline(32, color=CRIT, ls="--", lw=1)
ax.text(0.3, 27.12, "límite recomendado ASHRAE (27 °C)", ha="left", fontsize=8, color=GRIS)
ax.text(0.3, 32.12, "límite admisible / riesgo de throttling (32 °C)", ha="left", fontsize=8, color=CRIT)
ax.set_xlabel("Tiempo [h]"); ax.set_ylabel("Temperatura de la sala T [°C]")
ax.set_xlim(0, 24); ax.set_ylim(21.5, 34.8)
ax.set_title("Evolución térmica T(t) en los cinco escenarios (RK4, Δt = 10 s)", fontsize=10.5)
ax.legend(frameon=False, fontsize=8.0, loc="lower center",
          bbox_to_anchor=(0.5, -0.46), ncol=2)
fig.tight_layout(); fig.savefig("figuras/fig4_escenarios_T.png",
                                bbox_inches="tight"); plt.close(fig)

# --- Figura 5: PUE dinámico -------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for k in escenarios:
    s = res[k]
    ax.plot(s["t"]/3600, s["PUE"], color=cols[k], lw=1.6, label=k)
ax.set_xlabel("Tiempo [h]"); ax.set_ylabel("PUE(t) [-]")
ax.set_xlim(0, 24)
ax.set_title("PUE dinámico: PUE(t) = (P_IT + P_aux + P_cool)/P_IT", fontsize=10.5)
ax.legend(frameon=False, ncol=4, fontsize=9)
fig.tight_layout(); fig.savefig("figuras/fig5_escenarios_PUE.png"); plt.close(fig)

# --- Figura 6: energías de 24 h por escenario (barras apiladas) --------
fig, ax = plt.subplots(figsize=(6.8, 3.8))
ks = list(escenarios)
eIT  = [R["escenarios"][k]["E_IT_kWh"] for k in ks]
eCo  = [R["escenarios"][k]["E_cool_kWh"] for k in ks]
eAux = [R["escenarios"][k]["E_tot_kWh"] - R["escenarios"][k]["E_IT_kWh"]
        - R["escenarios"][k]["E_cool_kWh"] for k in ks]
x = np.arange(len(ks)); w = 0.58
ax.set_axisbelow(True)
ax.bar(x, eIT, w, color=C1, label="E. equipos TI", edgecolor="white", linewidth=2)
ax.bar(x, eCo, w, bottom=eIT, color=C2, label="E. refrigeración", edgecolor="white", linewidth=2)
ax.bar(x, eAux, w, bottom=np.array(eIT)+np.array(eCo), color=C4,
       label="E. auxiliares", edgecolor="white", linewidth=2)
for i, k in enumerate(ks):
    tot = R["escenarios"][k]["E_tot_kWh"]
    ax.text(i, tot + 4, f"{tot:.0f} kWh\nPUE {R['escenarios'][k]['PUE_medio']:.2f}",
            ha="center", fontsize=8.5, color="#0b0b0b")
ax.set_xticks(x); ax.set_xticklabels(ks)
ax.set_ylabel("Energía en 24 h [kWh]"); ax.set_ylim(0, max(np.array(eIT)+np.array(eCo)+np.array(eAux))*1.22)
ax.set_title("Desglose energético diario y PUE medio por escenario", fontsize=10.5)
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig("figuras/fig6_energias.png"); plt.close(fig)

# --- Figura 1: diagrama del sistema físico y sus fronteras -------------
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
fig, ax = plt.subplots(figsize=(7.4, 4.1)); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.16, 0.14), 0.56, 0.68, boxstyle="round,pad=0.012",
             lw=2, edgecolor=GRIS, facecolor="#f4f3ef", ls="--"))
ax.text(0.44, 0.855, "Frontera del sistema: sala TI  (nodo térmico concentrado, T(t), C)",
        ha="center", fontsize=9.5, color=GRIS)
for i in range(2):
    ax.add_patch(Rectangle((0.235 + 0.17*i, 0.30), 0.115, 0.34, facecolor=C1,
                 edgecolor="white", lw=2))
    ax.text(0.2925 + 0.17*i, 0.47, f"DGX\nH100\n#{i+1}", ha="center", va="center",
            color="white", fontsize=8.6, fontweight="bold")
ax.text(0.375, 0.225, "Clúster mínimo DeepSeek-V3/R1 (inferencia)\n"
        "P_s = P_IT(u) = N·P_max·[k_idle + (1−k_idle)·u]",
        ha="center", fontsize = 8.3, color="#0b0b0b")
# Flechas de energía
ax.add_patch(FancyArrowPatch((0.555, 0.47), (0.69, 0.47), arrowstyle="-|>",
             mutation_scale=22, color=C2, lw=2.6))
ax.text(0.625, 0.515, "Q_ref(t)", ha="center", fontsize=9.5, color=C2, fontweight="bold")
ax.add_patch(FancyArrowPatch((0.72, 0.68), (0.85, 0.68), arrowstyle="-|>",
             mutation_scale=18, color=C3, lw=2.2))
ax.text(0.785, 0.725, "h·A·(T − T_a)", ha="center", fontsize=9, color=C3)
ax.add_patch(FancyBboxPatch((0.70, 0.33), 0.135, 0.26, boxstyle="round,pad=0.012",
             lw=1.6, edgecolor=C2, facecolor="white"))
ax.text(0.7675, 0.46, "Refrigeración\nCRAC / D2C\nP_cool = Q_ref/COP", ha="center",
        va="center", fontsize=8.2, color=C2)
ax.text(0.925, 0.47, "Ambiente\nT_a(t)", ha="center", fontsize=9.5, color=GRIS)
ax.add_patch(FancyArrowPatch((0.05, 0.47), (0.165, 0.47), arrowstyle="-|>",
             mutation_scale=22, color=C4, lw=2.6))
ax.text(0.095, 0.53, "P_IT + P_aux\n(red eléctrica)", ha="center", fontsize=8.6, color="#8a6d00")
ax.text(0.44, 0.045, r"Balance:  C·dT/dt = P_s(t) − h·A·(T − T_a(t)) − Q_ref(t)",
        ha="center", fontsize=10.5, color="#0b0b0b",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fcfcfb", ec="#c3c2b7"))
ax.set_xlim(0, 1); ax.set_ylim(0, 0.92)
fig.tight_layout(); fig.savefig("figuras/fig1_sistema.png"); plt.close(fig)

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(R, f, indent=2, ensure_ascii=False)
print("\nListo: figuras/, tablas/, resultados.json")
