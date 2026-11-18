# -*- coding: utf-8 -*-
"""
simulador_streamlit.py — Centro de Datos Virtual (DeepSeek-V3/R1)
=================================================================
Simulador interactivo del balance térmico concentrado:

    C dT/dt = P_s(t) - h·A·(T - T_a(t)) - Q_ref(t)

Ejecución:
    pip install streamlit plotly numpy
    streamlit run simulador_streamlit.py

Requisitos de la guía cubiertos: parámetros modificables, condiciones
iniciales, gráficas de las variables físicas, evolución temporal,
comparación de escenarios, magnitudes energéticas, verificación de la
conservación de la energía e interpretación automática de resultados.
Los métodos Euler y RK4 están programados a mano en modelo.py.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modelo import Params, simular, perfil_Ta, perfil_carga, trapezoid

# Escenarios del estudio (Tabla 4 del artículo; mismos valores que analisis.py).
# K_p se expresa por nodo: K_p = N·k_p,nodo (con N = 2: 2500 W/K → 1250 W/K por nodo).
ESCENARIOS = {
    "E1 · carga base · templado · aire":
        dict(u=40, Ta=20, A=5, est="aire", Tset=24, kpn=1250),
    "E2 · pico · templado · aire":
        dict(u=85, Ta=20, A=5, est="aire", Tset=24, kpn=1250),
    "E3 · pico · cálido · aire":
        dict(u=85, Ta=32, A=4, est="aire", Tset=24, kpn=1250),
    "E4 · pico · cálido · líquida D2C":
        dict(u=85, Ta=32, A=4, est="liquida", Tset=24, kpn=1250),
    "E5 · calibración óptima (D2C, K_p=6000 W/K, T_set=22 °C)":
        dict(u=85, Ta=32, A=4, est="liquida", Tset=22, kpn=3000),
}
ESTRATEGIAS = ["aire", "liquida", "free"]


@st.cache_data(show_spinner="Simulando…")
def correr(N, C, hA, T_set, Kp, estrategia, u0, rafagas, Ta_m, Ta_A,
           T0, tf, dt, metodo):
    """Simulación con caché: si los parámetros no cambian, no se recalcula."""
    p = Params(N_serv=N, C=C, hA=hA, T_set=T_set, Kp=Kp, estrategia=estrategia)
    return simular(p, perfil_carga(u0, amp_rafagas=rafagas), perfil_Ta(Ta_m, Ta_A),
                   T0=T0, tf=tf, dt=dt, metodo=metodo)

# Paleta validada (dataviz, modo claro)
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRIS, CRIT = "#52514e", "#b3261e"

st.set_page_config(page_title="Centro de Datos Virtual — DeepSeek",
                   layout="wide", page_icon="🌡️")
st.title("🌡️ Centro de Datos Virtual — despliegue mínimo de DeepSeek-V3/R1")
st.caption("Balance térmico concentrado:  C·dT/dt = P_s(t) − h·A·(T − T_a(t)) − Q_ref(t)   ·   "
           "Euler y RK4 programados manualmente")

# ----------------------------- Panel lateral ---------------------------
with st.sidebar:
    st.header("⚙️ Parámetros de entrada")
    esc = st.selectbox("Escenario de partida (Tabla 4 del artículo)",
                       list(ESCENARIOS), index=0,
                       help="Carga los valores del escenario; luego puedes "
                            "modificar cualquier parámetro.")
    E = ESCENARIOS[esc]

    st.subheader("Clúster TI (DeepSeek)")
    N = st.slider("Nodos DGX H100 activos, N_serv", 1, 16, 2,
                  help="2 nodos (16× H100 de 80 GB = 1280 GB) es el mínimo "
                       "para servir DeepSeek-V3 (pesos FP8 ≈ 671 GB) con holgura de KV-cache.")
    u0 = st.slider("Carga media de inferencia u [%]", 5, 100, E["u"]) / 100
    rafagas = st.slider("Amplitud de ráfagas de tokens [%]", 0, 25, 10) / 100

    st.subheader("Ambiente y sala")
    Ta_m = st.slider("Temperatura ambiente media T_a [°C]", 5, 45, E["Ta"])
    Ta_A = st.slider("Amplitud diurna de T_a [°C]", 0, 12, E["A"])
    C_MJ = st.slider("Capacidad térmica C [MJ/K]", 0.5, 10.0, 2.0, 0.1)
    hA = st.slider("Disipación pasiva h·A [W/K]", 50, 1000, 250, 10)

    st.subheader("Refrigeración")
    estrategia = st.radio("Estrategia", ESTRATEGIAS,
                          index=ESTRATEGIAS.index(E["est"]),
                          format_func={"aire": "CRAC por aire",
                                       "liquida": "Líquida Direct-to-Chip",
                                       "free": "Free cooling + respaldo aire"}.get)
    T_set = st.slider("Consigna T_set [°C]", 16, 30, E["Tset"])
    kp_nodo = st.slider("Ganancia del control por nodo k_p [W/K por nodo]",
                        250, 5000, E["kpn"], 50,
                        help="K_p = N·k_p: la refrigeración se dimensiona con el "
                             "clúster, igual que Q_max = 1,2·P_IT,max.")
    Kp = N * kp_nodo
    st.caption(f"K_p total = {N} × {kp_nodo} = **{Kp} W/K**")

    st.subheader("Simulación")
    T0 = st.slider("Temperatura inicial T₀ [°C]", 10, 40, 22)
    horas = st.slider("Horizonte [h]", 1, 72, 24)
    dt = st.select_slider("Paso Δt [s]", [1, 5, 10, 30, 60, 120, 300, 600], 10)
    metodo = st.radio("Método numérico", ["rk4", "euler"],
                      format_func=lambda m: "Runge-Kutta 4" if m == "rk4" else "Euler")
    comparar = st.checkbox("Superponer Euler vs RK4", value=False)

p = Params(N_serv=N, C=C_MJ * 1e6, hA=hA, T_set=T_set, Kp=Kp,
           estrategia=estrategia)
args = (N, C_MJ * 1e6, hA, T_set, Kp, estrategia, u0, rafagas, Ta_m, Ta_A,
        T0, horas * 3600, dt)
s = correr(*args, metodo)
th = s["t"] / 3600
kWh = 1 / 3.6e6

# Estabilidad de Euler: Δt < 2C/(hA+Kp)
dt_crit = 2 * p.C / (p.hA + p.Kp)
if metodo == "euler" and dt > dt_crit:
    st.error(f"⚠️ Euler inestable: Δt = {dt} s > 2C/(hA+Kp) = {dt_crit:.0f} s. "
             "Reduce el paso o usa RK4.")

# ----------------------------- Indicadores -----------------------------
PUE_medio = float(trapezoid(s["P_tot"], s["t"]) / trapezoid(s["Ps"], s["t"]))
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("T máxima", f"{s['T'].max():.1f} °C",
          f"{s['T'].max() - p.T_critica:+.1f} vs 32 °C", delta_color="inverse")
c2.metric("PUE medio", f"{PUE_medio:.3f}")
c3.metric("Energía TI", f"{s['E_IT'][-1]*kWh:.0f} kWh")
c4.metric("Energía refrigeración", f"{s['E_cool'][-1]*kWh:.0f} kWh")
c5.metric("Energía total", f"{s['E_tot'][-1]*kWh:.0f} kWh")

# Alertas de calibración térmica
if s["frac_critica"] > 0:
    st.error(f"🔴 ALERTA CRÍTICA: T > 32 °C (límite admisible ASHRAE A1, riesgo de "
             f"*thermal throttling*) durante el {100*s['frac_critica']:.1f} % del tiempo.")
elif s["frac_alerta"] > 0:
    st.warning(f"🟠 Aviso: T > 27 °C (fuera del rango recomendado ASHRAE) durante "
               f"el {100*s['frac_alerta']:.1f} % del tiempo.")
else:
    st.success("🟢 Calibración correcta: la sala permanece en el rango recomendado "
               "ASHRAE (18–27 °C) todo el horizonte simulado.")

# ----------------------------- Gráficas --------------------------------
fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.11,
                    subplot_titles=("Temperatura de la sala T(t)",
                                    "Potencias [kW]",
                                    "PUE dinámico",
                                    "Energías acumuladas [kWh]"))
fig.add_scatter(x=th, y=s["T"], line=dict(color=C1, width=2.2),
                name="T sala", row=1, col=1)
fig.add_scatter(x=th, y=s["Ta"], line=dict(color=GRIS, width=1.4, dash="dot"),
                name="T ambiente", row=1, col=1)
for yv, col, nom in [(27, GRIS, "27 °C recomendado"), (32, CRIT, "32 °C admisible")]:
    fig.add_hline(y=yv, line=dict(color=col, width=1, dash="dash"), row=1, col=1)
if comparar:
    s2 = correr(*args, "euler" if metodo == "rk4" else "rk4")
    fig.add_scatter(x=s2["t"]/3600, y=s2["T"],
                    line=dict(color=C2, width=1.6, dash="dash"),
                    name=f"T sala ({'Euler' if metodo=='rk4' else 'RK4'})", row=1, col=1)

fig.add_scatter(x=th, y=s["Ps"]/1e3, line=dict(color=C1, width=2), name="P_IT", row=1, col=2)
fig.add_scatter(x=th, y=s["Qr"]/1e3, line=dict(color=C2, width=2), name="Q_ref", row=1, col=2)
fig.add_scatter(x=th, y=s["P_cool"]/1e3, line=dict(color=C4, width=2), name="P_cool", row=1, col=2)

fig.add_scatter(x=th, y=s["PUE"], line=dict(color=C3, width=2.2), name="PUE(t)", row=2, col=1)

fig.add_scatter(x=th, y=s["E_IT"]*kWh, line=dict(color=C1, width=2), name="E_IT", row=2, col=2)
fig.add_scatter(x=th, y=s["E_cool"]*kWh, line=dict(color=C2, width=2), name="E_cool", row=2, col=2)
fig.add_scatter(x=th, y=s["E_tot"]*kWh, line=dict(color=GRIS, width=2), name="E_total", row=2, col=2)

fig.update_layout(height=640, template="plotly_white", hovermode="x unified",
                  legend=dict(orientation="h", y=-0.09),
                  margin=dict(t=60, l=40, r=20, b=40))
fig.update_xaxes(title_text="Tiempo [h]", row=2)
try:                                    # Streamlit reciente
    st.plotly_chart(fig, width="stretch")
except TypeError:                       # versiones anteriores de Streamlit
    st.plotly_chart(fig, use_container_width=True)

# ------------------- Conservación de la energía ------------------------
b = s["balance"]
with st.expander("🔍 Verificación del principio de conservación de la energía"):
    st.latex(r"\int P_s\,dt \;=\; C\,\Delta T \;+\; \int hA\,(T-T_a)\,dt \;+\; \int Q_{ref}\,dt \;+\; \varepsilon")
    st.table({
        "Término": ["Energía térmica generada ∫P_s dt",
                    "Energía almacenada C·ΔT",
                    "Disipación pasiva ∫hA(T−T_a) dt",
                    "Extraída por refrigeración ∫Q_ref dt",
                    "Residuo ε", "Residuo relativo |ε|/E_in"],
        "Valor": [f"{b['E_in']*kWh:.2f} kWh", f"{b['dU']*kWh:.4f} kWh",
                  f"{b['E_env']*kWh:.2f} kWh", f"{b['E_ref']*kWh:.2f} kWh",
                  f"{b['residuo']*kWh:.2e} kWh", f"{b['residuo_rel']:.2e}"],
    })
    st.caption("Un residuo relativo ≪ 1 confirma que el integrador respeta el balance "
               "de energía (la discrepancia proviene del orden del método y de la "
               "cuadratura del trapecio).")

# ------------------- Interpretación automática -------------------------
st.subheader("🧠 Interpretación del resultado")
tau_cl = p.C / (p.hA + p.Kp)
Ps_med = float(np.mean(s["Ps"]))
msg = [
    f"- Constante de tiempo del lazo cerrado **τ = C/(hA+K_p) ≈ {tau_cl/60:.1f} min**: "
    f"tiempo característico con que la sala responde a cambios de carga o de consigna.",
    f"- Potencia TI media **{Ps_med/1e3:.1f} kW** para N={N} nodos "
    f"(pico instalado {p.P_IT_max/1e3:.1f} kW); el control comanda en promedio "
    f"**{np.mean(s['Qr'])/1e3:.1f} kW** de extracción con COP medio **{np.mean(s['COP']):.2f}**.",
]
frac_sat = float(np.mean(s["Qr"] >= p.Q_max * (1 - 1e-9)))
if frac_sat > 0:
    msg.append(
        f"- ⚠️ La refrigeración está **saturada** (Q_ref = Q_max = {p.Q_max/1e3:.1f} kW) "
        f"el {100*frac_sat:.1f} % del tiempo: el equipo ya no puede extraer más calor, "
        "así que subir K_p o bajar T_set no ayuda; hace falta más capacidad, menos carga "
        "o una estrategia más eficiente.")
else:
    msg.append(
        f"- El offset estacionario del control proporcional es ≈ Q_ref/K_p: con K_p={Kp} W/K "
        f"la sala se estabiliza ~{np.mean(s['Qr'])/Kp:.1f} °C por encima de la consigna; "
        "subir K_p o bajar T_set reduce la temperatura a costa de más energía de refrigeración.")
if estrategia == "aire" and Ta_m >= 30:
    msg.append("- Con clima cálido y CRAC por aire el COP cae y el PUE sube; prueba la "
               "estrategia **líquida D2C**: en el estudio redujo ~13 % la energía total diaria.")
st.markdown("\n".join(msg))
st.caption("Proyecto académico — modelo concentrado de un solo nodo térmico; "
           "no sustituye un análisis CFD ni el diseño de detalle.")
