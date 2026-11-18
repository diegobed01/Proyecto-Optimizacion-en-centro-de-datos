# -*- coding: utf-8 -*-
"""
modelo.py — Núcleo físico-matemático del proyecto
"Optimización Energética y Modelado Térmico de Centros de Datos
 para el Despliegue Mínimo Óptimo de DeepSeek"

Modelo de balance térmico concentrado (lumped) de la sala TI:

    C dT/dt = P_s(t) - h·A·(T - T_a(t)) - Q_ref(t)        [W]

donde:
    C      : capacidad térmica equivalente de la sala        [J/K]
    T      : temperatura media del aire de la sala (≈ T_in)  [°C]
    P_s    : potencia térmica disipada por los servidores    [W]
    h·A    : coeficiente global de disipación pasiva         [W/K]
    T_a    : temperatura ambiente exterior                   [°C]
    Q_ref  : calor extraído por la refrigeración activa      [W]

Los métodos de Euler y Runge-Kutta 4 están programados manualmente
(requisito de la guía del proyecto); SciPy NO se usa para integrar.
"""

import numpy as np
from dataclasses import dataclass, field

# Compatibilidad NumPy 1.x / 2.x: np.trapezoid existe desde NumPy 2.0 y
# np.trapz fue retirado en versiones recientes. Se usa el que exista.
trapezoid = getattr(np, "trapezoid", None) or np.trapz

# ----------------------------------------------------------------------
# 1. PARÁMETROS DEL CASO DE ESTUDIO (ver sección 2.4 del documento)
# ----------------------------------------------------------------------

@dataclass
class Params:
    # --- Carga TI: clúster mínimo DeepSeek-V3/R1 en inferencia ---------
    # DeepSeek-V3: MoE de 671B parámetros (37B activos/token), pesos FP8
    # ≈ 671 GB -> 16 GPU H100/H800 de 80 GB (1280 GB) = 2 nodos DGX H100.
    N_serv: int = 2              # nodos DGX H100 activos [-]
    P_nodo_max: float = 10_200.0 # consumo máximo por nodo DGX H100 [W] (NVIDIA)
    k_idle: float = 0.30         # fracción de potencia en reposo [-] (calibrado)
    f_aux: float = 0.05          # auxiliares eléctricos (distribución, luces) [-]

    # --- Sala / envolvente (parámetros de calibración declarados) ------
    C: float = 2.0e6             # capacidad térmica equivalente [J/K]
    hA: float = 250.0            # disipación pasiva global [W/K]

    # --- Control de refrigeración --------------------------------------
    T_set: float = 24.0          # consigna [°C] (ASHRAE recomendado 18-27 °C)
    Kp: float = 2_500.0          # ganancia proporcional [W/K]
    f_dim: float = 1.2           # sobredimensionamiento de Q_max vs P_IT,max [-]

    # --- Estrategia de refrigeración: 'aire' (CRAC), 'liquida' (D2C),
    #     'free' (free cooling con respaldo de aire) --------------------
    estrategia: str = "aire"

    # --- Umbrales de alerta térmica ------------------------------------
    T_alerta: float = 27.0       # sale del rango recomendado ASHRAE [°C]
    T_critica: float = 32.0      # límite admisible A1 / riesgo throttling [°C]

    @property
    def P_IT_max(self) -> float:
        return self.N_serv * self.P_nodo_max

    @property
    def Q_max(self) -> float:
        return self.f_dim * self.P_IT_max


# ----------------------------------------------------------------------
# 2. SUBMODELOS FÍSICOS
# ----------------------------------------------------------------------

def P_IT(u, p: Params):
    """Potencia eléctrica TI [W] con modelo lineal idle→pico
    (forma propuesta por Fan, Weber y Barroso, ISCA'07):
        P_IT(u) = N·P_max·(k_idle + (1-k_idle)·u),  u∈[0,1]."""
    u = np.clip(u, 0.0, 1.0)
    return p.N_serv * p.P_nodo_max * (p.k_idle + (1.0 - p.k_idle) * u)


def COP(T_a, p: Params):
    """Coeficiente de desempeño de la refrigeración, decreciente con la
    temperatura ambiente (parametrización calibrada, declarada en el doc)."""
    T_a = np.asarray(T_a, dtype=float)
    if p.estrategia == "aire":        # CRAC por aire
        return np.maximum(1.8, 4.0 - 0.08 * (T_a - 20.0))
    elif p.estrategia == "liquida":   # Direct-to-Chip
        return np.maximum(3.0, 7.0 - 0.05 * (T_a - 20.0))
    elif p.estrategia == "free":      # free cooling si T_a < T_set-4, si no aire
        cop_aire = np.maximum(1.8, 4.0 - 0.08 * (T_a - 20.0))
        return np.where(T_a < p.T_set - 4.0, 15.0, cop_aire)
    raise ValueError(f"estrategia desconocida: {p.estrategia}")


def Q_ref_control(T, p: Params):
    """Ley de control proporcional saturada:
        Q_ref = clip( Kp·(T - T_set), 0, Q_max )   [W]."""
    return np.clip(p.Kp * (np.asarray(T) - p.T_set), 0.0, p.Q_max)


def dTdt(t, T, p: Params, u_fun, Ta_fun):
    """Lado derecho de la EDO del balance térmico concentrado."""
    Ps = P_IT(u_fun(t), p)
    Qr = Q_ref_control(T, p)
    return (Ps - p.hA * (T - Ta_fun(t)) - Qr) / p.C


# ----------------------------------------------------------------------
# 3. SOLUCIÓN ANALÍTICA (regímenes lineales)
# ----------------------------------------------------------------------

def analitica_lazo_abierto(t, T0, Ps, Ta, Qr, p: Params):
    """P_s, T_a, Q_ref constantes (lazo abierto):
        τ = C/hA ;  T_ss = T_a + (P_s - Q_ref)/hA
        T(t) = T_ss + (T0 - T_ss)·exp(-t/τ)."""
    tau = p.C / p.hA
    T_ss = Ta + (Ps - Qr) / p.hA
    return T_ss + (T0 - T_ss) * np.exp(-np.asarray(t) / tau), tau, T_ss


def analitica_lazo_cerrado(t, T0, Ps, Ta, p: Params):
    """Control proporcional NO saturado (0 < Kp(T-T_set) < Q_max):
        C dT/dt = P_s - hA(T-T_a) - Kp(T-T_set)   (sigue siendo lineal)
        τ_cl = C/(hA+Kp)
        T_ss = (P_s + hA·T_a + Kp·T_set)/(hA+Kp)."""
    tau = p.C / (p.hA + p.Kp)
    T_ss = (Ps + p.hA * Ta + p.Kp * p.T_set) / (p.hA + p.Kp)
    return T_ss + (T0 - T_ss) * np.exp(-np.asarray(t) / tau), tau, T_ss


# ----------------------------------------------------------------------
# 4. MÉTODOS NUMÉRICOS PROGRAMADOS A MANO (requisito de la guía)
# ----------------------------------------------------------------------

def euler(f, t0, tf, T0, dt, *args):
    """Método de Euler explícito:  T_{n+1} = T_n + dt·f(t_n, T_n)."""
    n = int(round((tf - t0) / dt))
    t = t0 + dt * np.arange(n + 1)
    T = np.empty(n + 1); T[0] = T0
    for k in range(n):
        T[k + 1] = T[k] + dt * f(t[k], T[k], *args)
    return t, T


def rk4(f, t0, tf, T0, dt, *args):
    """Runge-Kutta clásico de 4.º orden:
        k1=f(t,T); k2=f(t+dt/2,T+dt·k1/2); k3=f(t+dt/2,T+dt·k2/2);
        k4=f(t+dt,T+dt·k3);  T_{n+1}=T_n+dt(k1+2k2+2k3+k4)/6."""
    n = int(round((tf - t0) / dt))
    t = t0 + dt * np.arange(n + 1)
    T = np.empty(n + 1); T[0] = T0
    for k in range(n):
        k1 = f(t[k], T[k], *args)
        k2 = f(t[k] + dt / 2, T[k] + dt * k1 / 2, *args)
        k3 = f(t[k] + dt / 2, T[k] + dt * k2 / 2, *args)
        k4 = f(t[k] + dt,     T[k] + dt * k3,     *args)
        T[k + 1] = T[k] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return t, T


# ----------------------------------------------------------------------
# 5. SIMULACIÓN COMPLETA + MAGNITUDES ENERGÉTICAS Y PUE
# ----------------------------------------------------------------------

def simular(p: Params, u_fun, Ta_fun, T0=22.0, tf=24 * 3600.0, dt=10.0,
            metodo="rk4"):
    """Integra la EDO y calcula en postproceso P_IT, Q_ref, COP, P_cool,
    P_total, PUE dinámico, energías acumuladas y el balance de energía."""
    integrador = rk4 if metodo == "rk4" else euler
    t, T = integrador(dTdt, 0.0, tf, T0, dt, p, u_fun, Ta_fun)

    u = np.clip(np.vectorize(u_fun)(t), 0, 1)
    Ta = np.vectorize(Ta_fun)(t)
    Ps = P_IT(u, p)
    Qr = Q_ref_control(T, p)
    cop = COP(Ta, p)
    P_cool = Qr / cop
    P_aux = p.f_aux * p.P_IT_max * np.ones_like(t)
    P_tot = Ps + P_cool + P_aux
    pue = P_tot / Ps

    # Energías acumuladas [J] por regla del trapecio
    E = lambda y: np.concatenate(([0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))))
    E_IT, E_cool, E_tot = E(Ps), E(P_cool), E(P_tot)

    # Verificación del principio de conservación de la energía (térmica):
    #   ∫P_s dt  =  C·ΔT  +  ∫hA(T-T_a) dt  +  ∫Q_ref dt  + residuo
    E_in = E(Ps)[-1]
    E_env = E(p.hA * (T - Ta))[-1]
    E_ref = E(Qr)[-1]
    dU = p.C * (T[-1] - T[0])
    residuo = E_in - (dU + E_env + E_ref)
    balance = dict(E_in=E_in, dU=dU, E_env=E_env, E_ref=E_ref,
                   residuo=residuo, residuo_rel=abs(residuo) / max(E_in, 1e-12))

    return dict(t=t, T=T, u=u, Ta=Ta, Ps=Ps, Qr=Qr, COP=cop, P_cool=P_cool,
                P_aux=P_aux, P_tot=P_tot, PUE=pue,
                E_IT=E_IT, E_cool=E_cool, E_tot=E_tot, balance=balance,
                frac_alerta=float(np.mean(T > p.T_alerta)),
                frac_critica=float(np.mean(T > p.T_critica)))


# ----------------------------------------------------------------------
# 6. PERFILES TEMPORALES (carga de inferencia y ciclo diurno)
# ----------------------------------------------------------------------

def perfil_Ta(T_media, amplitud):
    """Ciclo diurno sinusoidal con máximo a las 15 h."""
    return lambda t: T_media + amplitud * np.sin(2 * np.pi * (t / 86400.0) - 2 * np.pi * 9 / 24)


def perfil_carga(u_base, amp_diurna=0.20, amp_rafagas=0.10):
    """Carga de inferencia: valle nocturno + ciclo laboral + ráfagas de
    tokens (suma determinista de senos, reproducible)."""
    def u(t):
        h = t / 3600.0
        diurno = amp_diurna * np.sin(2 * np.pi * (h - 8.0) / 24.0)
        rafaga = amp_rafagas * (0.6 * np.sin(2 * np.pi * h / 1.7)
                                + 0.4 * np.sin(2 * np.pi * h / 0.45 + 1.3))
        return float(np.clip(u_base + diurno + rafaga, 0.05, 1.0))
    return u
