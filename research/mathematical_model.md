## Mathematical Model – Delay Propagation and Network Optimization

This document formalizes the key mathematical components used in the **Aviation Delay Intelligence System**.

### 1. Notation

- Let \( \mathcal{F} \) denote the set of flights.
- Let \( \mathcal{A} \) denote the set of airports.
- Each flight \( f \in \mathcal{F} \) has:
  - Origin \( o(f) \in \mathcal{A} \),
  - Destination \( d(f) \in \mathcal{A} \),
  - Scheduled departure time \( s_f \),
  - Flight duration \( \tau_f \),
  - Baseline (predicted) delay \( \hat{\Delta}_f \).
- Let \( \Delta_f \) be the **propagated delay** after network effects.

### 2. Delay Prediction

We learn a function \( g(\cdot) \) using Random Forest regression:

\[
\hat{\Delta}_f = g(w_f, c_f, p_f, \ell_f, u_f),
\]

where:

- \( w_f \) – weather severity,
- \( c_f \) – congestion level,
- \( p_f \) – previous delay,
- \( \ell_f \) – route distance,
- \( u_f \) – airport load.

This provides a baseline expected delay **without** explicit network propagation.

### 3. Exogenous Shock

An exogenous shock at airport \( a^\* \) (e.g., a storm) is modelled as an additional delay \( S \) minutes:

\[
\Delta_f^{(0)} =
\begin{cases}
\hat{\Delta}_f + S, & \text{if } o(f) = a^\*, \\
\hat{\Delta}_f, & \text{otherwise}.
\end{cases}
\]

This initializes the propagated delay.

### 4. Delay Propagation Dynamics

We consider two main propagation channels:

1. **Same-Aircraft Continuation (Tail Rotations).**
2. **Passenger Connections at Airports.**

#### 4.1 Time Model

For each flight \( f \), we define:

\[
t_f^{\text{dep}} = s_f,\quad
t_f^{\text{arr}} = s_f + \tau_f + \Delta_f.
\]

The connection time between flights \( i \) and \( j \) is:

\[
\Delta t_{ij} = t_j^{\text{dep}} - t_i^{\text{arr}}.
\]

#### 4.2 Propagation Equation

We use an **exponential-decay propagation model**:

\[
\Delta_j \mathrel{+}= \Delta_i \cdot \alpha \cdot e^{-\Delta t_{ij} / \beta},
\]

where:

- \( \alpha \in [0.5, 0.8] \) is the **propagation factor**,
- \( \beta > 0 \) is the **decay constant** in minutes,
- \( \Delta t_{ij} \ge 0 \) is the connection gap; negative values are clamped to zero.

Propagation is repeated iteratively until convergence:

\[
\max_{f \in \mathcal{F}} \left| \Delta_f^{(k+1)} - \Delta_f^{(k)} \right| < \varepsilon
\]

or a maximum number of iterations is reached.

#### 4.3 Congestion and Passenger Multipliers

We incorporate:

- Airport congestion \( c_f \) and load \( u_f \),
- Passenger connections \( P_f \).

An **amplification factor** for flight \( j \) is:

\[
\gamma_j = 1 + 0.5 c_j + 0.5 u_j.
\]

Passenger-driven propagation from flight \( i \) to \( j \) (sharing an airport) within a window \( W \) is:

\[
\Delta_j \mathrel{+}= \Delta_i \cdot \alpha \cdot \left(\frac{P_i}{P_{\max}}\right) \cdot e^{-\Delta t_{ij} / \beta},
\quad \text{if } 0 \le \Delta t_{ij} \le W.
\]

The final update at each iteration becomes:

\[
\Delta_j^{(k+1)} = \hat{\Delta}_j + \gamma_j \cdot \left( \sum_{i \in \mathcal{N}_j^{\text{tail}}} \alpha e^{-\Delta t_{ij}/\beta} \Delta_i^{(k)} + \sum_{i \in \mathcal{N}_j^{\text{pass}}} \alpha \frac{P_i}{P_{\max}} e^{-\Delta t_{ij}/\beta} \Delta_i^{(k)} \right),
\]

where:

- \( \mathcal{N}_j^{\text{tail}} \) – set of inbound flights on the same aircraft,
- \( \mathcal{N}_j^{\text{pass}} \) – set of inbound flights with passenger connections to \( j \).

### 5. Graph-Based Optimization

#### 5.1 Dijkstra – Least-Delay Passenger Path

We represent the network as a directed graph \( G = (V, E) \) where:

- \( V \) – airports,
- \( E \) – flights.

The **effective travel time** for an edge (flight) \( e \) is:

\[
w_e = \tau_e + \Delta_e.
\]

Given origin–destination airports \( (s, t) \), Dijkstra’s algorithm finds a path \( \pi^\* \) minimizing:

\[
T(\pi) = \sum_{e \in \pi} w_e.
\]

#### 5.2 Kruskal – Minimum Spanning Connectivity

We construct an undirected graph with the same vertices and edge weights \( w_e \). Kruskal’s algorithm yields a **minimum spanning tree (MST)**:

\[
\text{MST} = \arg \min_{T \subseteq G} \sum_{e \in T} w_e,
\]

which serves as a low-delay connectivity backbone.

### 6. Linear Programming – Network Control

We introduce binary decision variables:

\[
x_f =
\begin{cases}
1, & \text{if flight } f \text{ is operated}, \\
0, & \text{if flight } f \text{ is cancelled/retimed}.
\end{cases}
\]

Each flight has:

- Propagated delay \( \Delta_f \),
- Distance \( \ell_f \),
- Number of connecting passengers \( P_f \).

We define a **composite cost**:

\[
C_f = w_d \Delta_f + w_f \ell_f + w_p P_f,
\]

where \( w_d, w_f, w_p \) are weights for delay, fuel proxy, and passenger disruption.

#### Objective

\[
\min \sum_{f \in \mathcal{F}} x_f C_f.
\]

#### Connectivity Constraint

We require at least a fraction \( \eta \) of flights to operate:

\[
\sum_{f \in \mathcal{F}} x_f \ge \eta |\mathcal{F}|.
\]

This constraint approximates **robust connectivity** while allowing controlled cancellations.

### 7. Key Metrics

- **Total network delay before propagation:**
\[
D_{\text{before}} = \sum_{f \in \mathcal{F}} \Delta_f^{(0)}.
\]

- **Total network delay after propagation:**
\[
D_{\text{after}} = \sum_{f \in \mathcal{F}} \Delta_f^{(\text{final})}.
\]

- **Network delay reduction (percentage):**
\[
\text{Reduction}_{\text{network}} = 100 \cdot \frac{D_{\text{before}} - D_{\text{after}}}{D_{\text{before}}}.
\]

- **LP objective before control** (all flights operated):
\[
J_{\text{before}} = \sum_{f \in \mathcal{F}} C_f.
\]

- **LP objective after control:**
\[
J_{\text{after}} = \sum_{f \in \mathcal{F}} x_f^\* C_f.
\]

- **LP improvement (percentage):**
\[
\text{Improvement}_{\text{LP}} = 100 \cdot \frac{J_{\text{before}} - J_{\text{after}}}{J_{\text{before}}}.
\]

These formulations make the system suitable for theoretical analysis, parameter studies, and comparison against alternative models in future research.

