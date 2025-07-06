import numpy as np
import plotly.graph_objs as go
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
# -*- coding: utf-8 -*-
"""PN接合のバリア電圧と空乏層の電場を可視化するDashアプリケーション
"""

# 物理定数
q = 1.602e-19  # [C]
k = 1.38e-23   # [J/K]
T = 300  # [K] (室温)
epsilon = 11.7 * 8.854e-12  # [F/m]
ni = 1.5e10  # [cm^-3]
Vbi = 0.7  # 内臓電位 [V]
Eg = 1.12  # バンドギャップ [eV]

# ドーピング濃度のマークを定義
# 1e14 ~ 1e18 cm^-3の範囲で5点を対数スケールで生成
marks_nd = {
    int(val): f"{int(val):.0e}" for val in np.logspace(14, 18, num=5)
}

# Dashアプリケーションのインスタンスを作成
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            # 左カラム（スライダー群）
            dbc.Col([
                html.H4("PN接合モデル", style={'fontWeight': 'bold'}),

                html.Br(),

                html.H5("接合タイプ選択", style={'fontWeight': 'bold'}),
                dcc.RadioItems(
                options=[
                    {'label': '階段接合', 'value': 'step'},
                    {'label': '傾斜接合', 'value': 'graded'}
                ],
                value='step',
                id='junction-type',
                labelStyle={'display': 'inline-block', 'marginRight': '20px'},
                style={'marginBottom': '20px'}
                ),

                html.Br(),
                
                html.Div([
                    dcc.Markdown("**$P$型ドーピング**  $N_A\\,[\\mathrm{cm}^{-3}]$", mathjax=True, style={'color': 'red'}),
                    dcc.Slider(
                        # 対数スケールのスライダー
                        min=16, max=17, step=0.1, value=16.5,
                        marks={i: f"1.0e+{i:.0f}" for i in range(16, 18)},
                        id='na-slider',
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '500px', 'marginBottom': '20px'}),

                html.Div([
                    dcc.Markdown("**$N$型ドーピング**  $N_D\\,[\\mathrm{cm}^{-3}]$", mathjax=True, style={'color': 'blue'}),
                    dcc.Slider(
                        # 対数スケールのスライダー
                        min=16, max=17, step=0.1, value=16.5,
                        marks={i: f"1.0e+{i:.0f}" for i in range(16, 18)},
                        id='nd-slider',
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '500px', 'marginBottom': '20px'}),

                html.Div([
                    dcc.Markdown("**印加電圧** $[V]$", mathjax=True, style={'color': 'black'}),
                    dcc.Slider(
                        min=-1.0, max=1.0, step=0.01, value=0.0,
                        marks={-1.0: '-1.0 V', 0.0: '0.0 V', 1.0: '1.0 V'},
                        id='voltage-slider',
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '500px', 'marginBottom': '20px'}),

                html.Br(),

                html.H5("各分布の導出式", style={'fontWeight': 'bold'}),
                # 数式を表示するためのMarkdownコンポーネント
                html.Div([dcc.Markdown(id='formula-text', mathjax=True)], style={'whiteSpace': 'pre-wrap', 'marginTop': '20px'}),
            ], width=4, style={'padding': '20px', 'borderRight': '1px solid #ddd'}),
            

            # 右カラム（グラフ）
            dbc.Col([
                html.Div(dcc.Graph(id='charge-graph', style={'width': '800px', 'height': '300px'}), style={'marginBottom': '0px'}),
                html.Div(dcc.Graph(id='field-graph', style={'width': '800px', 'height': '300px'}), style={'marginBottom': '0px'}),
                html.Div(dcc.Graph(id='potential-graph', style={'width': '800px', 'height': '300px'}), style={'marginBottom': '0px'}),
            ], width=8, style={'padding': '10px'}),
        ])
    ], fluid=True),

], style={'fontFamily': 'Meiryo'})

def compute_distributions(junction_type, nd, na, Vapp):

    # 空乏層幅の計算
    W_cm = ((2*epsilon/q*(1/nd+1/na)*(Vbi-Vapp))**0.5)*0.1 # 0.1はepsilonの[F/m]→[F/cm]の補正分
    xp_cm = W_cm*nd/(na+nd) # P型領域の空乏層幅
    xn_cm = W_cm*na/(na+nd) # N型領域の空乏層幅

    x_cm = np.linspace(-xp_cm, xn_cm, 500)  # 空乏層範囲

    if junction_type == 'step':
        N = np.where(x_cm < 0, -na, nd)
        E = np.zeros_like(x_cm)
        E[x_cm<0] = -q*na/epsilon*(x_cm[x_cm<0] + xp_cm)* 1e2  # [V/cm]に変換
        E[x_cm>=0] = q*nd/epsilon*(x_cm[x_cm>=0] - xn_cm)* 1e2  # [V/cm]に変換
        V = np.zeros_like(x_cm)
        V[x_cm<0] = (q*na/(2*epsilon))*((x_cm[x_cm<0] + xp_cm)**2 - (xp_cm)**2)* 1e2
        V[x_cm>=0] = (q*nd/(2*epsilon))*(-(x_cm[x_cm>=0] - xn_cm)**2 + (xn_cm)**2)* 1e2
        V_bi = k*T/q*np.log(nd*na/(ni**2))

        formula = fr"""
$$
\text{{電荷分布：}}N(x) = 
\begin{{cases}}
-N_A & (-x_p \leq x < 0) \\
+N_D & (0 < x \leq x_n)
\end{{cases}}
$$

$$
\text{{電界分布：}}E(x) =
\begin{{cases}}
-\dfrac{{qN_A(x+x_p)}}{{\epsilon_s}} & (-x_p \leq x < 0) \\
+\dfrac{{qN_D(x-x_n)}}{{\epsilon_s}} & (0 < x \leq x_n)
\end{{cases}}
$$

$$
\text{{電位分布：}}V(x) =
\begin{{cases}}
+\dfrac{{qN_A}}{{2\epsilon_s}}\left\{{(x+x_p)^2 - x_p^2\right\}} & (-x_p \leq x < 0) \\
-\dfrac{{qN_D}}{{2\epsilon_s}}\left\{{(x-x_n)^2 - x_n^2\right\}} & (0 < x \leq x_n)
\end{{cases}}
$$

$$
\text{{内蔵電位：}}V_{{bi}} = \dfrac{{kT}}{{q}}\ln\left(\dfrac{{N_AN_D}}{{{{n_i}}^2}}\right) = {V_bi:.2f}\,\mathrm{{V}}
$$
"""
    elif junction_type == 'graded':
        N = np.where(x_cm < 0, na*x_cm/xp_cm, nd*x_cm/xn_cm)
        E = np.zeros_like(x_cm)
        E[x_cm<0] = q*na/epsilon*((x_cm[x_cm<0])**2/(2*xp_cm) - xp_cm/2)* 1e2  # [V/cm]に変換
        E[x_cm>=0] = q*nd/epsilon*((x_cm[x_cm>=0])**2/(2*xn_cm) - xn_cm/2)* 1e2  # [V/cm]に変換
        V = np.zeros_like(x_cm)
        V[x_cm<0] = -q*na/epsilon*((x_cm[x_cm<0])**3/(6*xp_cm) - xp_cm*x_cm[x_cm<0]/2)* 1e2
        V[x_cm>=0] = -q*nd/epsilon*((x_cm[x_cm>=0])**3/(6*xn_cm) - xn_cm*x_cm[x_cm>=0]/2)* 1e2
        # V_bi = q*(nd*xn_cm**2+na*xp_cm**2)/(3*epsilon) * 1e2

        formula = fr"""
$$
\text{{電荷分布：}}N(x) = 
\begin{{cases}}
\dfrac{{N_A}}{{x_p}}x & (-x_p \leq x < 0) \\
\dfrac{{N_D}}{{x_n}}x & (0 < x \leq x_n)
\end{{cases}}
$$

$$
\text{{電界分布：}}E(x) =
\begin{{cases}}
\dfrac{{qN_A}}{{2\epsilon_s}}\left(\dfrac{{x^2}}{{x_p}}-x_p\right) & (-x_p \leq x < 0) \\
\dfrac{{qN_D}}{{2\epsilon_s}}\left(\dfrac{{x^2}}{{x_n}}-x_n\right) & (0 < x \leq x_n)
\end{{cases}}
$$

$$
\text{{電位分布：}}V(x) =
\begin{{cases}}
-\dfrac{{qN_A}}{{2\epsilon_s}}\left(\dfrac{{x^3}}{{3x_p}} - x_px\right) & (-x_p \leq x < 0) \\
-\dfrac{{qN_D}}{{2\epsilon_s}}\left(\dfrac{{x^3}}{{3x_n}} - x_nx\right) & (0 < x \leq x_n)
\end{{cases}}
$$
"""
    return x_cm, N, E, V, formula

    # 他のモデルも同様に実装可...

@app.callback(
    Output('charge-graph', 'figure')
    ,Output('field-graph', 'figure')
    ,Output('potential-graph', 'figure')
    ,Output('formula-text', 'children')
    ,Input('junction-type', 'value')
    ,Input('nd-slider', 'value')
    ,Input('na-slider', 'value')
    ,Input('voltage-slider', 'value')
)
def update_graph(jtype, nd_log, na_log, Vapp):
    # ドーピング濃度を対数スケールに調整
    nd = 10 ** nd_log
    na = 10 ** na_log

    x_cm, N, E, V, formula = compute_distributions(jtype, nd, na, Vapp)
    x = x_cm * 1e4  # [cm]から[μm]に変換
    E_C =-V+Eg/2
    E_V =-V-Eg/2
    x_left = x[x < 0]
    x_right = x[x >= 0]
    N_left = N[x < 0]
    N_right = N[x >= 0]
    E_left = E[x < 0]
    E_right = E[x >= 0]

    charge_fig = go.Figure()
    charge_fig.add_trace(go.Scatter(x=x_left, y=N_left, mode='lines', name='p型中性領域', line=dict(color='red'), fill='tozeroy'))
    charge_fig.add_trace(go.Scatter(x=x_right, y=N_right, mode='lines', name='n型中性領域', line=dict(color='blue'), fill='tozeroy'))
    charge_fig.update_layout(
        title={"text": "<b>電荷分布</b>", "font": {"size": 20}}
        ,xaxis=dict(range=[-0.3, 0.3])
        ,yaxis=dict(
            range=[-1e17, 1e17],
            tickformat=".1e",           # 小数1桁の指数表記
            exponentformat="power",     # 上付きの10^n形式で表示
            showexponent="all",         # 全ての目盛に指数を表示
        )
        ,xaxis_title="位置 [um]"
        ,yaxis_title={"text": "N_D-N_A [/cm³]", "font": {"family": "Meiryo", "size": 16}}
    )


    field_fig = go.Figure()
    field_fig.add_trace(go.Scatter(x=x_left, y=E_left, mode='lines', name='p型中性領域', line=dict(color='red'), fill='tozeroy'))
    field_fig.add_trace(go.Scatter(x=x_right, y=E_right, mode='lines', name='n型中性領域', line=dict(color='blue'), fill='tozeroy'))
    field_fig.update_layout(
        title={"text": "<b>電界分布</b>", "font": {"size": 20}}
        ,xaxis=dict(range=[-0.3, 0.3])
        ,yaxis=dict(
            range=[-1e5, 1e5],
            tickformat=".1e",           # 小数1桁の指数表記
            exponentformat="power",     # 上付きの10^n形式で表示
            showexponent="all",         # 全ての目盛に指数を表示
        )
        ,xaxis_title="位置 [um]"
        ,yaxis_title={"text": "E [V/cm]", "font": {"family": "Meiryo", "size": 16}}
    )

    potential_fig = go.Figure()
    potential_fig.add_trace(go.Scatter(x=x, y=E_C, mode='lines', name='伝導帯', line=dict(color='gray')))
    potential_fig.add_trace(go.Scatter(x=x, y=E_V, mode='lines', name='価電子帯', line=dict(color='black')))
    potential_fig.update_layout(
        title={"text": "<b>エネルギーバンド</b>", "font": {"size": 20}}
        ,xaxis=dict(range=[-0.3, 0.3])
        ,yaxis=dict(range=[-2, 2])
        ,xaxis_title="位置 [um]"
        ,yaxis_title={"text": "Energy [eV]", "font": {"family": "Meiryo", "size": 16}}
    )

    return charge_fig, field_fig, potential_fig, formula


if __name__ == '__main__':
    # app.run(debug=False)
    app.run(debug=True)