import pandas as pd
import numpy as np
import plotly.graph_objects as go
from taipy.gui import Gui, Icon, State, navigate, notify, Markdown
from mmm_functions import (
    root_saturation,
    threshold_hill_saturation,
    logistic_saturation,
    tanh_saturation,
    michaelis_menten_saturation
)



# Generate simulated marketing data
np.random.seed(42)
num_points = 500
media_spending = np.linspace(0, 1000, num_points) # x-axis
#initial parameters
root_alpha = 0.45
hill_alpha = 0.45
hill_gamma = 100
logistic_lam = 500
tanh_b = 5
tanh_c = 50
mm_alpha = 25
mm_lam = 50

# Generate simulated datasets with noise
dummy_root = root_saturation(media_spending, alpha = 0.3) + np.random.normal(0, 0.3, num_points)
dummy_hill = threshold_hill_saturation(media_spending, alpha = 8, gamma = 400, threshold=200) + np.random.normal(0, 0.05, num_points)
dummy_logistic = logistic_saturation(media_spending, lam = 0.01) + np.random.normal(0, 0.1, num_points)
dummy_tanh = tanh_saturation(media_spending, b = 10, c = 20) + np.random.normal(0, 0.75, num_points)
dummy_m_m = michaelis_menten_saturation(media_spending, alpha = 20, lam = 200) + np.random.normal(0, 2, num_points)

def display_plot(spending, dummy, user_fun, title_text):
    plot_data = pd.DataFrame({'Media Spending':np.round(spending), 
                            'Conversions':dummy})
    plot_data = plot_data[plot_data.Conversions >= 0]
    
    # Plot
    fig_root = go.Figure()
    # Plot weekly spend and response data, every 5th to make the plot less crowded
    fig_root.add_trace(go.Scatter(x = plot_data['Media Spending'][::5],y = plot_data['Conversions'][::5],
                            mode = 'markers',name = 'Weekly Data', marker = dict(color='#AB63FA')))
    # Plot user-defined curve to match that data
    fig_root.add_trace(go.Scatter(x=media_spending,y=user_fun,mode='lines',
                                  name='Saturation Curve', line=dict(color='blue', dash='solid')))
    fig_root.update_layout(title_text=f'{title_text} Curve Saturation', xaxis_title='Media Spend', yaxis_title='Conversions')
    
    return fig_root


def root_curve(media_spending, root_alpha, title_text):
    user_root = root_saturation(media_spending, alpha=root_alpha)
    return display_plot(spending = media_spending, dummy= dummy_root, user_fun= user_root, title_text=title_text)

def hii_curve(media_spending, hill_alpha, hill_gamma, title_text):
    user_hill = threshold_hill_saturation(media_spending, alpha = hill_alpha, gamma = hill_gamma)
    return display_plot(spending = media_spending, dummy= dummy_hill, user_fun= user_hill, title_text=title_text)

def logit_curve(media_spending, logistic_lam, title_text):
    logistic_lam = logistic_lam/10000
    user_logistic = logistic_saturation(media_spending, lam=logistic_lam)
    return display_plot(spending=media_spending, dummy=dummy_logistic, user_fun=user_logistic, title_text=title_text)

def tanh_curve(media_spending, tanh_b, tanh_c, title_text):
    user_tanh = tanh_saturation(media_spending, b=tanh_b, c=tanh_c)
    return display_plot(spending=media_spending, dummy=dummy_tanh, user_fun=user_tanh, title_text=title_text)

def mm_curve(media_spending, mm_alpha, mm_lam, title_text):
    user_mm = michaelis_menten_saturation(media_spending, alpha=mm_alpha, lam=mm_lam)
    return display_plot(spending=media_spending, dummy=dummy_m_m, user_fun=user_mm, title_text=title_text)

def display_curve(display_saturation, media_spending, root_alpha, hill_alpha, hill_gamma, logistic_lam, tanh_b, 
                       tanh_c, mm_alpha, mm_lam):
    if display_saturation == 'Root':
        return root_curve(media_spending, root_alpha, title_text= display_saturation)
    elif display_saturation == 'Hill':
        return hii_curve(media_spending, hill_alpha, hill_gamma, title_text=display_saturation)
    elif display_saturation== 'Logistic' :
        return logit_curve(media_spending, logistic_lam, title_text=display_saturation)
    elif display_saturation == 'Tanh':
        return tanh_curve(media_spending, tanh_b, tanh_c, title_text=display_saturation)
    elif display_saturation == 'Michaelis-Menten':
        return mm_curve(media_spending, mm_alpha, mm_lam, title_text=display_saturation)
    
def update_saturation(state):
    media_spending = state.media_spending
    root_alpha = float(state.root_alpha)
    hill_alpha = float(state.hill_alpha)
    hill_gamma = state.hill_gamma
    logistic_lam = state.logistic_lam
    tanh_b = state.tanh_b
    tanh_c = state.tanh_c
    mm_alpha =  float(state.mm_alpha)
    mm_lam = state.mm_lam
    display_curve(state.display_saturation, media_spending, root_alpha, hill_alpha, hill_gamma, logistic_lam, tanh_b, tanh_c, mm_alpha, mm_lam)

saturation_types = ["Root", "Hill", "Logistic", "Tanh", "Michaelis-Menten"]
display_saturation = saturation_types[0]

saturation_layout = """

<|toggle|theme|>

## Saturation functions for MMM Bayesian Modelling
This page illustrates the types and shapes of saturation curves for MMM. These curves attempt to represent the link between weekly marketing 
spends for a specific channel (while keeping other channels constant) and the conversions that follow from that spend. <br/>
It does not have to be conversions; it might be revenue/sales or customers acquired - whatever the **Business Metric**.
<br/>
<|{display_saturation}|toggle|lov={saturation_types}|>
<|part|render={display_saturation == 'Root'}|
### Root Curve Saturation
Try to fit a saturation curve to the generated data! <br/>
Alpha &alpha;: <|{root_alpha}|> <br/>
<|{root_alpha}|slider|min=0|max=1|step=0.01|on_change=update_saturation|>
|>

<|part|render={display_saturation == 'Hill'}|
### Hill Curve Saturation
Try to fit a saturation curve to the generated data! <br/>
<|layout|columns=1 1|
<|first colum
<|container container-styling|
Alpha &alpha;: <|{hill_alpha}|> <br/>
<|{hill_alpha}|slider|min=0|max=10|step=0.01|on_change=update_saturation|>
|>
|>
<|second column
<|container container-styling|
Gamma &gamma;: <|{hill_gamma}|> <br/>
<|{hill_gamma}|slider|min=0|max=1000|on_change=update_saturation|>
|>
|>
|>
|>

<|part|render={display_saturation == 'Logistic'}|
### Logistic Curve Saturation
Try to fit a saturation curve to the generated data! <br/>
Lambda &lambda;: <|{logistic_lam}|> <br/>
<|{logistic_lam}|slider|min=0|max=1000|on_change=update_saturation|>
|>

<|part|render={display_saturation == 'Tanh'}|
### Tanh Curve Saturation
Try to fit a saturation curve to the generated data! <br/>
<|layout|columns=1 1|
<|first colum
<|container container-styling|
Tanh b: <|{tanh_b}|> <br/>
<|{tanh_b}|slider|min=0|max=20|on_change=update_saturation|>
|>
|>
<|second column
<|container container-styling|
Tanh c: <|{tanh_c}|> <br/>
<|{tanh_c}|slider|min=0|max=100|on_change=update_saturation|>
|>
|>
|>
|>

<|part|render={display_saturation == 'Michaelis-Menten'}|
### Michaelis-Menten Curve Saturation
Try to fit a saturation curve to the generated data! <br/>
<|layout|columns=1 1|
<|first colum
<|container container-styling|
Alpha &alpha;: <|{mm_alpha}|> <br/>
<|{mm_alpha}|slider|min=0|max=50|on_change=update_saturation|>
|>
|>
<|second column
<|container container-styling|
Lambda &lambda;: <|{mm_lam}|> <br/>
<|{mm_lam}|slider|min=0|max=500|on_change=update_saturation|>
|>
|>
|>
|>


<|chart|figure={display_curve(display_saturation, media_spending, root_alpha, hill_alpha, hill_gamma, logistic_lam, tanh_b, tanh_c, mm_alpha, mm_lam)}|height=500px|on_change=update_saturation|>

"""

if __name__=='__main__':
     Gui(saturation_layout).run(use_reloader=True, port=5001)