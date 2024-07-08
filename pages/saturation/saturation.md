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
<|{root_alpha}|slider|min=0|max=1|step=0.01|>
|>

<|part|render={display_saturation == 'Hill'}|
### Hill Curve Saturation
Try to fit a saturation curve to the generated data! <br/>
<|layout|columns=1 1|
<|first colum
<|container container-styling|
Alpha &alpha;: <|{hill_alpha}|> <br/>
<|{hill_alpha}|slider|min=0|max=10|step=0.01|>
|>
|>
<|second column
<|container container-styling|
Gamma &gamma;: <|{hill_gamma}|> <br/>
<|{hill_gamma}|slider|min=0|max=1000|>
|>
|>
|>
|>

<|part|render={display_saturation == 'Logistic'}|
### Logistic Curve Saturation
Try to fit a saturation curve to the generated data! <br/>
Lambda &lambda;: <|{logistic_lam}|> <br/>
<|{logistic_lam}|slider|min=0|max=1000|>
|>

<|part|render={display_saturation == 'Tanh'}|
### Tanh Curve Saturation
Try to fit a saturation curve to the generated data! <br/>
<|layout|columns=1 1|
<|first colum
<|container container-styling|
Tanh b: <|{tanh_b}|> <br/>
<|{tanh_b}|slider|min=0|max=20|>
|>
|>
<|second column
<|container container-styling|
Tanh c: <|{tanh_c}|> <br/>
<|{tanh_c}|slider|min=0|max=100|>
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
<|{mm_alpha}|slider|min=0|max=50|>
|>
|>
<|second column
<|container container-styling|
Lambda &lambda;: <|{mm_lam}|> <br/>
<|{mm_lam}|slider|min=0|max=500|>
|>
|>
|>
|>


<|chart|figure={display_curve(display_saturation, media_spending, root_alpha, hill_alpha, hill_gamma, logistic_lam, tanh_b, tanh_c, mm_alpha, mm_lam)}|height=500px|>
