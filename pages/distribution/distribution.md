<|toggle|theme|>
<|menu|label=MMM Modeling|lov={menu_lov}|>
## Distribution priors for Bayesian Modelling
<|{dis_show_chart}|toggle|lov={dis_choice_chart}|>
### Distribution
<|part|render={dis_show_chart == 'Uniform'}|
<|layout|columns=1 1|
<|first column
<|container container-styling|
Lower Bound: <|{lower}|> <br/>
<|{lower}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
<|second column
<|container container-styling|
Upper Bound: <|{upper}|> <br/>
<|{upper}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
|>
|>

<|part|render={dis_show_chart == 'Normal'}|
<|layout|columns=1 1|
<|first column
<|container container-styling|
mu: <|{mu}|> <br/>
<|{mu}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
<|second column
<|container container-styling|
sigma: <|{sigma}|> <br/>
<|{sigma}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
|>
|>

<|part|render={dis_show_chart == 'HalfNormal'}|
sigma: <|{sigma}|> <br/>
<|{sigma}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>

<|part|render={dis_show_chart == 'Beta'}|
<|layout|columns=1 1|
<|first column
<|container container-styling|
alpha: <|{alpha}|> <br/>
<|{alpha}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
<|second column
<|container container-styling|
beta: <|{beta}|> <br/>
<|{beta}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
|>
|>

<|part|render={dis_show_chart == 'Gamma'}|
<|layout|columns=1 1|
<|first column
<|container container-styling|
Alpha (Shape): <|{gmmalpha}|> <br/>
<|{gmmalpha}|slider|min=0.01|max=15|step=0.01|on_change=update_distribution|>
|>
|>
<|second column
<|container container-styling|
Beta (rate): <|{gmmbeta}|> <br/>
<|{gmmbeta}|slider|min=0|max=10|step=0.01|on_change=update_distribution|>
|>
|>
|>
|>

<|part|render={dis_show_chart == 'Poisson'}|
Lambda: <|{poisonlam}|> <br/>
<|{poisonlam}|slider|min=0|max=15|step=0.5|on_change=update_distribution|>
|>

<|part|render={dis_show_chart == 'Bernoulli'}|
rate: <|{p}|> <br/>
<|{p}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>

<|part|render={dis_show_chart == 'Exponential'}|
lambda: <|{lam}|> <br/>
<|{lam}|slider|min=0|max=15|step=0.05|on_change=update_distribution|>
|>

<|part|render={dis_show_chart == 'Weibull'}|
<|layout|columns=1 1|
<|first column
<|container container-styling|
Shape (K): <|{walpha}|> <br/>
<|{walpha}|slider|min=0|max=10|step=0.01|on_change=update_distribution|>
|>
|>
<|second column
<|container container-styling|
Scale (lam): <|{wbeta}|> <br/>
<|{wbeta}|slider|min=0|max=10|step=0.01|on_change=update_distribution|>
|>
|>
|>
|>

<|part|render={dis_show_chart == 'TruncatedNormal'}|
<|layout|columns=1 1 1 1|
<|first column
<|container container-styling|
mu: <|{tmu}|> <br/>
<|{tmu}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
<|second column
<|container container-styling|
sigma: <|{tsigma}|> <br/>
<|{tsigma}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>

<|third column
<|container container-styling|
Lower: <|{tlower}|> <br/>
<|{tlower}|slider|min=-10|max=10|step=0.1|on_change=update_distribution|>
|>
|>

<|fourth column
<|container container-styling|
Upper: <|{tupper}|> <br/>
<|{tupper}|slider|min=0|max=10|step=0.1|on_change=update_distribution|>
|>
|>
|>
|>

<|part|render={dis_show_chart == 'StudentT'}|
<|layout|columns=1 1 1|
<|first column
<|container container-styling|
nu: <|{nu}|> <br/>
<|{nu}|slider|min=0|max=15|step=0.1|on_change=update_distribution|>
|>
|>

<|second column
<|container container-styling|
mu: <|{stmu}|> <br/>
<|{stmu}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
<|third column
<|container container-styling|
sigma: <|{stsigma}|> <br/>
<|{stsigma}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
|>
|>

<|part|render={dis_show_chart == 'LogNormal'}|
<|layout|columns=1 1|
<|first column
<|container container-styling|
mu: <|{lmu}|> <br/>
<|{lmu}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
<|second column
<|container container-styling|
sigma: <|{lsigma}|> <br/>
<|{lsigma}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
|>
|>


<|chart|figure={get_selected_chart(dis_show_chart, lower, upper, mu, sigma, balpha, bbeta, gmmalpha,gmmbeta, poisonlam, p, lam, walpha, wbeta, tmu, tsigma, tlower, tupper,nu, stmu, stsigma, lmu, lsigma)}|height=500px|on_change=update_distribution|>