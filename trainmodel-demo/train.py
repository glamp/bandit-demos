from bandit import *
import datetime
import pandas as pd
import numpy as np
import time
import statsmodels.formula.api as sm
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

bandit = Bandit()

df = pd.DataFrame({ \
    "A": np.random.normal(100,10,50).tolist(), \
    "B": np.random.normal(50,5,50).tolist(), \
    "C": np.random.exponential(5,50).tolist() \
})
result = sm.ols(formula="A ~ B + C", data=df).fit()

metadata = {'R2': result.rsquared, 'AIC': result.aic}

with open(bandit.output_dir + 'model_stats.txt', "w") as text_file:
    model_summary = str(result.summary())
    text_file.write(model_summary)

# bandit = Bandit()
#
for x in range(10):
    for y in range(10):
        bandit.report('tag', float(np.log((10/(y+1)*10)) + np.random.rand()))
        time.sleep(0.1)


bandit.metadata.R2 = result.rsquared
bandit.metadata.AIC = result.aic

chart = sns.distplot(df.A)
chart.figure.savefig(bandit.output_dir + 'dist.png')

# save
df.head().to_csv(bandit.output_dir + 'datasample.csv')

today = datetime.date.today().strftime('%Y_%m_%d')

email = Email()
email.subject = '%s model results' % today

body = '''
Below is the result of the successful nightly model training script

Model Stats: %s
- Model:
- Adj. R2:
''' % result.model.formula, result.rsquared_adj

email.body = body
email.add_attachment(bandit.output_diroutput_dir + 'datasample.csv')
email.add_attachment(bandit.output_diroutput_dir + 'model_stats.txt')
email.add_attachment(bandit.output_diroutput_dir + 'dist.png')
email.send('colin@yhathq.com')
