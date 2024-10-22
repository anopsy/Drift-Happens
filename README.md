# Drift Happens: NannyML’s Guide to Model Monitoring and Covariate Drift

> **LinkedIn summary:** Drift happens. Learn how NannyML can help you monitor and prevent univariate and multivariate data drift in your machine learning models. Check out our latest blog post to see how NannyML works on a use case and keep your models on track.
> 

## **Introduction**

So you just did the job. You gathered some data, put a lot of effort into cleaning and wrangling to get the most out of it. After thorough experimentation, you’ve built a model; you tuned the heck out of your hyperparameters and achieved satisfying performance. It’s time to put the model into production and turn its predictions into business value. But then—sooner or later, the results start to [**deteriorate¹**](https://www.nature.com/articles/s41598-022-15245-z). The decisions made based on your model’s predictions don’t bring any improvement or even cause losses. You need answers: what went wrong?

### **The Zillow Example**

A similar thing happened at Zillow in 2021. The company providing real estate information to buyers had to close its house-flipping department and lay off 25% of its workforce due to the failure of the Zestimate algorithm. They reported a loss of more than [**$600M²**](https://www.deeplearning.ai/the-batch/price-prediction-turns-perilous/). The Zestimate algorithm was proprietary software that underwent a serious revamping, including employing Deep Neural Networks and NLP. However, the details of why it failed have never been disclosed. It has been speculated that the problems arose from manifold factors, including data latency and lack of robustness to handle the volatility in the market caused by the pandemic.

### **Monitoring Workflow**

It probably could have been prevented. By thorough monitoring of the model’s performance, you can mitigate model decay. You are able to counteract the results of data latency. You can also ensure that your model is robust enough to withstand volatility in the market you’re aiming to model. With a good monitoring system at your disposal, you should be able to track your model's performance and, whenever it deteriorates, perform Root Cause Analysis, which is crucial to resolve the issues that might have arisen.

![Fig.1 Monitoring Workflow that could spare you a lot of sleepless nights. Image by the author.](Drift%20Happens%20NannyML%E2%80%99s%20Guide%20to%20Model%20Monitoring%20%2012158e349680803cbf01fb7a88573790/ML_Monitoring_Workflow.png)

Fig.1 Monitoring Workflow that could spare you a lot of sleepless nights. Image by the author.

## **Reasons for Model Failure**

A machine learning model can fail in different ways and for different reasons. It can stop producing output due to a bug in the infrastructure, or it can produce inaccurate output. The reason can be hidden in the poor scoping of the project; in that case, the data at hand is an inaccurate proxy for the reality we’re attempting to model. But there’s another thing we can blame for the failure - when reality and, thus, our data change, and our model isn’t able to make accurate predictions based on the new distributions of our variables and relationships between the variables. In this blog, we will focus on data drift, also known as covariate shift, and how to prevent it using NannyML.

### **Univariate Data Drift**

Univariate data drift means that there have been changes in the probability distributions of a certain variable - P(X). This happens pretty often, as data is a proxy for the ever-changing reality. Those changes can be due to seasonality, new trends, or expanding markets. That’s why it’s important to monitor those changes and stay aware of them. Once detected, univariate data drift doesn’t necessarily mean trouble. If our production data contains more data points in the areas where our model is more certain, the drift won’t have any negative impact on the model’s performance. The situation changes if the production data shifts to regions that were under-represented during training or moves to less certain regions close to the decision boundary. Thus, triggering alerts every time univariate data drift occurs can cause alert fatigue—a situation where the team starts to ignore the warnings, which can be the true danger of univariate data drift.

### **Multivariate Data Drift**

But data drift can also occur when there’s no apparent change in the distributions of single covariates. That’s a very peculiar situation when the P(X) of a single variable doesn’t change, but the correlation between multiple variables does. Therefore, detecting multivariate data drift is a bit trickier than comparing the probability distributions of a single variable.

![Fig. 2 Illustration of Multivariate Data Drift from [“**A Comprehensive Evaluation of Data Drift Metrics for Air Transportation Applications”](https://www.researchgate.net/publication/382114073_A_Comprehensive_Evaluation_of_Data_Drift_Metrics_for_Air_Transportation_Applications) by Pablo Gasco and [Ramon Dalmau-Codina](https://www.researchgate.net/profile/Ramon-Dalmau-Codina?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6Il9kaXJlY3QiLCJwYWdlIjoicHVibGljYXRpb24iLCJwcmV2aW91c1BhZ2UiOiJfZGlyZWN0In19)**](Drift%20Happens%20NannyML%E2%80%99s%20Guide%20to%20Model%20Monitoring%20%2012158e349680803cbf01fb7a88573790/image.png)

Fig. 2 Illustration of Multivariate Data Drift from [“**A Comprehensive Evaluation of Data Drift Metrics for Air Transportation Applications”](https://www.researchgate.net/publication/382114073_A_Comprehensive_Evaluation_of_Data_Drift_Metrics_for_Air_Transportation_Applications) by Pablo Gasco and [Ramon Dalmau-Codina](https://www.researchgate.net/profile/Ramon-Dalmau-Codina?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6Il9kaXJlY3QiLCJwYWdlIjoicHVibGljYXRpb24iLCJwcmV2aW91c1BhZ2UiOiJfZGlyZWN0In19)**

## NannyML to the Rescue

NannyML is a powerful post-deployment Data Science tool that helps keep an eye on your model’s performance, discover when your model is experiencing problems, and detect the root cause of those issues. Let’s have a look at how it works.

For this purpose, I used a dataset I initially obtained from Sourcestack while participating in a hackathon last year. I tried to answer the following question: “Did ChatGPT replace interns and juniors in engineering jobs?” This year, I obtained another sample of the data, which makes it a perfect dataset to compare if the model trained on data from last year experienced any trouble from univariate or multivariate data drift due to data latency.

The dataset contains information about vacancies, specifically engineering jobs. It consists of variables describing the hourly type of the job, whether the job is remote, what education is required, the seniority expected from the candidate, and the country where the job offer is available. The target variable is compensation estimation in dollars.

To run univariate and multivariate drift detection algorithms, we need a reference dataset and an analysis dataset. The reference dataset is our baseline performance we will compare the analysis dataset to. We should use our test dataset as the reference and not the training dataset. The analysis dataset consists of our production data, the data we want to monitor. I load both datasets.

```python
import pandas as pd
import nannyml as nml
from IPython.display import display
```

```python
reference = pd.read_csv('data/reference.csv')
analysis = pd.read_csv('data/analysis.csv')
```

```python
reference['job_published_at'] = pd.to_datetime(reference['job_published_at'])
analysis['job_published_at'] = pd.to_datetime(analysis['job_published_at'])
```

## **Univariate Drift Detection**

Now we can start with the Univariate Drift Detection. I initialize the calculator, set the chunk size, specify categorical variables, and the timestamp column. I also pass statistical methods that the calculator can utilize to measure the distance between the data chunks from my reference and analysis datasets. I use a versatile Jensen-Shannon metric that can assess both categorical and continuous distributions and a chi2 statistical test. Although we’re only interested in univariate data drift, I thought it would be interesting to inspect the distribution of our target variable as this is the only continuous variable in this dataset. For this purpose, I employ the Kolmogorov-Smirnov Test and the aforementioned Jensen-Shannon metric. After fitting the UnivariateDriftCalculator on the reference set, I can now feed it to the calculate method and visualize the results.

```python
ud_calc = nml.UnivariateDriftCalculator(
    column_names=reference.columns.to_list(),
    treat_as_categorical = ['hours', 'remote', 'education', 'seniority', 'country'],
    timestamp_column_name='job_published_at',
    continuous_methods=['kolmogorov_smirnov', 'jensen_shannon'],
    categorical_methods=['chi2', 'jensen_shannon'], 
)

ud_calc.fit(reference)
results = ud_calc.calculate(analysis)
```

```python
figure = results.filter(
	column_names=results.categorical_column_names, 
	methods=['jensen_shannon']
	).plot(kind='drift')
figure.show()
```

![Fig. 3 Output of UnivariateDriftCalculator for the categorical variables, kind=drift.](Drift%20Happens%20NannyML%E2%80%99s%20Guide%20to%20Model%20Monitoring%20%2012158e349680803cbf01fb7a88573790/newplot_(2).png)

Fig. 3 Output of UnivariateDriftCalculator for the categorical variables, kind=drift.

From the plots, we can see that the last chunks of the dataset consistently triggered a drift alert. This is due to the sampling effect—the last chunk consists of only 1 data point. The only variable where univariate drift occurred is the ‘hours’ feature. To better understand the distributions of our categorical covariates, let’s have a look at them.

```python
figure = results.filter(
			column_names=results.categorical_column_names, 
			methods=['jensen_shannon']
			).plot(kind='distribution')
figure.show()
```

![Fig. 4 Output of UnivariateDriftCalculator for the categorical variables - kind=distribution.](Drift%20Happens%20NannyML%E2%80%99s%20Guide%20to%20Model%20Monitoring%20%2012158e349680803cbf01fb7a88573790/newplot_(3).png)

Fig. 4 Output of UnivariateDriftCalculator for the categorical variables - kind=distribution.

Again, we see the impact of the sampling effect on the drift alert. The distributions for most covariates changed slightly but not enough to trigger a drift warning. We are also able to spot the cause of data drift. In two data chunks in the 'hours' feature, the number of data points from 'part-time' and 'full-time' categories increased, and the number of occurrences of the 'unclear' category decreased. 

Let’s now have a look at the side-by-side comparison of distributions of our only continuous variable, the predicted values (y_pred) and the ground-truth values (comp_dol).

```python
figure = results.filter(
	column_names=results.continuous_column_names, 
	methods=['kolmogorov_smirnov']
	).plot(kind='distribution')
figure.show()
```

![Fig. 5 Output of UnivariateDriftCalculator for the target variable, kind=distribution](Drift%20Happens%20NannyML%E2%80%99s%20Guide%20to%20Model%20Monitoring%20%2012158e349680803cbf01fb7a88573790/newplot_(3)%201.png)

Fig. 5 Output of UnivariateDriftCalculator for the target variable, kind=distribution

Again, we can notice minor changes in distribution patterns, yet they remain consistent across both the reference and analysis sets.

```python
figure = results.filter(
	column_names=results.continuous_column_names, 
	methods=['kolmogorov_smirnov']
	).plot(kind='drift')
figure.show()
```

![Fig. 6 Output of UnivariateDriftCalculator for the target variable, kind = drift.](Drift%20Happens%20NannyML%E2%80%99s%20Guide%20to%20Model%20Monitoring%20%2012158e349680803cbf01fb7a88573790/newplot_(4).png)

Fig. 6 Output of UnivariateDriftCalculator for the target variable, kind = drift.

As we can see, there has been no univariate drift in the input variables' distributions or the output, except for the last smaller chunk.

## **Multivariate Drift Detection**

However, no univariate drift doesn’t mean that our model won’t suffer from multivariate data drift, mentioned earlier in this blog post. Unfortunately, detecting this type of drift isn’t as straightforward as applying some basic statistical tests. NannyML offers two methods to capture occurrences of multivariate data drift.

The Data Reconstruction Drift Calculator method uses a compression algorithm that reduces the dimensionality of data, transforming it into a new set of uncorrelated variables, called principal components, which retain the most important information while discarding noise and redundancy. The trick is to reverse that compression and compare the original and decompressed dataset and compute the decompression error—how much information has been lost. If we do that to our analysis dataset, we can then compare the error to our baseline dataset. If the error exceeds a given threshold, it means there must have been changes in the correlation between the variables of our dataset.

Another way of detecting multivariate data drift is the Domain Classifier Calculator. This method utilizes the LGBM Classifier to discern between data samples coming from our reference and analysis datasets. The AUROC metrics give insight into how easy it is for the model to differentiate between those two datasets. If it’s easy (high AUROC value), it means the multivariate data shift occurred in our production dataset. If the model has trouble telling the difference, it means there haven’t been any significant changes to our datasets.

Does the sample from 2024 show signs of multivariate data drift?

### **Data Reconstruction Drift Calculator**

Let’s initialize the Data Reconstruction Drift Calculator, passing the column names we want to monitor, timestamp data, and chunk period or, in this case, chunk size. Let’s fit it on the reference dataset and calculate the reconstruction error for the analysis dataset.

```python
drd_calc = nml.DataReconstructionDriftCalculator(
    column_names=reference.columns.to_list(),
    timestamp_column_name='job_published_at',
    chunk_size=200
)
reference['job_published_at']=reference['job_published_at'].astype('object')
analysis['job_published_at']=analysis['job_published_at'].astype('object')
```

```python
drd_calc.fit(reference)
results = drd_calc.calculate(analysis)
```

```python
figure = results.plot()
figure.show()
```

![Fig. 7 Output from DataReconstructionDriftCalculator for chunk_size=200 ](Drift%20Happens%20NannyML%E2%80%99s%20Guide%20to%20Model%20Monitoring%20%2012158e349680803cbf01fb7a88573790/newplot_(6).png)

Fig. 7 Output from DataReconstructionDriftCalculator for chunk_size=200 

The reconstruction error exceeded the threshold in two data chunks. The difference from the reconstruction error in our reference dataset isn’t huge. Let’s check the results for a chunk size of 500 to exclude the sampling effect as the culprit.

```python
drd_calc = nml.DataReconstructionDriftCalculator(
    column_names=reference.columns.to_list(),
    timestamp_column_name='job_published_at',
    chunk_size=500
)
```

![Fig. 8 Output from DataReconstructionDriftCalculator for chunk_size=500](Drift%20Happens%20NannyML%E2%80%99s%20Guide%20to%20Model%20Monitoring%20%2012158e349680803cbf01fb7a88573790/newplot.png)

Fig. 8 Output from DataReconstructionDriftCalculator for chunk_size=500

After changing the chunk size to 500, the alert isn’t triggered, although we can see that the error values are really close to the threshold. This highlights the importance of proper chunking to counteract the sampling effect. Let’s now check if the Domain Classifier Calculator will yield similar results.

### Domain Classifier Calculator

This time, I will initialize the calculator with a chunk size equal to 500.

```python
dc_calc = nml.DomainClassifierCalculator(
    feature_column_names=['hours', 'remote', 'education', 'seniority', 'country'],
    timestamp_column_name='job_published_at',
    chunk_size=500
)
```

```python
dc_calc.fit(reference)
results = dc_calc.calculate(analysis)
```

```python
figure = results.plot()
figure.show()
```

![newplot (1).png](Drift%20Happens%20NannyML%E2%80%99s%20Guide%20to%20Model%20Monitoring%20%2012158e349680803cbf01fb7a88573790/newplot_(1).png)

The Domain Classifier Calculator with a chunk size of 500 detected multivariate data drift in the last 3 chunks. The last chunk contained only 50 records and could succumb to the sampling effect. However, the results from the preceding chunks suggest that there might be changes in the correlation between variables and that the model and production data should be observed to catch technical performance deterioration in time.

## Summary

Thorough monitoring of machine learning models is crucial to prevent performance degradation and ensure robustness, especially in volatile markets. The Zillow case is an example of  how costly it can be to neglect monitoring your model's performance. By understanding and addressing both univariate and multivariate data drift, we can ensure model accuracy and reliability.

NannyML offers methods to detect and mitigate these issues, providing safeguards against the risks of model decay. If you want to learn more about NannyML and how to leverage its functionalities to keep an eye on your business value, check out: 

[**NannyML on GitHub**](https://github.com/NannyML/nannyml)

.
