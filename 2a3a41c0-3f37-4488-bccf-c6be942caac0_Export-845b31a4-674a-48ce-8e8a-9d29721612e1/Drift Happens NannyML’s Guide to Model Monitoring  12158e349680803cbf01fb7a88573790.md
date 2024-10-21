# Drift Happens: NannyML’s Guide to Model Monitoring and Covariate Drift

## **Introduction**

So you just did the job. You gathered some data, cleaned and wrangled it to get the most out of it. After thorough experimentation you’ve built a model; you tuned the heck out of your hyper-parameters and achieved satisfying performance. It’s time to put the model into production and turn its predictions into business value. But then - sooner or later, the results [deteriorate¹](https://www.nature.com/articles/s41598-022-15245-z).

The decisions made based on your model’s predictions don’t bring any improvement or even cause losses. You need answers: what went wrong?

### **The Zillow Example**

A similar thing happened at Zillow in 2021. The company providing real estate information to buyers had to close its house-flipping department and lay off 25% of their workforce due to the failure of the Zestimate algorithm. They reported a loss of more than [$600M**².**](https://www.deeplearning.ai/the-batch/price-prediction-turns-perilous/)

The Zestimate algorithm was proprietary software that underwent a serious revamping, including employing Deep Neural Networks and NLP. However, the details of why it failed have never been disclosed. It has been speculated that the problems arose from manifold factors, including data latency and lack of robustness to handle the volatility in the market caused by the pandemic.

### Monitoring Workflow

It probably could have been prevented. By thorough monitoring of the model’s performance, you can mitigate model decay. You are able to counteract the results of data latency. You can also ensure that your model is robust enough to withstand volatility on the market you’re aiming to model. With a good monitoring system at your hand you should be able to track your models performance and whenever it deteriorates be able to perform Root Cause Analysis, which is crucial to resolve the issues that might have arisen.

![Fig.1 Monitoring Workflow that could spare you a lot of sleepless nights. Image by the author.](Drift%20Happens%20NannyML%E2%80%99s%20Guide%20to%20Model%20Monitoring%20%2012158e349680803cbf01fb7a88573790/ML_Monitoring_Workflow.png)

Fig.1 Monitoring Workflow that could spare you a lot of sleepless nights. Image by the author.

### **Reasons for Model Failure**

A Machine Learning model can fail in different ways and for different reasons. It can stop producing output due to a bug in the infrastructure or it can produce inaccurate output. The reason can be hidden in the poor scoping of the project, in that case, the data we have is an inaccurate proxy of the reality we’re attempting to model. But there’s another thing we can blame for the failure - when the reality and so our data change, and our model isn’t able to make predictions that reflect the new distributions of our variables and relationships between the variables. In this blog, we will focus on data drift, also known as covariate shift.

### **Univariate Data Drift**

Univariate data drift means that there have been changes in the probability distributions of a certain variable - P(X). This happens pretty often, as the data we’re working with is a proxy for the ever-changing reality. Those changes can be due to seasonality, new trends, or expanding markets. It’s important to monitor those changes,

Detecting univariate data drift doesn’t necessarily mean trouble. If our production data contains more data points in the areas where our model is more certain, the drift won’t have a negative impact on the model’s performance. The situation changes if our production data shifts to the regions that were under-represented at the training time or moves to less certain regions close to the decision boundary. Thus triggering alerts every time a univariate data drift occurs can cause alert fatigue—a situation where the team starts to ignore the warnings, which can be the true danger of univariate data drift.

### **Multivariate Data Drift**

But data drift can also occur when no univariate data drift happens. That’s a very peculiar situation when the distribution of a single variable doesn’t change, but the correlation between the variables does. Therefore, detecting multivariate data drift is a bit trickier than comparing the probability distributions of a single variable.

![Fig. 2 Illustration of Multivariate Data Drift from [“**A Comprehensive Evaluation of Data Drift Metrics for Air Transportation Applications”](https://www.researchgate.net/publication/382114073_A_Comprehensive_Evaluation_of_Data_Drift_Metrics_for_Air_Transportation_Applications) by Pablo Gasco and [Ramon Dalmau-Codina](https://www.researchgate.net/profile/Ramon-Dalmau-Codina?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6Il9kaXJlY3QiLCJwYWdlIjoicHVibGljYXRpb24iLCJwcmV2aW91c1BhZ2UiOiJfZGlyZWN0In19)**](Drift%20Happens%20NannyML%E2%80%99s%20Guide%20to%20Model%20Monitoring%20%2012158e349680803cbf01fb7a88573790/image.png)

Fig. 2 Illustration of Multivariate Data Drift from [“**A Comprehensive Evaluation of Data Drift Metrics for Air Transportation Applications”](https://www.researchgate.net/publication/382114073_A_Comprehensive_Evaluation_of_Data_Drift_Metrics_for_Air_Transportation_Applications) by Pablo Gasco and [Ramon Dalmau-Codina](https://www.researchgate.net/profile/Ramon-Dalmau-Codina?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6Il9kaXJlY3QiLCJwYWdlIjoicHVibGljYXRpb24iLCJwcmV2aW91c1BhZ2UiOiJfZGlyZWN0In19)**

## NannyML

NannyML is a powerful tool that helps to discover when your model is experiencing problems. It helps to detect the root cause of those issues. It stays aware of the changes in your data and can discern between changes that will impact your model’s technical performance and the resulting business impact. 

Let’s have a look at how it works.

I used a dataset I initially obtained from Sourcestack while participating in a hackathon last year. 

I tried to answer the following question: “Did  ChatGPTreplace interns and juniors in engineering jobs?” This year, I obtained another sample of that data, which makes it a perfect dataset to compare if the model trained on data from last year experienced any trouble from univariate or multivariate data drift due to data latency.

The dataset contains information about vacancies, specifically engineering jobs. It consists of variables describing the hourly type of the job, if the job is remote, what education is required, seniority expected from the candidate, and the country the job offer is available in. The target variable is compensation estimation in dollars.

To run univariate and multivariate drift detection algorithms, we need a reference dataset and the analysis dataset. The reference dataset is our baseline performance we will compare the analysis dataset to. We should use our test dataset as the reference and not the training dataset. The analysis dataset consists of our production data, the data we want to monitor.

I load both datasets.

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

Now we can start with the Univariate Drift Detection. I initialize the detector, set the chunk size, specify categorical variables, and the timestamp column.  I also pass statistical methods that the calculator can utilize to measure the distance between the data chunks stemming from my reference and analysis datasets. I use a versatile Jensen-Shannon metric that can assess both categorical and continuous distributions and a chi2 statistical test. Although we’re only interested in univariate data drift I thought it would be interesting to inspect the distribution of our target variable as this is the only continuous variable in this dataset. For this purpose I employ the Kolmogorov-Smirnov Test and the aforementioned Jensen-Shannon metric. After fitting the UnivariateDriftCalculator on the reference set, I can now feed it to the calculate method and visualize the results.

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

![Fig. 3 Output of UnivariateDriftCalculator for the categorical variables.](Drift%20Happens%20NannyML%E2%80%99s%20Guide%20to%20Model%20Monitoring%20%2012158e349680803cbf01fb7a88573790/newplot_(2).png)

Fig. 3 Output of UnivariateDriftCalculator for the categorical variables.

From the plots we can read, that only the last chunks of the dataset caused drift alert. This is due sampling effect. The last chunk consist of only 1 data point. To better understand the distributions of our categorical covariates let’s have a look at them.

```python
figure = results.filter(
			column_names=results.categorical_column_names, 
			methods=['jensen_shannon']
			).plot(kind='distribution')
figure.show()
```

![Fig. 3 Output of UnivariateDriftCalculator for the categorical variables.](Drift%20Happens%20NannyML%E2%80%99s%20Guide%20to%20Model%20Monitoring%20%2012158e349680803cbf01fb7a88573790/newplot_(3).png)

Fig. 3 Output of UnivariateDriftCalculator for the categorical variables.

Again we see the impact of the sampling effect on the drift alert. We can see that the distributions change slightly but not enough to trigger drift warning. Let’s now have a look at  the side by side comparison of distributions of our only continuous variable, the predicted and true target.

```python
figure = results.filter(
	column_names=results.continuous_column_names, 
	methods=['kolmogorov_smirnov']
	).plot(kind='distribution')
figure.show()
```

![Fig. 5 Output of UnivariateDriftCalculator for the target variable](Drift%20Happens%20NannyML%E2%80%99s%20Guide%20to%20Model%20Monitoring%20%2012158e349680803cbf01fb7a88573790/newplot_(3)%201.png)

Fig. 5 Output of UnivariateDriftCalculator for the target variable

Again we can notice changes slight changes in distributions but they hold for both the reference and analysis sets.

```python
figure = results.filter(
	column_names=results.continuous_column_names, 
	methods=['kolmogorov_smirnov']
	).plot(kind='drift')
figure.show()
```

![Fig. 6 Output of UnivariateDriftCalculator for the target variable](Drift%20Happens%20NannyML%E2%80%99s%20Guide%20to%20Model%20Monitoring%20%2012158e349680803cbf01fb7a88573790/newplot_(2)%201.png)

Fig. 6 Output of UnivariateDriftCalculator for the target variable

As we can see there has been no univariate drift neither in the input variables distributions nor in the output.

## Multivariate Drift Detection

However, no univariate drift doesn’t mean that our model won’t suffer from multivariate data drift, mentioned earlier in this blog post. Unfortunately detecting this type of drift isn’t as straightforward as applying some basic statistical tests. NannyML offers two methods to capture occurrences of multivariate data drift.

The **Data Reconstruction Drift Calculator** method uses a compression algorithm that reduces the dimensionality of data, transforming it into a new set of uncorrelated variables, called principal components, which retain the most important information while discarding noise and redundancy. The trick is to reverse that compression and compare the original and decompressed dataset and compute the decompression error—how much information has been lost. If we do that to our analysis dataset, we can then compare the error to our baseline dataset. If the error exceeds a given threshold, it means there must have been changes in the correlation between the variables of our dataset

Another way of detecting multivariate data drift is the **Domain Classifier Calculator**:

This method utilizes the LGBM Classifier to discern between data samples coming from our reference and analysis datasets. The AUROC metrics give insight into how easy it is for the model to differentiate between those two datasets. If it’s easy (high AUROC value), it means the multivariate data shift occurred in our production dataset. If the model has trouble telling the difference, it means there haven’t been any significant changes to our datasets.

Does the sample from 2024 show signs of multivariate data drift?

### Data Reconstruction Drift Calculator

Let’s initialize the Data Reconstruction Drift Calculator, passing the column names we want to monitor, timestamp data and chunk period or in this case chunk size. Let’s fit it on the reference dataset and calculate the reconstruction error for the analysis dataset.

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

![Fig. 7 Output from DataReconstructionDriftCalculator for chunk_size=200](Drift%20Happens%20NannyML%E2%80%99s%20Guide%20to%20Model%20Monitoring%20%2012158e349680803cbf01fb7a88573790/newplot_(6).png)

Fig. 7 Output from DataReconstructionDriftCalculator for chunk_size=200

The reconstruction error exceeded the threshold in two data chunks. The difference to the reconstruction error from our reference dataset isn’t huge. Let’s check the results for chunk_size=500 to exclude the sampling effect as the culprit. 

```python
drd_calc = nml.DataReconstructionDriftCalculator(
    column_names=reference.columns.to_list(),
    timestamp_column_name='job_published_at',
    chunk_size=500
)
```

![Fig. 8 Output from DataReconstructionDriftCalculator for chunk_size=500](Drift%20Happens%20NannyML%E2%80%99s%20Guide%20to%20Model%20Monitoring%20%2012158e349680803cbf01fb7a88573790/newplot.png)

Fig. 8 Output from DataReconstructionDriftCalculator for chunk_size=500

After changing the chunk_size to 500 the alert isn’t triggered, although we can see that the error values are really close to the threshold. This stresses the importance of the right chunking to counteract the sampling effect. Let’s now check if the Domain Classifier Calculator will yield similar results.

### Domain Classifier Calculator

This time I will initialize the calculator with a chunk_size equal to 500.

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

The Domain Classifier Calculator with chunk_size 500 was able to detect multivariate data drift in the last 3 chunks. The last chunk contained only 50 records and could succumb to sampling effect. But the results from the preceding chunks can suggest that there might be changes in correlation between variables and that the model and production data should be observed to catch technical performance deterioration on time.

## Conclusion

In conclusion, thorough monitoring of machine learning models is crucial to prevent performance degradation and ensure robustness, especially in volatile markets. The Zillow case is an example of how costly it can be to neglect monitoring of your model performance. By understanding and addressing both univariate and multivariate data drift, we can maintain model accuracy and reliability. Tools like NannyML offer methods to detect and mitigate these issues and provide safeguard against the risks of model decay and ensuring sustained business value.