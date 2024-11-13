# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest Classifier developed to predict whether an individual's annual income exceeds $50K based on demographic and employment data from the U.S. Census dataset. The model was trained using the scikit-learn library and is part of a scalable machine learning pipeline deployed using FastAPI.
Model type: Random Forest Classifier
Framework: scikit-learn
Input features: The model uses 14 features, including age, education, workclass, occupation, marital status, race, sex, hours-per-week, and native country.
Output: The model outputs a binary classification (<=50K or >50K).

## Intended Use
The model is intended for educational purposes to demonstrate how to build, evaluate, and deploy a machine learning pipeline. It predicts whether an individual's annual income is greater than $50K based on demographic data. This model should not be used for real-world decision-making without further validation and testing.
Primary users: Data scientists and machine learning engineers looking to learn about building scalable ML pipelines.
Primary use cases: Educational purposes and demonstrations of ML techniques.
Limitations: The model may not generalize well to populations outside of the U.S. or datasets with different distributions.



## Training Data
The training data comes from the U.S. Census dataset (census.csv). This dataset contains demographic information about individuals, including their age, education level, occupation, marital status, race, sex, capital gain/loss amounts, hours worked per week, native country, and income level (the target variable). The target variable is binary: <=50K or >50K.
The dataset contains 32,561 records.
Features used include age, workclass, education level (and number), marital status, occupation type, relationship status within household (e.g., husband/wife), race/ethnicity groupings (e.g., White/Black), sex (Male/Female), capital gain/loss amounts during the year prior to census collection period, hours worked per week on average during that year period before census collection date(s), native country origin location(s) reported by respondent(s).


## Evaluation Data
The evaluation data is a subset of the census dataset that was held out for testing purposes. It contains the same features as the training data and was processed similarly using the process_data function.
Test set size: 20% of the original dataset (approximately 6,512 records).

## Metrics
The model was evaluated using precision, recall, and F1-score on the test set. These metrics are important for understanding how well the model balances false positives and false negatives.
Overall Performance:
Metric	Value
Precision	0.7376
Recall	0.6288
F1 Score	0.6789

Performance on Categorical Slices:
Performance metrics were also computed for specific slices of the data based on categorical features such as workclass and education.
Workclass Slice:
Workclass	Precision	Recall	F1 Score	Count
Federal-gov	0.5000	0.4762	0.4878	188
Local-gov	0.8197	0.7353	0.7752	430
Private	1.0000	1.0000	1.0000	4,595
Education Slice:
Education	Precision	Recall	F1 Score	Count
Bachelors	0.7500	0.8182	0.7826	X
HS-grad	0.6792	0.5538	0.6102	X
These metrics show how well the model performs across different subgroups in the data.

## Ethical Considerations

This model was trained on historical census data that may contain inherent biases related to race, gender, and socioeconomic status. It is important to recognize that these biases could affect the fairness of predictions made by the model.
The dataset contains sensitive demographic attributes such as race and gender.
The model may reinforce existing biases present in the historical data.
Users should be cautious when applying this model in real-world scenarios where fairness and equity are important considerations.

## Caveats and Recommendations
While this Random Forest Classifier performs well on the test set with reasonable precision and recall scores:
It may not generalize well to populations outside of the U.S.
The dataset contains missing values (e.g., "?" in some categorical features), which may affect performance.
Further work could be done to improve recall rates while ensuring fairness across different demographic groups within future iterations before deploying it into production environments where decisions based solely upon its outputs would carry significant consequences affecting those impacted directly by said decisions.
