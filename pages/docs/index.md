---
title: What is a Random Forest?
---

# {% $markdoc.frontmatter.title %}


Random Forest is one of my favorite algorithms because you can pretty much drop anything into it and get a good result. It’s like the Vitamix smoothie of my ML dreams. It’s considered a Blackbox model which is a technical term for, "we don’t know why it does what it does, but we sure do like the result."  

It’s a rather simple model to implement and it gets results.  

## What is a Random Forest, anyway?  

Random Forest is one of my favorite algorithms. You can pretty much drop anything into it and get a decent result. Random Forest models are a Blackbox Model. A Blackbox Model is a technical term for, "we don’t know why it does what it does, but we sure do like the results."  

It’s also a rather simple model to get started.  


### 1. Random Forests are a Non-Parametric Statistical model 

But what does that mean?  Well, this is the best definition that I’ve found so far:  
>"Nonparametric statistics is a method that makes statistical inferences without regard to any underlying distribution. The method fits a normal distribution under no assumptions."

In more layman's terms: 
>"The data you have doesn’t follow a normal distribution or follows a strange shape. There can be outliers, shifts or other weird things."  

Simpler still, when plotted on a histogram, the data doesn't look like the image below. 


![image](https://www.freecodecamp.org/news/content/images/2020/08/normal_dist_68_rule.jpg )

{% callout type="note" %}
A Normal distribution is also called a bell curve. The shape comes from the assumption that 99.7% of the data is within one standard deviation from the mean. 
{% /callout %}

### 2. Random Forests are a type of Ensemble Model 

As you may know, Ensemble Models are models that are comprised of smaller models. Random Forests follow this pattern are  comprised of a collection of smaller models called decision trees. Random Forests combine weak tree models then averages the result to provide an answer. Each split creates a new tree which contains a random subset of features or columns.

There are two types of methods for Random Forests: 
-Bagging
-Boosting such as XGBOOST, CATBOOST, LIGHTGBM package implementations

{% callout type="note" %}
Research has shown that they usually stabilize at about 200 trees. Boosted trees stablize around 1000 trees. More detailed information on Random Forest can be found in [*Elements of Statistical Learning*](https://hastie.su.domains/ElemStatLearn/)
{% /callout %}

Since it doesn’t care about the distribution, it performs well on: 
Data imbalances and missing values. 

How Random Forests handling missing values internally:

**Training Set:** 
- Regression: 
    - It fills missing values is the median value.
    - For continuous values, it takes the average of non-missing class weighted by proximity. 

- Classification: 
    - It fills missing values  with  the most frequent non-missing value in column. 
    - For continuous values, it takes the most frequent non-missing value in column weighted by proximity. 

{% callout type="note" %}
Regression and Classification are two types of supervised learning techniques.More on that in a different post. 
{% /callout %}

**Test Set:**

- •	If a label, or column name exists it fills the missing data with values derived from the training set and uses it as the replacement value.
-•	If labels do not exist, then it replicates each case in the test set *nclass* times. The first replica of a case is assumed to be *class 1* and the *class 1* the original case. 

The 2nd replica is assumed to be class 2 and the class 2 fills that value. In each set of replicas, the one receiving the most votes will determine the class of the original case. More detailed research into various methods of handing missing data in Random Forests is availale in this paper. [this paper.](https://arxiv.org/pdf/1701.05305.pdf) 

 

### Key Features of Random Forest Models 

One key feature of a Random Forest is Out of Bag Samples, or OOB. Your Random Forest takes a bootstrap sample of rows to construct a Decision Tree. The OOB samples are left out of that bootstrap sample by your Decision Tree. The OOB samples are then given to the Decision Tree to predict the outcome. Essentially, it constructs its own validation set.  To quote my favorite statistical modeling book,  

> “An OOB error estimate is almost identical to that obtained by N-fold cross validation”. 
That is why you don’t need to validate output of  Random Forest with cross validation. Cross Validation is another step in the machine learning process. Find out more here. 


Additional information along with visuals on Out of Bag Samples from Stat Quest. [Additional information along with visuals on Out of Bag Samples from Stat Quest.]( https://youtu.be/J4Wdy0Wc_xQ) 

### Why Use Random Forest 

Random Forest models are great for getting Machine Learning projects off the ground. Now let's look at the bennefits of using it? 

- It is a good for taking a first look at a model outcome. If you don't know anything about your data, but it is tabular. You can just run it through.

- It has variance reduction for estimated prediction function which limits overfitting. It does this through bagging or bootstrap aggregation.  

- You act as a Data Engineer/Data Scientist and you don’t want to do intensive data cleaning. Startups and smaller businesses often hire Data Scientist to work in dual roles. Even some larger companies make it easier for you.  

- Train faster than neural networks. A let’s face it neural networks may be popular but sometimes they are over kill. Businesses want results more than they want you to have fun...sorry.  

- It’s simpler to deploy. Putting a model into production is costly. Most companies don’t have large R&D budgets; simple deployments can produce consistent results.  

- Can estimate feature importance. It can  demystify the black box model. 

- A Random Forest model is considered Black box. But a proximity plot can be produced and it could be useful. It shows ‘which observations are effectively close together’.


### When should you use with the Random Forest Model: 

Categorical(binning), Continuous, and Boolean data 

    Ex. Data with a large number of features / columns

###  When you should not to use the Random Forest Model: 

- when working with a linear model

- when working with high cardinality with categorical variables 

- when working on time series forecasting 

- when working with Natural Language Processing (NLP ) problems


**References**
1. [*Elements of Statistical Learning*](https://hastie.su.domains/ElemStatLearn/)
2. [https://corporatefinanceinstitute.com/resources/data-science/nonparametric-statistics/](https://corporatefinanceinstitute.com/resources/data-science/nonparametric-statistics/)
3. [https://www.stat.berkeley.edu/%7Ebreiman/RandomForests/cc_home.htm#missing1](https://www.stat.berkeley.edu/%7Ebreiman/RandomForests/cc_home.htm#missing1)