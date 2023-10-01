---
title: What is a Random Forest
---

# {% $markdoc.frontmatter.title %}


Random Forest is one of my favorite algorithms because you can pretty much drop anything into it and get a good result. It’s like the Vitamix smoothie of my ML dreams. It’s considered a Blackbox model which is a technical term for, "we don’t know why it does what it does, but we sure do like the result."  

It’s a rather simple model to implement but it gets results.  

## What is a Random Forest, anyway?  

Random Forest is one of my favorite algorithms. You can pretty much drop anything into it and get a decent result. It’s like the Vitamix smoothie of my ML dreams. Random Forest models are a Blackbox model. That is  a technical term for, "we don’t know why it does what it does, but we sure do like the results."  

It’s also a rather simple model to get started.  


### 1. Random Forests are a Non-Parametric Statistical model 

But what does that mean?  Well, this is the best definition that I’ve found so far:  
>"Nonparametric statistics is a method that makes statistical inferences without regard to any underlying distribution. The method fits a normal distribution under no assumptions."

In more layman's terms: 
>"The data you have doesn’t follow a normal distribution or follows a strange shape. There can be outliers, shifts or other weird things."  

Simpler still, when plotted on a histogram, the data doesn't look like the image below. 


![image](https://www.freecodecamp.org/news/content/images/2020/08/normal_dist_68_rule.jpg )

{% callout type="note" %}
A Normal distribution is also called a bell curve. The shape comes from the assumption that 99.7% of the data is within on standard deviation form the mean. More on that in a different post. 
{% /callout %}

### 2. Random Forests are a type of Ensemble Model 

Random Forest are  comprised of a collection of smaller models called decision trees. It combines weak tree model then averages the result to provide an answer. Each split creates a new tree which contains a random subset of features (or columns). This is one reason why feeding it data with varying features or columns is a good idea.

There are 2 types of random forests: 
-Bagging
-Boosting such as XGBOOST, CATBOOST, LIGHTGBM package implementations

{% callout type="note" %}
Research has shown that they usually stabilize at about 200 trees. Boosted trees continue improving at 1000 trees. More detailed information on Random Forest can be found in [*Elements of Statistical Learning*](https://hastie.su.domains/ElemStatLearn/)
{% /callout %}

Since it doesn’t care about the distribution, it performs well on: 
Data imbalances and missing values. 

Methods it uses to handling missing values:

**Training Set:** 
- Regression: It fills missing values is the median value. For continuous values, it takes the average of non-missing class weighted by proximity. 

- Classification: 
It fills missing values  with  the most frequent non-missing value in column. For continuous values, it takes the most frequent non-missing value in column weighted by proximity. 

{% callout type="note" %}
Regreession and Classification are two types of supervised learning techniques.More on that in a different post. 
{% /callout %}

**Test Set:**

- If a label, or column name, exists. It fills the missing data with values derived from the training set and uses it as replacements. 

- If labels do not exist, then it replicates each case in the test set nclass times. The first replicate of a case is assumed to be class 1 and the class 1 replaces missing values. 

The 2nd replicate is assumed to be class 2 and the class 2 fills it. In each set of replicates, the one receiving the most votes determines the class of the original case. 

 More detailed research into various methods of handing missing data in random forests. Why it gets complicated, but you can see research into in the [paper.](https://arxiv.org/pdf/1701.05305.pdf) 

 

### Key Features of Random Forest Models 

One key feature of a Random Forest is Out of Bag Samples, or OOB. Your Random Forest takes a bootstrap sample of rows to construct a Decision Tree. The OOB samples are the ones that are left out of that bootstrap sample by your Decision Tree. The OOB samples are then given to the Decision Tree to predict the outcome. Essentially, it constructs its own validation set.  To quote my favorite statistical modeling book,  

> “An OOB error estimate is almost identical to that obtained by N-fold cross validation”7. That is why you don’t need to validate output of  Random Forest with cross validation.  “Once the OOB error stabilizes the training is terminated.” 
This is also used to construct the variable importance measure. 

Extra information from Stat Quest with much better [visuals.]( https://youtu.be/J4Wdy0Wc_xQ) 

### Why Use Random Forest 

If you haven’t figured it out, this is my favorite model for getting ML projects of the ground. But why would you want to use it? 

- Good first pass model if you have no preconceived notions about the data.  

- It has variance reduction for estimated prediction function which limits overfitting. It does this through bagging or bootstrap aggregation.  

- You act as a Data Engineer/Data Scientist and you don’t want to do intensive data cleaning. Startups and smaller businesses often hire Data Scientist to work in dual roles. Even some larger ones, make it easier for you.  

- Train faster than neural networks. A let’s face it neural networks may be popular but sometimes they are over kill. Businesses want results more than they want you to have fun...sorry.  

- It’s simpler to deploy. Putting a model into production is costly. Most companies don’t have large R&D budgets; simple deployments can produce consistent results.  

- Can estimate feature importance. It can  demystify the black box model. 

- Black box but a proximity plot would be useful. It shows ‘which observations are effectively close together’.

 together’.  

### Best data to use with examples 

Data: Categorical(binning), Continuous, Boolean 

Like a lot of features i.e. large number of columns 

###  When not to use it with examples 

Don’t use it when: 

- A linear model 

- High cardinality categorical variables 

- Time series forecasting 

- Natural Language Processing (NLP )

These all have caveats because you can always layer on and construct your own ensemble models but don’t just dump things into these models. 