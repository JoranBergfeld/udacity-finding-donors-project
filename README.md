# Udacity - Finding Donors Project

This repository holds the code for the finding donors project, provided by Udacity. Starting this project I had a few things in mind when implementing this project:
- Considering I prefer to have my code in python files, I wanted to keep most of the logic within python scripts and load these inside the notebook.
- Add a bit more quality controls than my previous project. Within this context I meant code quality controls for python, defensive posture for potential data issues as this was eluded to in the assignment, and the option to run multiple experiments.
- Much like the previous project, I wanted to run three different experiments to compare results. I decided to split these three experiments up in data ingestion, model training, model evaluation and improvements. More on that below.

# Code quality

I wanted to ensure I keep improving my Python coding skills, as they are admittedly a bit rusty. As a result, I am enforcing some quality gates. 

# Experiments

## Reasoning

Much like the previous project, I wanted to spend some more time on this project and try multiple approaches to this project. Given that this project is a bit more involved, and could reflect a real life scenario, I took it pretty far, perhaps too far. 

## Data pipeline options

### Bare minimum

What I could do is the bare minimum, and pass the rubric. This is not considering Kaggle, and basing the logic on the [supplied data in the csv](starter/census.csv). As hinted in the project, this would not work for Kaggle, and not be sufficient for real world scenarios. This option, as a result, would not be used.

### Defensive data loading

The more logical step would be to handle NaN values, and potentially introduce "default" values. I did not decide to go with default values, as this would introduce bias from my side. 

### Feature binning: Introducing new features derived on continuous data points

This is where I was curious the most: Would it matter if I decorated the data in a way that would be more meaningful for the model training? Applying labels to continuous values like hours per week worked could introduce bias from my side, or could help the model. I wanted to train the same model with and without my own defined buckets, and see what the end result would be. 

An example would be the hours per week. Any hours above 40 is working overtime, which logistic regression would not be able to understand. Adding the feature of "part-time", "full-time" or "overtime" would help these models understand the implication of the hours beyond a numerical number.

I will to introduce a simply boolean to enable adding these features to ensure we could compare the model performance with and without these labels. This way we can run the data pipeline easily for each model, eventually running twice: Once with added features, another time without.

### Scaler choice

My understanding of the maths behind AI is still pretty flimsy. This made my choice of scaler a bit harder, so I spend some time trying to understand the impliciations of each. The notebook standard scaler usage is the `MinMaxScaler`, but I would want to challenge that choice. As seen in the capital gains data, there's some heavy outliers. The `MinMaxScaler` would push more common values together due to the outliers. I ended up not overriding the default in the notebook, as it's defined in a cell without the `Implementation` prefix. I'd like to try this out in another project.

## Model selection
