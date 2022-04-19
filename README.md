This project created in order to answer two main question:
    1. Find the different types of activities in walking activity
    2. Identification and authentication of people using motion patterns

Here we have accelerometers data belonging to 22 participants who walk over a predefined path.
data have timestamps which are float nad three axis x,y,z which show different measures.
Data is pretty much noisy, and also not enough to help us to answer our questions.
For answering the first question I had to create two new features and by using them I tried the clustering method to
find different types of activities.
For the second question, it was impossible to use data the way it is even with new features so
I needed to extract new futures with respect to different time periods.
And by using CNN I tried to create a classifier to identify each person by its walking pattern data.
A huge part of my solution is thanks to two paper which their links are below:

https://www.researchgate.net/publication/227192676_Personalization_and_user_verification_in_wearable_systems_using_biometric_walking_patterns
https://github.com/theumairahmed/User-Identification-and-Classification-From-Walking-Activity

In order to run the main.py you need install some specific libraries such as :
fast_ml,
bayesian-optimization
tensorflow
sklearn
statsmodels

