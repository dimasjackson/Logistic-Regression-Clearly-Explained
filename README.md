# Logistic Regression

Logistic Regression is a statistical model that is used in classification and predictive analysis. This model is also know as Logit and it is characterized by a single binary dependent variable, i. e. a variable that only can take two values, often labeled as 0 and 1. Binary values are widely udsed in statistics to model the probablity of a certain event occuring, such as the probability of a pacient being health, a tumor being malignant or not, if an email is spam or not and if a team win or loose. Therefore, Logistic Regression has a large variety of applications. In this tutorial we will present a clear explanation about logistic regression models, build from scratch a model and compare it with the scikit learn implementaion.

The logistic function is defined by:
$$ p(x) = \dfrac{1}{1+e^{-(x-\mu)/s}} $$
where $\mu$ is the midpoint of the curve ($p(\mu)=1/2$) and $s$ a scale parameter that determines the spread of the probability distribution. This function is also called **sigmoid function**, because of its 's' shape.

The logistic fuction is used to generate predictions as we will see below. Furthermore, it is interesting to write this function as 
$$ p(x) = \dfrac{1}{1+e^{-(\beta_0 + \beta_1 x)}} \,,$$ 
where $\beta_0 = -\mu/s$ is the intercept of the line $y = \beta_0 + \beta_1 x$ and $\beta_1 = 1/s$ is its slope. The parameter $\beta_0$ is also called "bias" and $\beta_1$ "weights". The particular values of these quantities that maximizes the likelyhood function are what we need to find to make predictions.

![image](https://user-images.githubusercontent.com/114688989/232498711-17e90cde-7a8d-4ac5-9d5e-3134d6077c34.png)

# Types of Logistic Regression

1. **Binary Logistic Regression:** In this case the categorical variable (target) has only two possible outcomes. Examples: Spam or Not, diagnostic positive or negative, some equipment failed or not, students pass or fail in an exam.

2. **Multinomial Logistic Regression:** The target contains three or more categories without ordering. Examples: Predicting if a individual is vegetarian, meat-eater or vegan, customer segmentation.

3. **Ordinal Logistic Regression:** The target containd three or more categories with ordering. Example: Movie rating from 1 to 5.

In this tutorial we will implement from scratch a binary logistic regression model.

# Example

The binary logistic regression can be used with one **explanatory variable** and two **categories** to answer a question. As an example, consider the case of a group of students which are preparing for a exam. The question is: how does the number of study hours affect the probability of the student passing the exam? In this case the explanatory variable is the number of hours studying and the two categories are being approved or not. Let's consider 10 students that spends between 0 to 10 hours studing and use 1 to identify "pass" and 0 to "fail". Then, we have the following "sintetic" data:
![image](https://user-images.githubusercontent.com/114688989/232499069-502a6418-dc20-40ba-87f5-66f81ae60d91.png)

To predict if a student will pass or not we need to consider the loss function, which is a function that represents the "price to pay" for inaccuracy of predictions. Usually, the logistic loss function is used as the measure of goodness of a fit in logistic regression. Considering that $p_k$ is the probability that the $k-$th student will pass the exam and $1-p_k$ the fail probability, the log loss for the $k-$th point is:
$$ \begin{cases}
-\ln p_{k} & \text{if }y_{k}=1\\
-\ln(1-p_{k}) & \text{if }y_{k}=0
\end{cases} $$

Note that log loss is always greater than or equal to 0, equals 0 only in case of a perfect prediction, and approaches to infinity when predictions get worse. The two cases can be combined into a function called cross entropy:
$$ -y_k \ln p_k -(1-y_k) \ln (1-p_k) $$

The sum of the this term for all elements is the negative likelihood function:
$$ -l = -\sum_k^N [y_k \ln p_k +(1-y_k) \ln (1-p_k)]$$

To estimate the probability of an outcome we need to find the values of $\beta_0$ and $\beta_1$ that minimizes the negative likelihood (or maximizes the positive likelihood, if you prefer). This is accomplished taking the derivative of the likelihood with respect to these parameters (or applying the gradient operator in high dimension problems):
$$ 0 = \frac{\partial l}{\partial \beta_0} = \sum_{k=1}^N(y_k-p_k) $$
$$ 0 = \frac{\partial l}{\partial \beta_1} = \sum_{k=1}^N(y_k-p_k)x_k $$
Then we need to solve the above two equations for  $\beta_{0}$ and $\beta_{1}$, which generally requires the use of numerical methods. For example, we may start with $\beta_0 = 0$ and $ \beta_1 = 0$ and iterate the model increasing these parameters by a fraction of the gradient. After that, the probability of passing or failing the exam can be calculated using the sigmoid function
$$ p = \dfrac{1}{1+e^{-t}}\,,$$
where $t = \beta_0 + x\beta_1$. Let's implement this calculation using python functions


# Our Implementation

 We will start with $\beta_0 = 0$ and $ \beta_1 = 0.01(1,1,\dots,1)$ and iterate the model increasing these parameters by a fraction of the gradient. This fraction amount is usually called "learning rate". (code in the notebook)
 
 ```
 # Defining the initial parameters (zeros)
def weightInitialization(n_features):
    beta_1 = np.full((n_features),0) # Note that beta_1 must have the same dimension of x
    beta_0 = 0
    return beta_1,beta_0

# Defining the sigmoid function
def sigmoid_vec(t):
    p = 1/(1+np.exp(-t)) 
    return p

# Calculates the likelihood and the derivatives
def log_model(beta_1, beta_0, x, y):
       
    #Prediction
    t = np.dot(beta_1, x.T) + beta_0 # The .T means the transpose of x and np.dot the dot product
    prob = sigmoid_vec(t)
    y = np.array(y)
    likelihood = np.sum( -y*np.log(prob) - (1 - y)*np.log(1 - prob) )/x.shape[1] # The factor 1/x.shape[1] is for scaling by the n# of features
    
    #Gradient calculation
    dbeta_1 = np.dot(x.T, (prob-y))/x.shape[1]
    dbeta_0 = np.sum(prob-y)/x.shape[1]
    
    gradients = {"dbeta_1": dbeta_1, "dbeta_0": dbeta_0}
    
    return gradients, likelihood, prob 

# Iterate the model to make it learn
def model_predict(n_features, x, y, learning_rate, n_iterations):
    beta_1, beta_0 = weightInitialization(n_features)
    costs = []
    
    for i in range(n_iterations):
        
        gradients, likelihood, prob = log_model(beta_1, beta_0, x, y)
        
        dbeta_1 = gradients["dbeta_1"]
        dbeta_0 = gradients["dbeta_0"]
        #weight update
        beta_1 = beta_1 - (learning_rate * dbeta_1)
        beta_0 = beta_0 - (learning_rate * dbeta_0)
        costs.append(likelihood)
        
            
    #final parameters
    y_prob = prob
    parameters = {"beta_1": beta_1, "beta_0": beta_0}
    gradient = {"dbeta_1": dbeta_1, "dbeta_0": dbeta_0}
        
    return parameters, gradient, costs, y_prob

# Create an array with the final predictions
# Given a bias beta_0, a weight beta_1 and a input x return the predicted probability of pass or fail
def predict(beta_1, beta_0, x): 
    t = np.dot(beta_1, x.T) + beta_0
    z = sigmoid_vec(t)
    y_pred = np.zeros(x.shape[0])
    for i in range(y_pred.shape[0]):
        if z[i] > 0.5:
            y_pred[i] = 1
    return y_pred
 ```
 Finally, we define a function to encompass all the functions above:
 
 ```
 # Defining our logistic regression model
def my_Logistic_Regression(x_train, y_train, x_test, learning_rate, n_iterations):
    n_features = x_train.shape[1] # Dimension of the input
    beta_1, beta_0 = weightInitialization(n_features) # Defined above
    parameters, gradients, costs, y_prob = model_predict(n_features, x_train, y_train, learning_rate, n_iterations)# Defined above
    y_pred = predict(parameters["beta_1"], parameters["beta_0"], x_test) # Defined above
    return y_pred, y_prob
 ```
 
 # Scikit Learn Implementation

The Scikit Learn `linear_model.LogisticRegression()` class implements a regularized logistic regression. his implementation can fit binary, One-vs-Rest, or multinomial logistic regression.

Let's us import a data set to make predictions. As an example we will use a data set from National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of  these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

![image](https://user-images.githubusercontent.com/114688989/232500293-555aa487-bc83-4f07-adc9-c4708d577e83.png)

# Measuring performance of Scikit Learn predictions

### Confusion Matrix

The confusion matrix can tells us the number of true/false positive/negative predictions. By the time, we can calculate the precision rate, the missclassification rate, accuracy, prevalacence, etc. Let's see with more details the quantities that we can calculate using confusion matrix:

1) Accuracy = (TP+TN/total) Is the overall evaluation of the classifier.

2) Sensitivity = TP/(actual number of 1's) Indicates how many times 1 is predicted correctly.

3) Fall-out = FP/(actual number of 0's) Indicates how many times 1 is predicted when actual answer is 0.

4) Specificity = TN/(actual number of 0's) Indicates how many times 0 is predicted correctly.

5) Misclassification Rate = (FP+FN)/(total) Tells about how often our model is wrong.

6) Precision = TP/(TP + FP) Gives the rate of correct predictions.

7) Prevalence = (actual number of 1's /total) Indicates how often condition 1 really occurs.

![image](https://user-images.githubusercontent.com/114688989/232500441-1a53a41e-8478-4156-9f00-8f28d0b5dbe5.png)

This means that the Scikit Learn logistic model with the parameters choice above hit the predictions nearly $77\%$ of the time, for the Diabets dataset. Its not so good having $23\%$ of a population being incorrectly diagnosed, but its important to note that this is a simple example, to improve this score one can vary some parameters of the Logistic Regression class and fetch for correlation between the dependent variables. 

The Cumulative Distribution Function-CDF helps us understand how good is this model:
![image](https://user-images.githubusercontent.com/114688989/232500690-317a0f14-31d5-42b2-8d11-8e3488ce0902.png)

Another important validation method is the Area Under Curve - AUC, wich can be calculated using the Receiver Operating Characteristic-ROC curve ploted below. The ROC curve is the curve generated by the plotting of True Positive Rate by the False Positive rate.

![image](https://user-images.githubusercontent.com/114688989/232500807-327d13c6-0610-4eb5-bb4a-5e1f9822d461.png)
![image](https://user-images.githubusercontent.com/114688989/232505872-37b8b205-b8dd-4cc0-a255-d52d46896874.png)

The Kolmogorv-Smirnov coeficient is the maximum distance between the two curves, such as the AUC, higher values are preferable. Although, the model may be improved using thecniques like crossvalidation and standardizaton, this simple model shows a reasonable value for these metrics with a very low p-value (with means a high confidence). 

# Comparing our Logistic Model with the Scikit Learn Class 

As expected, the likelihood tends to a minima when the optimal value of iterations is reached.

![image](https://user-images.githubusercontent.com/114688989/232501004-750bcb71-6041-4175-bf29-6b3cf0a42160.png)

Let's plot the confusion matrix of our model and calculate the precision, recal and f1-score:

![image](https://user-images.githubusercontent.com/114688989/232501301-60b38634-9c80-41ef-a083-f36e8f09f7d7.png)

![image](https://user-images.githubusercontent.com/114688989/232501387-4ea64553-8a52-4a1b-9fab-4aad0adc38dc.png)

Also, let's plot the ROC curve and calculate the AUC of our logistic regression implementation:

![image](https://user-images.githubusercontent.com/114688989/232501534-c6986519-4225-4bc1-a9c8-c04765026425.png)

![image](https://user-images.githubusercontent.com/114688989/232505675-af23f145-8b72-4783-9f87-4803834c2829.png)

This model present a very low AUC and KS coefficient compared to Scikit Leanr implementaion. This was expected, since this is a model for didact purpose only.

# Conclusion

We build from scratch a logistic regression model and compared it to the scikit learn implementation default model. Our focus was in undestand the mechanisms behind the blackbox of logistic regression models and clear explain it. As expected for a didatic simple model, our implementation performed worse than the scikit learn model, but the objective of the tutorial was succesfully achieved. 


# References

* [Logistic Regression - Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression#Definition)
* [Logistic Regression — Detailed Overview](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc)
* [Logistic Regression Implementation](https://www.kaggle.com/code/kanncaa1/logistic-regression-implementation)
* [Performance Measurement in Logistic Regresion](https://meettank29067.medium.com/performance-measurement-in-logistic-regression-8c9109b25278)
* [Understanding AUC](https://medium.com/@data.science.enthusiast/auc-roc-curve-ae9180eaf4f7)




