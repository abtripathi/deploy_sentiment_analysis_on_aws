# Deploy Sentiment Analysis on AWS Sagemaker

In this project, I built and trained a RNN to determine the sentiment of a movie review using [IMDb dataset](http://ai.stanford.edu/~amaas/data/sentiment/).The model was built and  trained using Pytorch 1.5.0 with Sagemaker Python SDK and deployed as an endpoint on AWS Sagemaker.

The model is accessed through a simple web app using below architecture

<img src="Web App Diagram.svg">

The architecture above gives an overview of how the various services are working together. On the far right is the model which is trained and deployed using SageMaker. On the far left is a web app that collects a user's movie review, sends it off and expects a positive or negative sentiment in return.

In the middle is a Lambda function,a straightforward Python 3.7 function that can be executed whenever a specified event occurs.In this architecture,lambda function executes the deployed model's endpoint by sending and receiving data to it.Any permission required by Lambda function to execute the deployed model's endpoint is given

Lastly, I created a public endpoint URL using API Gateway to execute the Lambda function.This endpoint listens for data to be sent to it from a web app.Once it gets some data, it delegates to the Lambda function and then returns the unmodified response given by Lambda function.

