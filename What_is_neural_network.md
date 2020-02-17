### What is neural network?

It is a powerful learning algorithm inspired by how the brain works.



##### Example 1- Housing Price Prediction

We know the prices can never be negative so we are creating a function called Rectified Linear Unit (ReLU) which starts at zero.

<img src="https://user-images.githubusercontent.com/56706812/74636702-e1562c80-51ab-11ea-92db-df4466805e0f.png" alt="image-20200216105211181" style="zoom:67%;" />

- The input is the size of the house (x) 

- The output is the price (y) 

- The “neuron” implements the function ReLU (blue line) 

  

##### Example 2- Multiple neural network

The role of the neural network is to predicted the price and it will automatically generate the hidden units. We only need to give the inputs x and the output y.  



<img src="https://user-images.githubusercontent.com/56706812/74636744-f92db080-51ab-11ea-90e2-2b3d1097c949.png" alt="image-20200216110608398" style="zoom:67%;" />



### Supervised learning for Neural Network

Supervised learning problems are categorized into **"regression"** and **"classification"** problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.

There are different types of neural network, for example **Convolution Neural Network (CNN)** used often for image application and **Recurrent Neural Network (RNN)** used for one-dimensional sequence data such as translating English to Chinese or a temporal component such as text transcript. As for the autonomous driving, it is a hybrid neural network architecture.

### Structured vs unstructured data

Structured data refers to things that has a defined meaning such as price, age whereas unstructured data refers to thing like pixel, raw audio, text.

<img src="https://user-images.githubusercontent.com/56706812/74636771-08146300-51ac-11ea-9144-ab4112cc6dee.png" alt="image-20200216111822134" style="zoom:67%;" />



### Why is deep learning taking off?

<img src="https://user-images.githubusercontent.com/56706812/74636799-15315200-51ac-11ea-94a1-e450484f9dbd.png" alt="image-20200216115650410" style="zoom:67%;" />

Two things have to be considered to get to the high level of performance: 

1. Being able to train a big enough neural network 
2.  Huge amount of labeled data 



The process of training a neural network is iterative. 

<img src="https://user-images.githubusercontent.com/56706812/74636831-237f6e00-51ac-11ea-8ede-59eefa173648.png" alt="image-20200216120638938" style="zoom:67%;" />

It could take a good amount of time to train a neural network, which affects your productivity. Faster computation helps to iterate and improve new algorithm. 



### Binary classification

##### Example: Cat vs Non-Cat

The goal is to train a classifier that the input is an image represented by a feature vector, 𝑥, and predicts whether the corresponding label 𝑦 is 1 or 0. In this case, whether this is a cat image (1) or a non-cat image (0).

<img src="https://user-images.githubusercontent.com/56706812/74636867-35f9a780-51ac-11ea-9190-d59a98222830.png" alt="image-20200216121427217" style="zoom:67%;" />

An image is store in the computer in three separate matrices corresponding to the Red, Green, and Blue color channels of the image. The three matrices have the same size as the image, for example, the resolution of the cat image is 64 pixels X 64 pixels, the three matrices (RGB) are 64 X 64 each.

To create a feature vector, 𝑥, the pixel intensity values will be “unroll” or “reshape” for each color. The dimension of the input feature vector 𝑥 is 𝑛𝑥 = 64 * 64 * 3 = 12 288. 

 <img src="https://user-images.githubusercontent.com/56706812/74636894-43169680-51ac-11ea-8f9a-332fd07448eb.png" alt="image-20200216135212110" style="zoom:67%;" />

### logistic regression

Logistic regression is a learning algorithm used in a supervised learning problem when the output 𝑦 are all either zero or one. The goal of logistic regression is to minimize the error between its predictions and training data. 



##### Example: Cat vs No - cat 

$$
𝐺𝑖𝑣𝑒𝑛 𝑥 , 𝑦 ̂ = 𝑃(𝑦 = 1|𝑥), where 0 ≤ 𝑦 ̂ ≤ 1
$$

The parameters used in Logistic regression are: 

• The input features vector:   𝑥 ∈ ℝ^𝑛^𝑥, where 𝑛𝑥 is the number of features
• The training label:  𝑦 ∈ 0,1 
• The weights: 𝑤 ∈ ℝ^𝑛^𝑥, where 𝑛~𝑥~ is the number of features 
• The threshold: 𝑏 ∈ ℝ • The output: 𝑦 ̂ = 𝜎(𝑤^𝑇^𝑥 + 𝑏) 
• Sigmoid function: s = 𝜎(𝑤^𝑇^𝑥 + 𝑏) = 𝜎(𝑧)= 1 / 1 + 𝑒^−𝑧^

<img src="https://user-images.githubusercontent.com/56706812/74636917-5164b280-51ac-11ea-8f2d-07b044ff1d97.png" alt="image-20200216140249312" style="zoom:67%;" />![image-20200216142716063](https://user-images.githubusercontent.com/56706812/74637443-5a09b880-51ad-11ea-9cb1-2fa09a097b38.png)

(𝑤𝑇𝑥 + 𝑏)  is a linear function (𝑎𝑥 + 𝑏), but since we are looking for a probability constraint between [0,1], the sigmoid function is used. The function is bounded between [0,1] as shown in the graph above.

Some observations from the graph: 

• If 𝑧 is a large positive number, then 𝜎(𝑧) = 1 
• If 𝑧 is small or large negative number, then 𝜎(𝑧) = 0 
• If 𝑧 = 0, then 𝜎(𝑧) = 0.5 



### Standard notations for Deep Learning

<img src="https://user-images.githubusercontent.com/56706812/74637479-67bf3e00-51ad-11ea-9e4a-ad863b159b76.png" alt="image-20200216142716063" style="zoom:67%;" />

<img src="https://user-images.githubusercontent.com/56706812/74637479-67bf3e00-51ad-11ea-9e4a-ad863b159b76.png" alt="image-20200216142716063" style="zoom:67%;" />



### Logistic Regression : Cost Function

##### Loss (error) function: 

The loss function measures the discrepancy between the prediction (𝑦 ̂(𝑖)) and the desired output (𝑦(𝑖)). In other words, the loss function computes the error for a single training example. 


$$\ 𝐿(𝑦 ̂(𝑖),𝑦(𝑖)) = 1 /2 (𝑦 ̂(𝑖) − 𝑦(𝑖))2 $$


$$\ 𝐿(𝑦 ̂(𝑖),𝑦(𝑖)) = −( 𝑦(𝑖) log(𝑦 ̂(𝑖)) + (1 − 𝑦(𝑖))log (1 − 𝑦 ̂(𝑖))$$

​	• If $$ 𝑦(𝑖) = 1: 𝐿(𝑦 ̂(𝑖),𝑦(𝑖)) = −log(𝑦 ̂(𝑖))$$ where $\ log(𝑦 ̂(𝑖))$ and $\ 𝑦 ̂(𝑖)$​  should be close to 1  

​	• If $$𝑦(𝑖) = 0: 𝐿(𝑦 ̂(𝑖),𝑦(𝑖)) = −log(1 − 𝑦 ̂(𝑖))$$ where $\ log(1 − 𝑦 ̂(𝑖))$ and $\ 𝑦 ̂(𝑖)$ should be close to 0 

##### Cost function

 The cost function is the average of the loss function of the entire training set. We are going to find the parameters 𝑤 𝑎𝑛𝑑 𝑏 that minimize the overall cost function. 
$$ 𝐽(𝑤,𝑏) = (1/ 𝑚)∑𝐿(𝑦 ̂(𝑖),𝑦(𝑖)) = − 1/𝑚∑[( 𝑦(𝑖) log(𝑦 ̂(𝑖)) + (1 − 𝑦(𝑖))log (1 − 𝑦 ̂(𝑖))] $$