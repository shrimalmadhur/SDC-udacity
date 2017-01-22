# Explaination

## Data
I used udacity data for training my model since I didn't had joystick to generate
## Augmentation
### Original data
I used original data as it is without any changes
### Flipped Data
Original data was not enough and I wasn't able to even last for few seconds. I then flipped the original data and multiplied the steering angles by -1. After looking at the data distribution and some Udacity post I realized that there are so many data with 0 or small angles. This was creating bias towards near 0 steering angles in the model. So I removed all the data with angles ranging from -0.02 to 0.02. That helped me passing the first bridge and the left turn after that.
### Brightness data
I was not able to pass the next steep right turn. I tried with different hyper parameters but I wasn't able to tune it. So I was looking at some Udacity post and found [this](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.vzzysbnmc) link. I used the brightness portion of it and created more data and also removed data from angle -0.1 to 0.1.
## Cropping the data
I cropped the top 40% and bottom 10% of data since I thought they were not useful in training and removing it will help me in training the model faster. So I reduced my image size to 80*320. Also I converted BGR to RGB, since cv2.imread reads in BRG and simulator was giving in RGB.

## Model
- I started with two pipelines. I found the links on Udacity forums. First was comma.ai one which wasn't working for me so I switched to the second one i.e. Nvidia pipepine. I played around with it and found some good paramters. 
- Initially I used the learning rate and other paramters as specified by [this](https://arxiv.org/pdf/1412.6980v8.pdf) paper. So the learning rate was 0.001 which didn't work for me so I tried playing around and found out and 0.0001 was working for me.
- I used ELU as my activation after each layer.
- I used dropout of 0.5 after each fully connected layer
- I ran my model for 10 epochs and batch size of 128. I tried with 15 epochs too but there was not much loss in mse.

I also have other decent models in the goodmodels folder in this repo. The first 4 was not passing the whole lap. The last one ith the folder name "fifth" passed the whole lap with a little glitch on the last steep right turn where it went too right but recovered itself. I will work on that.
