SemEval Task 7
Kat Young, Varsha Vatsavai, Humera Aamreen
File Written by Kat Young

Member Roles: For the code, part of it was collaborative between the whole group, and part done by Kat separately. These parts are labelled in the jupyter notebook where the code is. The final paper was also done collaboratively. Each section was split into three parts (one portion written by each of us) before the short-term checkpoints, and the last section (Results and Conclusion) was done by Varsha and Humera. The presentations were divided up as well. For the first presentation, Varsha presented the first two sections, Kat presented the second two, and Humera presented the last two. For the second presentation, Varsha and Humera each presented half (since Kat was traveling for volleyball). The install file was written by Varsha and Humera. Kat wrote this README.txt file.

Problem Explanation: The goal of the work we are doing is to write a model to detect humor in the way news headlines were edited. The original news headlines in the dataset are not necessarily considered funny, but by replacing one word a headline has the potential to become funny. For example, the sentence 'Trump is caught promising to 'release the ' on hot mic, would change the word 'memo' to 'kraken' to make it potentially funny. We are coming up with a model to judge the headline funniness based on the training data in the dataset which was annotated by a group of real people for funniness ratings. Funniness is up to bias and discretion which makes humor a challenging concept to judge via an algorithm.

Phase 1: To solve this problem, we began by processing the data from the dataset. First we obtained the sentence with the edit in place, that sentence divided into individual words of importance, word embeddings for each word in that sentence, as well as sentence embeddings. In the process of getting these new representations, we converted all words to lowercase, removed stop words, removed punctuation, removed some of the common words (such as 's' and 'nt', which represented possession and 'not'), removed rare words, and removed digits. The words were represented numerically through word embeddings. We got the word embeddings from GoogleNews-vectors. Then, we decided to look at running two different models: a simple neural network and a linear regression. The inputs to the simple neural network were the sentence embeddings, in the form of a single number which was the average of all the numbers from vectors in the sentences. The inputs to the linear regression were the sentence embeddings, which were matched up with the funniness scores. A set of predictions included [0.99423773, 0.97713483, 1.0045303, 0.97696265, 0.99313956], while the actual scores were [0.2, 1.6, 1.0, 0.4, 1.2]. In this set of predictions the only predictions that were somewhat close to correct were the third and the last. To see how far off our regression model was, we looked at what is called the "root mean square error" (RMSE), which essentially told us generally how far away from the true value our predictions were. We obtained 10 different RMSE values by dividing the training set up into 10 sections and using each section as the testing section (a method called 10-fold cross-validation). All of the RMSE values were near 0.5, which means the values we got were all around 0.5 off from the true value.

Phase 2: We changed the method of calculating the sentence embeddings such that the resulting sentence embedding would also be a 300 dimension vector that contained averages (in each vector position) of all the words in the sentence. These sentence embeddings were then fed into a Neural Network with a single LSTM, two-way (bi)LSTM, and different variations of biLSTM Neural Network (i.e. one with additional layers, different activation functions, different optimizers, and more). The biLSTM was implemented to be able to use the context in front of and behind the word in the analysis. We also implemented a Linear Regression using the new form of sentence embeddings. After each model is run, the MSE and accuracy can be seen. To get RMSE from MSE, we take the square root. After we run our initial models, we print out the RMSE values for each. The results look as follows:

Linear Regression RMSE: 0.5571047393476176
biLSTM RMSE (relu): 0.5817875898138096
biLSTM RMSE (additional layer): 0.5626690928067789
biLSTM RMSE base: 0.5575103067837445
LSTM RMSE base: 0.5576421632088139

Then, cross validation was executed on biLSTM (with Sigmoid activation function, dropout=0.3, batch size 16). These were found to be the most successful parameters. Two epochs were done for each validation, and the two RMSE values from each validation set are as follows:

Validation 1: 0.5573987624169929, 0.5557729861873885
Validation 2: 0.5547900375522811, 0.5531956939460426
Validation 3: 0.5581081808398746, 0.5547156774391875
Validation 4: 0.5569560534305721, 0.5550539641347731
Validation 5: 0.5573316405305709, 0.5544757990175536
Validation 6: 0.5575315204725305, 0.5551254392946691
Validation 7: 0.5574718322690859, 0.5544982908250331
Validation 8: 0.5555450724982951, 0.5529569491632386
Validation 9: 0.5547703419501918, 0.5526916471404962
Validation 10: 0.5564491083948483, 0.5543730390260023

See our Models: Luckily, GitHub displays the output from running the models without need of running them on your computer. Our GitHub repository with our code can be found on this link: https://github.com/youngk6/SemEval-Task-7 .

To run the program locally (unnecessary): Install all dependencies in the "install" file. Launch jupyter notebook (type in your command line: jupyter notebook). Navigate to our notebook file ("Sem Eval Task - Models.ipynb"). Click "Run All". This will take multiple hours. 
