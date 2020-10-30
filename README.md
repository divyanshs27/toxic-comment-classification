# toxic-comment-classification
Problem Statement and Background
Maintaining a decorum on an online forum is the need of the hour. Defined as “willful and repeated harm inflicted through the medium of electronic text”, cyber bullying puts targets under attack from a barrage of degrading, threatening, and/or sexually explicit messages and images conveyed using web sites, instant messaging, blogs, chat rooms, cell phones, web sites, e‐mail, and personal online profiles. Thus, the task of identifying and removing toxic communication from public forums is critical and is infeasible for human moderators.
The exact problem statement was thus as below:
Given a group of sentences or paragraphs, used as a comment by a user in an online platform, classify it to belong to one or more of the following categories — toxic, severe-toxic, obscene, threat, insult or identity-hate with either approximate probabilities or discrete values (0/1).
#Dataset and EDA
The Dataset used for this task is sourced from a Kaggle competition and is titled as the Jigsaw/Conversation AI Toxic Comment Classification Challenge Dataset. The creator have so far built a range of publicly available models served through the Perspective API and created this competition to enable participants to build a multi-headed model that is capable of detecting different types of toxicity like threats, obscenity, insults, and identity based hate better than their models. The dataset is composed of comments from Wikipedia’s talk page edits. The various categorizations for the comments are: toxic, severe toxic, obscene, threat, insult, and identity hate.The training dataset consists of 160k training samples and the test set consists of 153k samples.Understanding the dataset is an extremely vital task and there are several insights to be drawn from the dataset.
#Word Frequencies
Calculate word frequencies: Calculate Term Frequency-Inverse Document Frequency which are the components of the resulting scores assigned to each word.
- Term Frequency: Summarizes how often a given word appears within a document
- IDF: Downscales words that appear a lot across documents
This highlights those words that are more interesting, e.g. frequent in a document but not across documents. The weight of a term that occurs in a document is simply proportional to the term frequency.
#Model: Multinomial Logistic Regression
Multinomial Logistic Regression is a classification method that generalizes logistic regression to multiclass problems and predicts the probabilities of the different possible outcomes of a categorically distributed dependent variable, given a set of independent variable. The Python scikit learn library has an inbuilt tool that can be used for this task and for our first baseline model

deployement link- https://toxic-comment-classification.herokuapp.com/
