import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
def main():
    activities=['About','Toxic Comment Classification System','Developer']
    option=st.sidebar.selectbox('Menu Bar:',activities)
    if option=='About':
        html_temp = """
        <div style = "background-color: yellow; padding: 10px;">
            <center><h1>ABOUT PROJECT</h1></center>
        </div><br>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        image=Image.open('Sources/1st.jpeg')
        st.image(image,use_column_width=True)
        st.title("Problem Statement and Background ")
        st.subheader("Maintaining a decorum on an online forum is the need of the hour. Defined as “willful and repeated harm inflicted through the medium of electronic text”, cyber bullying puts targets under attack from a barrage of degrading, threatening, and/or sexually explicit messages and images conveyed using web sites, instant messaging, blogs, chat rooms, cell phones, web sites, e‐mail, and personal online profiles. Thus, the task of identifying and removing toxic communication from public forums is critical and is infeasible for human moderators.")
        
       
        st.header("The exact problem statement was thus as below:")
        st.subheader("Given a group of sentences or paragraphs, used as a comment by a user in an online platform, classify it to belong to one or more of the following categories — toxic, severe-toxic, obscene, threat, insult or identity-hate with either approximate probabilities or discrete values (0/1). Consider the following examples :")
        image=Image.open('Sources/1.png')
        st.image(image,use_column_width=True)
        st.title("Dataset and EDA")
        st.subheader("The Dataset used for this task is sourced from a Kaggle competition and is titled as the Jigsaw/Conversation AI Toxic Comment Classification Challenge Dataset. The creator have so far built a range of publicly available models served through the Perspective API and created this competition to enable participants to build a multi-headed model that is capable of detecting different types of toxicity like threats, obscenity, insults, and identity based hate better than their models. The dataset is composed of comments from Wikipedia’s talk page edits. The various categorizations for the comments are: toxic, severe toxic, obscene, threat, insult, and identity hate.The training dataset consists of 160k training samples and the test set consists of 153k samples.Understanding the dataset is an extremely vital task and there are several insights to be drawn from the dataset. ")
        image=Image.open('3rd.png')
        st.image(image,use_column_width=True)
        st.subheader("Another interesting data analysis that has been performed is the analysis of the correlation across several parameters shown in the following figures.")
        image=Image.open('4th.png')
        st.image(image,use_column_width=True)
        image=Image.open('5th.png')
        st.image(image,use_column_width=True)
       
        st.title("Word Frequencies")
        st.subheader("Calculate word frequencies: Calculate Term Frequency-Inverse Document Frequency which are the components of the resulting scores assigned to each word.")
        st.subheader("- Term Frequency: Summarizes how often a given word appears within a document")
        st.subheader("- IDF: Downscales words that appear a lot across documents ")
        st.subheader("This highlights those words that are more interesting, e.g. frequent in a document but not across documents. The weight of a term that occurs in a document is simply proportional to the term frequency.")
        pics = {
                    "Toxic": Image.open('6th.png'),
                    "Severe_Toxic": Image.open('7th.png'),
                    "Obscene": Image.open('8th.png'),
                    "Threat": Image.open('9th.png'),
                    "Insult": Image.open('10th.png'),
                    "Identity_hate": Image.open('11th.png'),

                    
                }
        pic = st.selectbox("Picture choices", list(pics.keys()), 0)
        st.image(pics[pic], use_column_width=True, caption=pics[pic])
        st.title("Model: Multinomial Logistic Regression")
        st.subheader("Multinomial Logistic Regression is a classification method that generalizes logistic regression to multiclass problems and predicts the probabilities of the different possible outcomes of a categorically distributed dependent variable, given a set of independent variable. The Python scikit learn library has an inbuilt tool that can be used for this task and for our first baseline model")
    elif option=='Toxic Comment Classification System':
        st.title("Toxic Comment Classification System")
        train = pd.read_csv('Data/train_final.csv')
        st.set_option('deprecation.showfileUploaderEncoding', False)#to remove error
        st.subheader("Please Upload Your Dataset")
        data=st.file_uploader("Upload your dataset",type=['csv','xlsx','txt','json'])
        if data is not None:
            test = pd.read_csv(data)
            st.dataframe(test.head(10))
            st.success("Data Successfully loaded")
            if st.button("Predict"):
                v = TfidfVectorizer()
                X_train = v.fit_transform(train['comment_text'])
                X_test = v.transform(test['comment_text'])
                for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
                    y = train[label]
                    model = LogisticRegression()
                    model.fit(X_train, y)
                    test[label] = model.predict_proba(X_test)[:, 1]
    
                test.drop('comment_text', axis=1, inplace=True)
                test.to_csv('solution.csv', index=False)
                #st.dataframe(test.style.highlight_max(axis=0)
                st.balloons()  
                st.dataframe(test.head(10))
    
    elif option=='Developer':
        st.balloons()
        st.title('Prepared by:-')
        st.header('Divyansh Singhal :sunglasses:')
        st.subheader('Machine Learning Intern, Technocolab')
        html= '<a href="https://github.com/divyanshs27/toxic-comment-classification" target="_blank" class="nav-link"><span style="font-size: 20px;color:black;" class="fa fa-github"></span>Github</a>'
        st.write(html,unsafe_allow_html=True)




            
if __name__ == '__main__':
    main()
    

