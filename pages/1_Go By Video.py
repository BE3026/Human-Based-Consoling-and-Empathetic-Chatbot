import numpy as np
import streamlit as st
import pandas as pd
import cv2
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D



df=pd.read_csv('muse_v3.csv')

df['link']=df['lastfm_url']
df['name']=df['track']
df['emotional']=df['number_of_emotion_tags']
df['pleasant']=df['valence_tags']

df=df[['name','emotional','pleasant','link','artist']]

df=df.sort_values(by=["emotional","pleasant"])
df.reset_index()

df_sad=df[:18000]
df_fear=df[12000:36000]
df_angry=df[36000:54000]
df_neutral=df[54000:72000]
df_happy=df[72000:]

df2=pd.read_csv('movie.csv')
df2['director']=df2['director_name']
df2['title']=df2['movie_title']
df2['url']=df2['movie_imdb_link']

df2=df2[['index','title','director','url','genres']]

df2_action=pd.DataFrame()
df2_adventure=pd.DataFrame()
df2_drama=pd.DataFrame()
df2_comedy=pd.DataFrame()
df2_crime=pd.DataFrame()
df2_documentry=pd.DataFrame()
df2_family=pd.DataFrame()
df2_horror=pd.DataFrame()

i,j,k,l,m,o=0,0,0,0,0,0
for ind,row in df2.iterrows():
    if('Action' in row['genres'] and i<10):
        df2_action=df2_action._append(row)
        i=i+1
    if('Adventure' in row['genres'] and j<10):
        df2_adventure=df2_adventure._append(row)
        j=j+1
    if('Drama' in row['genres'] and k<10):
        df2_drama=df2_drama._append(row)
        k=k+1
    if('Comedy' in row['genres'] and l<10):
        df2_comedy=df2_comedy._append(row)
        l=l+1
    if('Horror' in row['genres'] and m<10):
        df2_horror=df2_horror._append(row)
        m=m+1
    if('Family' in row['genres'] and o<10):
        df2_family=df2_family._append(row)
        o=o+1
df3=pd.read_csv('regional_metadata.csv')
df3_sad=pd.DataFrame()
df3_fear=pd.DataFrame()
df3_angry=pd.DataFrame()
df3_neutral=pd.DataFrame()
df3_happy=pd.DataFrame()
for ind,row in df3.iterrows():
    if('sad' in row['Emotion']):
        df3_sad=df3_sad._append(row)
    if('fear' in row['Emotion']):
        df3_fear=df3_fear._append(row)
    if('angry' in row['Emotion']):
        df3_angry=df3_angry._append(row)
    if('happy' in row['Emotion']):
        df3_happy=df3_happy._append(row)
    if('neutral' in row['Emotion']):
        df3_neutral=df3_neutral._append(row)
def fun(list):
    data=pd.DataFrame()
    movie_data=pd.DataFrame()
    data2=pd.DataFrame()
    if len(list)==1:
        v=list[0]
        t=30
        if v=='Neutral':
            data=data._append(df_neutral.sample(n=t))
            movie_data=movie_data._append(df2_adventure)
            movie_data=movie_data._append(df2_action)
            data2=data2._append(df3_neutral)
        elif v== 'Angry':
            data=data._append(df_angry.sample(n=t))
            movie_data=movie_data._append(df2_drama)
            movie_data=movie_data._append(df2_comedy)
            movie_data=movie_data._append(df2_family)
            data2=data2._append(df3_angry)
        elif v=='fear':
            data=data._append(df_fear.sample(n=t))
            movie_data=movie_data._append(df2_drama)
            movie_data=movie_data._append(df2_comedy)
            movie_data=movie_data._append(df2_family)
            data2=data2._append(df3_fear)
        elif v=='happy':
            data=data._append(df_happy.sample(n=t))
            movie_data=movie_data._append(df2_adventure)
            movie_data=movie_data._append(df2_action)
            movie_data=movie_data._append(df2_horror)
            movie_data=movie_data._append(df2_adventure)
            data2=data2._append(df3_happy)
        else:
            data=data._append(df_sad.sample(n=t))
            movie_data=movie_data._append(df2_drama)
            movie_data=movie_data._append(df2_comedy)
            movie_data=movie_data._append(df2_family)
            data2=data2._append(df3_sad)

    elif len(list)==2:
        times = [20,10]
        for i in range(len(list)):
            v=list[i]
            t=times[i]
            if v=='Neutral':
                data=data._append(df_neutral.sample(n=t))
                movie_data=movie_data._append(df2_adventure)
                movie_data=movie_data._append(df2_action)
                data2=data2._append(df3_neutral)
            elif v== 'Angry':
                data=data._append(df_angry.sample(n=t))
                movie_data=movie_data._append(df2_drama)
                movie_data=movie_data._append(df2_comedy)
                movie_data=movie_data._append(df2_family)
                data2=data2._append(df3_angry)
            elif v=='fear':
                data=data._append(df_fear.sample(n=t))
                movie_data=movie_data._append(df2_drama)
                movie_data=movie_data._append(df2_comedy)
                movie_data=movie_data._append(df2_family)
                data2=data2._append(df3_fear)
            elif v=='happy':
                data=data._append(df_happy.sample(n=t))
                movie_data=movie_data._append(df2_adventure)
                movie_data=movie_data._append(df2_action)
                movie_data=movie_data._append(df2_horror)
                movie_data=movie_data._append(df2_adventure)
                data2=data2._append(df3_happy)
            else:
                data=data._append(df_sad.sample(n=t))
                movie_data=movie_data._append(df2_drama)
                movie_data=movie_data._append(df2_comedy)
                movie_data=movie_data._append(df2_family)
                data2=data2._append(df3_sad)
    elif len(list)==3:
        times = [15,10,5]
        for i in range(len(list)):
            v=list[i]
            t=times[i]
            if v=='Neutral':
                data=data._append(df_neutral.sample(n=t))
                movie_data=movie_data._append(df2_adventure)
                movie_data=movie_data._append(df2_action)
                data2=data2._append(df3_neutral)
            elif v== 'Angry':
                data=data._append(df_angry.sample(n=t))
                movie_data=movie_data._append(df2_drama)
                movie_data=movie_data._append(df2_comedy)
                movie_data=movie_data._append(df2_family)
                data2=data2._append(df3_angry)
            elif v=='fear':
                data=data._append(df_fear.sample(n=t))
                movie_data=movie_data._append(df2_drama)
                movie_data=movie_data._append(df2_comedy)
                movie_data=movie_data._append(df2_family)
                data2=data2._append(df3_fear)
            elif v=='happy':
                data=data._append(df_happy.sample(n=t))
                movie_data=movie_data._append(df2_adventure)
                movie_data=movie_data._append(df2_action)
                movie_data=movie_data._append(df2_horror)
                movie_data=movie_data._append(df2_adventure)
                data2=data2._append(df3_happy)
            else:
                data=data._append(df_sad.sample(n=t))
                movie_data=movie_data._append(df2_drama)
                movie_data=movie_data._append(df2_comedy)
                movie_data=movie_data._append(df2_family)
                data2=data2._append(df3_sad)  
    elif len(list)==4:
        times = [10,9,8,3]
        for i in range(len(list)):
            v=list[i]
            t=times[i]
            if v=='Neutral':
                data=data._append(df_neutral.sample(n=t))
                movie_data=movie_data._append(df2_adventure)
                movie_data=movie_data._append(df2_action)
                data2=data2._append(df3_neutral)
            elif v== 'Angry':
                data=data._append(df_angry.sample(n=t))
                movie_data=movie_data._append(df2_drama)
                movie_data=movie_data._append(df2_comedy)
                movie_data=movie_data._append(df2_family)
                data2=data2._append(df3_angry)
            elif v=='fear':
                data=data._append(df_fear.sample(n=t))
                movie_data=movie_data._append(df2_drama)
                movie_data=movie_data._append(df2_comedy)
                movie_data=movie_data._append(df2_family)
                data2=data2._append(df3_fear)
            elif v=='happy':
                data=data._append(df_happy.sample(n=t))
                movie_data=movie_data._append(df2_adventure)
                movie_data=movie_data._append(df2_action)
                movie_data=movie_data._append(df2_horror)
                movie_data=movie_data._append(df2_adventure)
                data2=data2._append(df3_happy)
            else:
                data=data._append(df_sad.sample(n=t)) 
                movie_data=movie_data._append(df2_drama)
                movie_data=movie_data._append(df2_comedy)
                movie_data=movie_data._append(df2_family) 
                data2=data2._append(df3_sad)  
    else:
        times = [10,7,6,5,2]
        for i in range(len(list)):
            v=list[i]
            t=times[i]
            if v=='Neutral':
                data=data._append(df_neutral.sample(n=t))
                movie_data=movie_data._append(df2_adventure)
                movie_data=movie_data._append(df2_action)
                data2=data2._append(df3_neutral)
            elif v== 'Angry':
                data=data._append(df_angry.sample(n=t))
                movie_data=movie_data._append(df2_drama)
                movie_data=movie_data._append(df2_comedy)
                movie_data=movie_data._append(df2_family)
                data2=data2._append(df3_angry)      
            elif v=='fear':
                data=data._append(df_fear.sample(n=t))
                movie_data=movie_data._append(df2_drama)
                movie_data=movie_data._append(df2_comedy)
                movie_data=movie_data._append(df2_family) 
                data2=data2._append(df3_fear) 
            elif v=='happy':
                data=data._append(df_happy.sample(n=t))
                movie_data=movie_data._append(df2_adventure)
                movie_data=movie_data._append(df2_action)
                movie_data=movie_data._append(df2_horror)
                movie_data=movie_data._append(df2_adventure)
                data2=data2._append(df3_happy)
            else:
                data=data._append(df_sad.sample(n=t))
                movie_data=movie_data._append(df2_drama)
                movie_data=movie_data._append(df2_comedy)
                movie_data=movie_data._append(df2_family)
                data2=data2._append(df3_sad)   

    return data,movie_data,data2

def pre(l):
    result = [item for items,c in Counter(l).most_common() for item in [items] * c]
    ul = []
    for x in result:
        if x not in ul:
            ul.append(x)
    return ul
model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7,activation='softmax'))

model.load_weights('model.h5')
emotion_dict={0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}

cv2.ocl.setUseOpenCL(False)
cap=cv2.VideoCapture(0)

st.markdown("<h2 style='text-align: center; color: white;'><b>Human Emotion Based Consoling And Empathetic Chatbot</b></h2>",unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: white;'><b>Select your mode</b></h5>",unsafe_allow_html=True)

col1,col2,col3=st.columns(3)

list=[]
with col1:
    pass
with col2:
    if st.button('SCAN EMOTION (Click Here)'):
        count=0
        list.clear()
        while True:
            ret,frame=cap.read()
            if not ret:
                break
            face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=face.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
            count=count+1

            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y-50),(x+w,y+h+10),(255,0,0),2)
                roi_gray=gray[y:y+h,x:x+w]

                cropped_img=np.expand_dims(np.expand_dims(cv2.resize(roi_gray,(48,48)),-1),0)
                prediction = model.predict(cropped_img)
                max_index=int(np.argmax(prediction))
                list.append(emotion_dict[max_index])

                cv2.putText(frame,emotion_dict[max_index],(x+20,y-60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                cv2.imshow('Video',cv2.resize(frame,(1000,700),interpolation=cv2.INTER_CUBIC))

            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

            if count >= 50:
                break

        cap.release()
        cv2.destroyAllWindows()

        list=pre(list)



with col3:
    pass

new_df,movie_data,data2=fun(list)

st.write("")



try:
    st.markdown("<h3 style='text-align:center;color:grey;'><b>Emotion Detected :- {}</b></h3>".format(list[0]),unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;color:grey;'><b>Songs With Artist Names</b></h2>",unsafe_allow_html=True)

    st.write("-------------------------------------------------------------------------------------------------------")
    no=1
    for l,a,n,i in zip(new_df["link"],new_df['artist'],new_df['name'],range(15)):
        st.markdown("""<h4 style='text-align:center;'><a href={}>{} - {} (English)</a></h4>""".format(l,no,n),unsafe_allow_html=True)
        st.markdown("<h4 style='text-align:center;color:grey'><i>{}</i></h5>".format(a),unsafe_allow_html=True)
        st.write("-------------------------------------------------------------------------------------------------------")
        no=no+1
    i=1
    for index,row in data2.iterrows():
        st.markdown("""<h4 style='text-align:center;'><a href={}>{} - {} ({})</a></h4>""".format(row['Source'],no,row['Song  Name'],row['Language']),unsafe_allow_html=True)
        st.markdown("<h4 style='text-align:center;color:grey'><i>{}</i></h5>".format(row['Artist']),unsafe_allow_html=True)
        st.write("-------------------------------------------------------------------------------------------------------")
        if(i>70):
            break
        i=i+1
        no=no+1
    st.markdown("<h2 style='text-align:center;color:grey;'><b>Movies With Director Names</b></h2>",unsafe_allow_html=True)
    i=1
    for index,row in movie_data.iterrows():
        st.markdown("""<h4 style='text-align:center;'><a href={}>{}-{}</a></h4>""".format(row["url"],i,row["title"]),unsafe_allow_html=True)
        st.markdown("<h4 style='text-align:center;color:grey'><i>Director - {} </i></h5>".format(row["director"]),unsafe_allow_html=True)
        st.write("-------------------------------------------------------------------------------------------------------")
        i=i+1
        if(i>30):
            break
        #print(row["title"],row["url"],row["director"])

except:
    pass
