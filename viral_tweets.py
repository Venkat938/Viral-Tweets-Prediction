# essentials
import pandas as pd
import numpy as np

# surpress warnings
import warnings
warnings.filterwarnings('ignore')


# model building
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_tweets = pd.read_csv(r'C:\Users\venkat\Downloads\Viral Tweets Prediction Challenge Dataset\Dataset\Tweets\train_tweets.csv')
train_tweets_vectorized_media = pd.read_csv(r'C:\Users\venkat\Downloads\Viral Tweets Prediction Challenge Dataset\Dataset\Tweets\train_tweets_vectorized_media.csv')
train_tweets_vectorized_text = pd.read_csv(r'C:\Users\venkat\Downloads\Viral Tweets Prediction Challenge Dataset\Dataset\Tweets\train_tweets_vectorized_text.csv')

# Load test dataset
test_tweets = pd.read_csv(r'C:\Users\venkat\Downloads\Viral Tweets Prediction Challenge Dataset\Dataset\Tweets\test_tweets.csv')
test_tweets_vectorized_media = pd.read_csv(r'C:\Users\venkat\Downloads\Viral Tweets Prediction Challenge Dataset\Dataset\Tweets\test_tweets_vectorized_media.csv')
test_tweets_vectorized_text = pd.read_csv(r'C:\Users\venkat\Downloads\Viral Tweets Prediction Challenge Dataset\Dataset\Tweets\test_tweets_vectorized_text.csv')

# Load user dataset
users = pd.read_csv(r'C:\Users\venkat\Downloads\Viral Tweets Prediction Challenge Dataset\Dataset\Users\users.csv')
user_vectorized_descriptions = pd.read_csv(r'C:\Users\venkat\Downloads\Viral Tweets Prediction Challenge Dataset\Dataset\Users\user_vectorized_descriptions.csv')
user_vectorized_profile_images = pd.read_csv(r'C:\Users\venkat\Downloads\Viral Tweets Prediction Challenge Dataset\Dataset\Users\user_vectorized_profile_images.csv')


train_tweets=train_tweets.ffill(axis=0)
cols = ['tweet_hashtag_count', 'tweet_url_count', 'tweet_mention_count']
train_tweets[cols] = train_tweets[cols].applymap(np.int64)
train_tweets[cols].head()
topic_ids = (
    train_tweets['tweet_topic_ids'].str.strip('[]').str.split('\s*,\s*').explode()
    .str.get_dummies().sum(level=0).add_prefix('topic_id_')
) 
topic_ids.rename(columns = lambda x: x.replace("'", ""), inplace=True)
year = pd.get_dummies(train_tweets.tweet_created_at_year, prefix='year')
month = pd.get_dummies(train_tweets.tweet_created_at_month , prefix='month')
day = pd.get_dummies(train_tweets.tweet_created_at_day, prefix='day')
attachment = pd.get_dummies(train_tweets.tweet_attachment_class, prefix='attatchment')
language = pd.get_dummies(train_tweets.tweet_language_id, prefix='language')

## Cyclical encoding
sin_hour = np.sin(2*np.pi*train_tweets['tweet_created_at_hour']/24.0)
sin_hour.name = 'sin_hour'
cos_hour = np.cos(2*np.pi*train_tweets['tweet_created_at_hour']/24.0)
cos_hour.name = 'cos_hour'
columns_drop = [
                "tweet_topic_ids",
                "tweet_created_at_year",
                "tweet_created_at_month",
                "tweet_created_at_day",
                "tweet_attachment_class",
                "tweet_language_id",
                "tweet_created_at_hour",
               ]

dfs = [topic_ids, year, month, day, attachment, language, 
       sin_hour, cos_hour]

train_tweets_final = train_tweets.drop(columns_drop, 1).join(dfs)

train_tweets_final.head()


year = pd.get_dummies(users.user_created_at_year, prefix='year')
month = pd.get_dummies(users.user_created_at_month , prefix='month')
user_verified = pd.get_dummies(users.user_verified, prefix='verified')

columns_drop = [
                "user_created_at_year",
                "user_created_at_month",
                "user_verified"
              ]

dfs = [
        year,
        month,
        user_verified
      ]

users_final = users.drop(columns_drop, 1).join(dfs)

users_final.head()



vectorized_media_df = pd.merge(train_tweets,train_tweets_vectorized_media, on ='tweet_id', how = 'right')
vectorized_media_df.drop(train_tweets.columns.difference(['virality']), axis=1, inplace=True)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
y = vectorized_media_df['virality']
x = vectorized_media_df.loc[:, vectorized_media_df.columns.str.contains("img_")] 
lda = LDA(n_components = None)
x = lda.fit_transform(x, y)
print(vectorized_media_df.shape)
x=pd.DataFrame(x)
train_tweets_media_final = pd.concat([train_tweets_vectorized_media[['media_id', 'tweet_id']], x], axis=1)
print(train_tweets_media_final.shape)
train_tweets_media_final.columns=['media_id', 'tweet_id','img_0','img_1','img_2','img_3']

vectorized_text_df = pd.merge(train_tweets,train_tweets_vectorized_text, on ='tweet_id', how = 'right')
vectorized_text_df.drop(train_tweets.columns.difference(['virality']), axis=1, inplace=True)
y1 = vectorized_text_df['virality']
x1 = vectorized_text_df.loc[:, train_tweets_vectorized_text.columns.str.contains("feature_")] 
x1 = lda.fit_transform(x1, y1)
print(x1.shape)
x1=pd.DataFrame(x1)
train_tweets_text_final = pd.concat([train_tweets_vectorized_text[['tweet_id']], x1], axis=1)
train_tweets_text_final.columns=['tweet_id','feature_0','feature_1','feature_2','feature_3']



#x=vmediadf, x1=vtextdf, x2=user desc, x3=profile img

average_virality_df =train_tweets_final.groupby('tweet_user_id').agg(pd.Series.median)['virality']

descriptions_df = pd.merge(average_virality_df, user_vectorized_descriptions, left_on ='tweet_user_id', right_on = 'user_id', how = 'right')
profile_images_df = pd.merge(average_virality_df, user_vectorized_profile_images, left_on ='tweet_user_id', right_on = 'user_id', how = 'right')
y2 = descriptions_df['virality']
x2 = descriptions_df.loc[:, descriptions_df.columns.str.contains("feature_")] 
x2=lda.fit_transform(x2,y2)
print(x2.shape)
x2 = pd.DataFrame(x2)
user_descriptions_final = pd.concat([user_vectorized_descriptions[['user_id']], x2], axis=1)
user_descriptions_final.columns=['user_id','feature_x','feature_y','feature_z']



y3 = profile_images_df['virality']
x3 = profile_images_df.loc[:, profile_images_df.columns.str.contains("feature_")] 
x3=lda.fit_transform(x3,y3)
x3 = pd.DataFrame(x3)
user_profile_images_final = pd.concat([user_vectorized_profile_images[['user_id']], x3], axis=1)
user_profile_images_final.columns=['user_id','feature_x1','feature_y1','feature_z1']




print(train_tweets_final.shape)
print(train_tweets_media_final.shape) # join on tweet id
print(train_tweets_text_final.shape) # join on tweet id
print(users_final.shape) # join on user_id
print(user_profile_images_final.shape) # join on user_id




media_df =train_tweets_media_final.groupby('tweet_id').mean()
cols = train_tweets_text_final.columns[train_tweets_text_final.columns.str.contains('feature_')]
train_tweets_text_final.rename(columns = dict(zip(cols, 'text_' + cols)), inplace=True)
train_tweets_text_final.head()

tweet_df = pd.merge(media_df, train_tweets_text_final, on = 'tweet_id', how = 'right')
tweet_df.fillna(0, inplace=True)

# join users data
user_df = pd.merge(users_final, user_profile_images_final, on='user_id')

# join tweets data on train_tweets
tweet_df_final = pd.merge(train_tweets_final, tweet_df, on = 'tweet_id')

# join that with the users data
final_df = pd.merge(tweet_df_final, user_df, left_on = 'tweet_user_id', right_on='user_id')

final_df.shape

X = final_df.drop(['virality', 'tweet_user_id', 'tweet_id', 'user_id'], axis=1)
y = final_df.iloc[:,6].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 600, criterion = 'entropy', random_state = 41, max_depth=40)
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)
accuracy=accuracy_score(y_pred, y_test)

test_tweets.ffill(axis=0)
cols = ['tweet_hashtag_count', 'tweet_url_count', 'tweet_mention_count']
test_tweets[cols] = test_tweets[cols].applymap(np.int64)
test_tweets[cols].head()
topic_ids = (
    test_tweets['tweet_topic_ids'].str.strip('[]').str.split('\s*,\s*').explode()
    .str.get_dummies().sum(level=0).add_prefix('topic_id_')
) 
topic_ids.rename(columns = lambda x: x.replace("'", ""), inplace=True)

year = pd.get_dummies(test_tweets.tweet_created_at_year, prefix='year')
month = pd.get_dummies(test_tweets.tweet_created_at_month , prefix='month')
day = pd.get_dummies(test_tweets.tweet_created_at_day, prefix='day')
attachment = pd.get_dummies(test_tweets.tweet_attachment_class, prefix='attatchment')
language = pd.get_dummies(test_tweets.tweet_language_id, prefix='language')

## Cyclical encoding
sin_hour = np.sin(2*np.pi*test_tweets['tweet_created_at_hour']/24.0)
sin_hour.name = 'sin_hour'
cos_hour = np.cos(2*np.pi*test_tweets['tweet_created_at_hour']/24.0)
cos_hour.name = 'cos_hour'

test_tweets=test_tweets.ffill(axis=0)
columns_drop = [
                "tweet_topic_ids",
                "tweet_created_at_year",
                "tweet_created_at_month",
                "tweet_created_at_day",
                "tweet_attachment_class",
                "tweet_language_id",
                "tweet_created_at_hour",
              ]

dfs = [
        topic_ids,
        year,
        month,
        day,
        attachment,
        language,
        sin_hour,
        cos_hour,
      ]

test_tweets_final = test_tweets.drop(columns_drop, 1).join(dfs)

test_tweets_final.head()


# columns missing in test from train
cols_test = set(test_tweets_final.columns) - set(train_tweets_final.columns)
print(cols_test)
for col in cols_test:
  final_df[col] = 0
cols_train = set(train_tweets_final.columns) - set(test_tweets_final.columns)
cols_train.remove('virality') # remove virality from columns to add to test
len(cols_train)

for col in cols_train:
  test_tweets_final[col] = 0
test_tweets_media_final = pd.concat([test_tweets_vectorized_media[['media_id', 'tweet_id']], x], axis=1)
test_tweets_text_final = pd.concat([test_tweets_vectorized_text[['tweet_id']], x], axis=1)
test_tweets_text_final.columns=['tweet_id','feature_0','feature_1','feature_2','feature_3']
test_tweets_media_final.columns=['media_id', 'tweet_id','img_0','img_1','img_2','img_3']

media_df = test_tweets_media_final.groupby('tweet_id').mean()

cols = test_tweets_text_final.columns[test_tweets_text_final.columns.str.contains('feature_')]
test_tweets_text_final.rename(columns = dict(zip(cols, 'text_' + cols)), inplace=True)

# join tweets data
tweet_df = pd.merge(media_df, test_tweets_text_final, on = 'tweet_id', how = 'right')
tweet_df.fillna(0, inplace=True)

# join users data
user_df = pd.merge(users_final, user_profile_images_final, on='user_id')

# join tweets data on train_tweets
tweet_df_final = pd.merge(test_tweets_final, tweet_df, on = 'tweet_id')

# join that with the users data
p_final_df = pd.merge(tweet_df_final, user_df, left_on = 'tweet_user_id', right_on='user_id')

p_final_df.shape



X = p_final_df.drop(['tweet_user_id', 'tweet_id', 'user_id','language_26', 'topic_id_117', 'topic_id_123', 'topic_id_38'], axis=1)

ID=test_tweets['tweet_id']
solution = classifier.predict(X)
solution_df = pd.concat([p_final_df[['tweet_id']], pd.DataFrame(solution, columns = ['virality'])], axis=1)
solution_format = pd.read_csv(r'C:\Users\venkat\Downloads\Viral Tweets Prediction Challenge Dataset\Dataset\solution_format.csv')

solution1=pd.merge(solution_format,solution_df,  on = 'tweet_id')
solution1=solution1.drop(['virality_x'],axis=1)
solution1.to_csv('solution5.csv', index=False)
