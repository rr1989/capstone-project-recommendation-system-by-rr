
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def sentiment_result(product_name,Pickled_RFC_Model):
  import pandas as pd
  #product_name = "Clorox Disinfecting Bathroom Cleaner"



  from sklearn.model_selection import cross_val_score
  from scipy.sparse import hstack
  from sklearn.feature_extraction.text import TfidfVectorizer
  Ebuss_df = pd.read_csv("./data/sample30.csv")
  Ebuss_df_subset = Ebuss_df.dropna(subset=['reviews_text'])
  review_text=Ebuss_df_subset['reviews_text']
  #TF-IDF
  word_vectorizer = TfidfVectorizer(
  sublinear_tf=True,
  strip_accents='unicode',
  analyzer='word',
  token_pattern=r'\w{1,}',
  stop_words='english',
  ngram_range=(1, 1),
  max_features=10000)
  word_vectorizer.fit(review_text)
  char_vectorizer = TfidfVectorizer(
  sublinear_tf=True,
  strip_accents='unicode',
  analyzer='char',
  stop_words='english',
  ngram_range=(2, 6),
  max_features=50000)
  char_vectorizer.fit(review_text)

  index= Ebuss_df[Ebuss_df['name']==product_name].index.tolist()

  item_text = Ebuss_df.loc[index]['reviews_text']


  train_word_features = word_vectorizer.transform(item_text)

  train_char_features = char_vectorizer.transform(item_text)

  item_features = hstack([train_char_features, train_word_features])

  predicted_result = Pickled_RFC_Model.predict(item_features)

  if predicted_result.mean() > 4:
    return 1
  else:
    return 0


def recommendation_system(p_username):
    import pandas as pd



    import pickle

    # Load the Model back from file
    from sklearn.externals import joblib
    Pkl_Filename = "Pickle_LR1_Model.pkl"

    Pickled_RFC_Model = joblib.load("./models/Pickle_LR1_Model.pkl")

    #Pickled_RFC_Model

    #recommendation system

    # import libraties
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    ratings = pd.read_csv('./data/sample30.csv' , encoding='latin-1')
    #ratings.head()

    ratings = ratings[["reviews_username","name","reviews_rating"]]

    # Test and Train split of the dataset.
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(ratings, test_size=0.30, random_state=31)

    train = train.groupby(by=["reviews_username",'name']).mean()

    train.reset_index(level=0, inplace=True)
    train.reset_index(level=0, inplace=True)

    # Item Based Similarity
    df_pivot = train.pivot(
    index='reviews_username',
    columns='name',
    values='reviews_rating').T

    #df_pivot.head()

    mean = np.nanmean(df_pivot, axis=1)
    df_subtracted = (df_pivot.T-mean).T

    from sklearn.metrics.pairwise import pairwise_distances

    # Item Similarity Matrix
    item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
    item_correlation[np.isnan(item_correlation)] = 0

    item_correlation[item_correlation<0]=0

    #prediction
    item_predicted_ratings = np.dot((df_pivot.fillna(0).T),item_correlation)

    # Copy the train dataset into dummy_train
    dummy_train = train.copy()

    # The products not rated by user is marked as 1 for prediction.
    dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)

    # Convert the dummy train dataset into matrix format.
    dummy_train = dummy_train.pivot(
    index='reviews_username',
    columns='name',
    values='reviews_rating').fillna(1)

    item_final_rating = np.multiply(item_predicted_ratings,dummy_train)

    user_input = p_username

    d = item_final_rating.loc[user_input].sort_values(ascending=False)#[0:5]


    j = 0
    k = 0
    predicted_list = []
    original_list = []
    for i in d:
        predicted_result = sentiment_result(d.index[j],Pickled_RFC_Model)
        #print(d.index[j])
        #print(predicted_result)
        if predicted_result == 1:
            #print(d.index[j] + ' : ' + str(i))
            predicted_list.append(d.index[j])
            k = k + 1
            if k == 5:
              break
        original_list.append(d.index[j])
        j = j + 1
        #print(j)
    #return d,predicted_list,original_list
    return predicted_list
