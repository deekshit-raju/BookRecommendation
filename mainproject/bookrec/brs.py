#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("C:/Users/deeks/Desktop/Main Project")

print(os.getcwd())


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz

# ### 1. Data loading and reading

# The data used for this project consist of 5 different datasets, which are loaded and presented above: 

# In[3]:


books = pd.read_csv('books.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')
bookTags = pd.read_csv('book_tags.csv')
toRead = pd.read_csv('to_read.csv')


# <br>

# --------

# <br>

# **1) Books** contains all the information about the rated books, including author, title, book ID, publication year, average rating, etc.

# In[4]:


books.head()


# In[5]:


books.shape    # 10.000 rows x 23 columns imported


# In[6]:


len(books.id.unique())  # There is information for 10.000 different books in total


# With the purpose of having a quick look at what are the books with the highest average rating, let's sort the dataset by average rating, selecting only the most relevant out of the 23 columns.
# Apparently, the selected sample of users seem to like quite a lot Bill Watterson's 'Calvin and Hobbes' comic series.

# In[7]:


# Book with the highest average rating:

books[['id', 'book_id', 'authors', 'title', 'average_rating', 'ratings_count']]      .sort_values('average_rating', ascending = False).head(20)


# <br>

# --------

# <br>

# **2) Ratings** includes all the rates given by our selected group of users to the books they have chosen to rate.

# In[8]:


ratings.head()


# In[9]:


ratings.shape  # 981.756 rows x 3 columns imported


# In[10]:


len(ratings.book_id.unique())  # There are ratings for 10.000 different books in total


# In[11]:


len(ratings.user_id.unique())  # Ratings to these 10k books have been given by 53.424 unique users


# <br>

# ------

# <br>

# **3) Tags** collects all the tags included in 'bookTags' and explains their meaning.

# In[12]:


tags.head()


# In[13]:


tags.head()   # 34.252 rows x 2 columns importedtags.shape    


# In[14]:


len(tags.tag_id.unique())  # There are 34.252 different tags, which means no duplicates


# <br>

# -----

# <br>

# **4) Book Tags** contains all the tags associated to each and every book included in the analysis.

# In[15]:


bookTags.head()


# In[16]:


bookTags.shape    # 999.912 rows imported x 3 columns imported


# In[17]:


len(bookTags.goodreads_book_id.unique())  # As expected, there are 10.000 books in total


# In[18]:


len(bookTags.tag_id.unique())  # As expected, there are 34.252 different tags in total


# <br>

# -------

# <br>

# **5) to Read** indicates all the books that have been flagged as 'to read' by the readers included in the analysis.

# In[19]:


toRead.head()


# In[20]:


toRead.shape    # 912.705 rows x 2 columns imported


# In[21]:


len(toRead.book_id.unique())  # 9.986 different books have been given the 'toRead' tag


# In[22]:


len(toRead.user_id.unique())   # 48.871 unique users have used the 'toRead' tag to flag at least one the 10k books


# <br>

# --------

# <br>

# #### 1.1 Adding own personal ratings to the dataset

# In order to make the analysis more interesting and easier to follow, it is good to select a specific user and keep track of him along the whole process, and see what recommendations the system will be able to give him. 
# For this purpose, I have leveraged my own personal Goodreads account, scraping all the ratings I have been given, since I had a Goodreads account, to the books I have read.

# The ratings are stored in the 'maxRatings.csv' file, that I have created through the carping process included in the previous notebook.

# In[23]:


maxRatings = pd.read_csv('maxRatings.csv')


# In[24]:


maxRatings.head()


# In[25]:


maxRatings.shape    # 97 rows x 3 columns imported


# For the purpose of the analysis, we have to assign a user ID to my account. In the following lines, I have checked that the maximum userID in the dataset is 53424, hence I have assigned the user number 55000 to my account, and appended a column that indicates it.

# In[26]:


max(ratings['user_id'])


# In[27]:


maxRatings['user_id'] = 55000


# In order to see how many of the 97 books I have actually read and rated are in the books dataset selected for this analysis, let's merge maxRatings with the rating dataset.

# In[28]:


maxRatingsWithId = pd.merge(maxRatings, books, on = 'title')
maxRatingsWithId.head()


# Next step will be to append my ratings to the ratings dataset. To do so, firstly I have to recondut maxRatings dataset to the exact same layout and format as the ratings one. Hence, I select only the columns needed, renaming one of them, given that 'id' in the 'books' dataset corresponds to 'book_id' in the rating dataset.

# In[29]:


ratings.head()


# In[30]:


maxRatingsWithId = maxRatingsWithId[['id', 'user_id', 'rating']]                     .rename(columns = {'id' : 'book_id'})


# In[31]:


maxRatingsWithId.head()


# In[32]:


ratings = ratings.append(maxRatingsWithId)


# In[33]:


len(ratings.user_id.unique())    # As expected, now ratings has one more unique user id


# <br>

# ------

# Taking a closer look at the ratings dataset I have realized that there are several users who have rated the same book more than once, for whatever reason.
# For this reason, we will proceed to erase those duplicated ratings from the rating table: for those user-book pairs that have more than 1 rating we will then keep only one record, indicating the average of the given ratings.

# In[34]:


# The following table will show which pair of user-book have more than 1 rating in the dataset: 

userRatesPerBookCount = ratings.groupby(['user_id', 'book_id'], as_index = False).count().sort_values('rating', ascending = False)
userRatesPerBookCount.head(15)


# In[35]:


# Let's double check if this is true:

ratings[(ratings.user_id == 3204) & (ratings.book_id == 8946)]  


# In[36]:


# Here I calculate the rating mean per user and book, so that I can later replace the mean to the rating value
# and get rid of the duplicates rows

userRatesPerBookMean = ratings.groupby(['user_id', 'book_id'], as_index = False).mean().round(0).sort_values('rating', ascending = False)
userRatesPerBookMean.head(15)


# In[37]:


# Let's double check if that worked ok:

userRatesPerBookMean[(userRatesPerBookMean.user_id == 3204) & (userRatesPerBookMean.book_id == 8946)]  


# In[38]:


# We create now a table grouped by user and book pair, calculating mean and number of rating that a user has made for the
# same book

ratings['count'] = ratings['rating']

userRatesPerBook = ratings.groupby(['user_id', 'book_id'], as_index = False)                    .agg({'rating' : 'mean', 'count' : 'count'})                    .rename(columns = {'rating':'mean'})                    .sort_values('count', ascending = False)

ratings = ratings.drop(['count'], axis = 1)
            
userRatesPerBook['mean'] = userRatesPerBook['mean'].round(0)
userRatesPerBook.head()


# In[39]:


# Now I append the mean and count per each user and book to the ratings table:

ratings = pd.merge(ratings, userRatesPerBook, on = ['user_id', 'book_id'])
ratings.head()


# In[40]:


ratings.sort_values('count', ascending = False).head(15)


# In[41]:


ratings.shape    # 981.820 rows x 5 columns


# In[42]:


# We can now drop the duplicates records for the combination user/book with more than 1 rating

ratings = ratings.drop_duplicates(subset = ['book_id', 'user_id'], keep = 'first', inplace = False)                                  .sort_values('count', ascending = False)


# In[43]:


ratings = ratings.drop(['rating', 'count'], axis = 1)


# In[44]:


ratings.head()


# In[45]:


ratings.shape  # 979.542 rows x 3 columns: 2.278 records removed


# In[ ]:





# In[46]:


# I restore the 'ratings' table column names as they were prior to the modification I have made
ratings.columns = ['book_id', 'user_id', 'rating']


# In[47]:


ratings.columns


# <br>

# With the purpose of making more relevant recommendations, let's have a look at what is the ratings distribution per users: in other words, let's see if there are users who have rated very few books.

# In[48]:


ratesPerUser = ratings[['user_id', 'rating']].groupby(['user_id'], as_index = False)                       .count()                       .rename(columns = {'rating' : 'totalRatings'})                       .sort_values('totalRatings', ascending = False)

ratesPerUser.head(10)


# In[49]:


plt.rcParams['figure.figsize'] = [20, 6]
plt.hist(ratesPerUser['totalRatings'], bins = 50)
plt.xticks(np.arange(min(ratesPerUser['totalRatings']), max(ratesPerUser['totalRatings'])+1, 4.0))
plt.show()


# The histogram above shows how many users have rated only 1 to 15 books. Considered that we are using only a subset (10k) of all the books listed in Goodreads, it will be pretty difficult to give accurate recommendation to users with that few ratings.  
# Cosnidering also the fact that this project has rather an illustrative purpose, I have decided to include only user that have rated at least 20 of the 10K books included in the analysis.

# In[50]:


ratings = pd.merge(ratings, ratesPerUser, on = ['user_id'])


# In[51]:


ratings.sort_values('totalRatings', ascending = False).head(10)


# In[52]:


# Keeping only users with more than a 'minimum' of ratings
minimumRatings = ratings.totalRatings <= 20
np.sum(minimumRatings)


# In[53]:


ratings = ratings[-minimumRatings]
len(ratings)   # only 707.456 ratings have been kept


# In[54]:


len(ratings.user_id.unique())  # total of remaining users after removing the ones with less than the established minimum


# In[55]:


len(ratings.book_id.unique())  # total of remaining books after removing users with less than the established minimum


# Then I had a look at the same thing but about the books: how many books have very few ratings? Looking at the results, I have decided - for the same reason as above, that is to avoid 'obscure' recommendations of books that only few users have read - to remove books with less than 30 ratings received.

# In[56]:


ratesPerBook = ratings[['book_id', 'rating']].groupby(['book_id'], as_index = False)                       .count()                       .rename(columns = {'rating' : 'receivedRatings'})                       .sort_values('receivedRatings', ascending = False)

ratesPerBook.tail()


# In[57]:


plt.rcParams['figure.figsize'] = [20, 6]
plt.hist(ratesPerBook['receivedRatings'], bins = 50)
plt.xticks(np.arange(min(ratesPerBook['receivedRatings']), max(ratesPerBook['receivedRatings'])+1, 4.0))
plt.show()


# In[58]:


ratings = pd.merge(ratings, ratesPerBook, on = ['book_id'])


# In[59]:


# Keeping only books with more than a 'minimum' of ratings received

minimumReceived = ratings.receivedRatings < 30
np.sum(minimumReceived)


# In[60]:


ratings = ratings[-minimumReceived]
len(ratings)


# In[61]:


len(ratings.user_id.unique())   # total of remaining users after removing the ones with less than the established minimum


# In[62]:


len(ratings.book_id.unique())    # total of remaining books after removing the ones with less than the established minimum


# In[63]:


ratings.shape


# <br>

# -------

# <br>

# To start off, I have tried to calculate a first type of Top-N recommendation: the top rated books. 
# After having removed the books and users above, the result doesn't change much compared to the what shown right after the  data loading: Bill Watterson and his Calvin and Hobbes is still dominating the chart.

# In[64]:


# For illustrative purpose, let's first grab author and title for the books dataset:
showingTitles = pd.merge(ratings, books[['id', 'authors', 'title']], how = 'left', left_on = ['book_id'], right_on = ['id'])


# In[65]:


topRated = showingTitles.groupby(['book_id'], as_index = False)                       .agg({'rating' : 'mean',                            'authors': 'first',                            'title' : 'first'})
topRated.sort_values('rating', ascending = False).head(10)


# <br>

# ---------

# <br>

# ### 2. Content-Based Filtering (Tag-Based)

# As long as the first Top-N recommendation created doesn't seem at all to be useful, I have build a "Content-Based" recommendation, leveraging the knowledge we have from the tags dataset using tags and other information from the book metadata.

# However, as shown below, the tags dataset contains format-free definition, which include a vast variety of tags. Most of them are not saying much of the book itself and its contect or characteristics (e.g. 'to-read', 'favourite', 'book-I-own', 'made-me-cry', etc).
# For this reason I have decide to include only the tags that are representative of the book 'genre' according to Goodreads itself: the tags used in fact are scraped from their genre section and contain a vast variety of tags. The others have been filtered out.

# In[66]:


showingTagName = pd.merge(bookTags, tags, on = 'tag_id')
showingTagName.sort_values('goodreads_book_id').head(20)


# In[67]:


mostUsedTags = showingTagName.groupby(['tag_name'], as_index = False)                       .agg({'goodreads_book_id' : 'count'})                       .rename(columns = {'goodreads_book_id' : 'number'})                       .sort_values('number', ascending = False)
mostUsedTags.head()


# <br>

# So I have used the genres.csv files, which was generated by scraping Goodreads website, and it is shown in the other notebook.

# In[68]:


genres = pd.read_csv('genres.csv')
genres.head(10)


# In[69]:


genres.shape    # 1235 rows X 2 columns


# In[70]:


# First I convert the genres to a list
genreList = genres['tag_name'].tolist()


# In[71]:


len(genreList)    # 1235 genre-tags have been scraped from Goodreads website


# In[72]:


len(tags.tag_name.unique())   # the original tag dataset included 34.252 tags


# In[73]:


genreTags = tags.loc[tags['tag_name'].isin(genreList)]
len(genreTags)    # 832 tags (of out the 1235) scraped from the Genre Section in Goodreads website are included in the 
                  # original tags table.


# Then I have merged the bookTags dataset with the genreTags, so that now I have the information of what books in the dataset have been tagged with at least one of the 832 genre-tags used by Goodreads in their genres section.
# 
# The idea is then to measure the similarity between each book pair, so that the system will be able to recommend Top-N similar books for any selected title from the ones included in the analysis.
# In order to be able to properly perform content based filtering, first I have to put all the tags related to one single book in a single string: each tag must be separated from each other with a space, to be accounted as a single word.
# Then I will perform pairwise similarity scores between books.

# In[74]:


mostCommonTags = pd.merge(bookTags, genreTags, on = ['tag_id'])


# In[75]:


stringedTags = mostCommonTags.groupby('goodreads_book_id')['tag_name'].apply(lambda x: "%s" % ' '.join(x)).reset_index()


# In[76]:


stringedTags.head(20)


# In[77]:


# I go retrieve the author and book information from the book dataset, so that I will be able to actually see
# the title of the book instead of just seeing the id.

stringedTags = pd.merge(stringedTags, books[['book_id', 'authors', 'title']], left_on = ['goodreads_book_id'],                        right_on = ['book_id']).drop('book_id', axis = 1)


# In[78]:


stringedTags.head(5)


# Other than the genre-tags,a very important information that we can add to the stringed tag of each book is the author. Including the author will produce a higher similarity score between 2 books written by the same author, which seems reasonable a as someone who likes a book from an author is most likely going to like also other books from the same person.

# In[79]:


# First of all, I put everything in lowercase and I remove the space between the name and the surname, to make it count
# as one single word all the time.
stringedTags['authors'] = stringedTags['authors'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))


# In[80]:


stringedTags.head(5)


# In[81]:


# With the same operation, I remove the comma between the names as well
stringedTags['authors'] = stringedTags['authors'].astype('str').apply(lambda x: str.lower(x.replace(",", " ")))


# In[82]:


stringedTags.head(20)


# In[83]:


# Then I add the author(s) to the stringed genre-tags
stringedTags['all_tags'] = stringedTags['tag_name'] + " " + stringedTags['authors']


# In[84]:


# A example with one of Harry Potter's series book:
stringedTags.all_tags[1]


# Now I use the 'CountVectorizer' function, that will generate a matrix in which the columns represent all the tag-words that I have included in the analysis and the rows represent the books.
# 
# From that matrix (the 'tagMatrix'), I have calculated a similarity score between each book pair, choosing the 'cosine' as a metric of distance.

# In[85]:


countVec = CountVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = 'english')


# In[86]:


tagMatrix = countVec.fit_transform(stringedTags['all_tags'])


# In[87]:


tagMatrix


# In[88]:


cosineSim = cosine_similarity(tagMatrix, tagMatrix)


# In[89]:


cosineSim.shape


# In[90]:


cosineSim


# The next step is retrieving the titles of the books (from the 'stringedTags' dataset, column 'title') and use tham as index, so that I can create a function, called 'topRecommendations' that takes in a book title and scores all cosine similarity of that title with each book in the dataset; then - after sorting by the highest similarity and after dropping the first observation (that will show the similarity of the book with itself, which is the maximum valu) it grabs only the first 10 most similar books to that title and return them in the form of a top-10 recommendation.

# In[91]:


stringedTags = stringedTags.reset_index()
bookTitles = stringedTags['title']
indices = pd.Series(stringedTags.index, index = bookTitles)


# In[92]:


def topRecommendations(title):
    index = indices[title]
    similarityScore = list(enumerate(cosineSim[index]))
    similarityScore = sorted(similarityScore, key = lambda x: x[1], reverse = True)
    similarityScore = similarityScore[1:10]
    bookIndex = [i[0] for i in similarityScore]
    return bookTitles.iloc[bookIndex]


# Having a look at few books, it looks like this content-based recommender system is able to capture similarity within books based on the tags I have used, giving interesting and relevant recommendations.

# In[93]:


topRecommendations('Harry Potter and the Half-Blood Prince (Harry Potter, #6)').head(10)


# In[94]:


topRecommendations('The Catcher in the Rye').head(10)


# In[95]:


topRecommendations('Fear and Loathing in Las Vegas').head(10)


# In[96]:


topRecommendations('The Great Gatsby').head(10)


# In[97]:


topRecommendations('Middlesex').head(10)


# In[98]:


topRecommendations('Slaughterhouse-Five').head(10)


# ### 3. Item-Based Collaborative Filtering

# Another type of recommender that I have decided to implement is based on 'Collaborative Filtering' technique. In this case, I have considered a Item-Based Collaborative Filtering.
# 
# The starting point of this type of system is a user by book matrix, in which each row indicates a single user, and each columns represents a single book.
# With the intent of making the table more easy to read and displaying the book titles rather than the book ids, I have used the 'ratingsWithTitles' table.
# 
# The resulting userByBook table contains an extremely high number of NAs, due to the fact that - as seen before - even the users who have read the highest number of books in our dataset, they have just read roughly 1% of all the books in the analysis.

# In[99]:


ratingsWithTitles = pd.merge(ratings, books[['id', 'title']], left_on = ['book_id'], right_on = ['id'])

book_user_mat = ratingsWithTitles.pivot(index='book_id', columns='user_id', values='rating').fillna(0)
print("the shape is",book_user_mat.shape)
def fuzzy_matching(mapper, fav_movie, verbose=True):
	match_tuple = []
	# get match
	for title, idx in mapper.items():
		if idx!=8884:
			ratio = fuzz.ratio(title.lower(), fav_movie.lower())
			if ratio >= 60:
				match_tuple.append((title, idx, ratio))
	# sort
	match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
	if not match_tuple:
		print('Oops! No match is found')
		return
	if verbose:
		print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
	return match_tuple[0][1]


def make_recommendation(model_knn, data, mapper, fav_movie, n_recommendations):
    # fit
    model_knn.fit(data)
    # get input movie index
    print('You have input book:', fav_movie)
    idx = fuzzy_matching(mapper, fav_movie, verbose=True)
    # inference
    print('Recommendation system start to make inference')
    print('......\n')
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
    # get list of raw idx of recommendations
    raw_recommends = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    # get reverse mapper
    reverse_mapper = {v: k for k, v in mapper.items()}
    # print recommendations
    print('Recommendations for {}:'.format(fav_movie))
    for i, (idx, dist) in enumerate(raw_recommends):
        print('{0}: {1}, with distance of {2}'.format(i+1, reverse_mapper[idx], dist))

book_to_idx = {book: i for i, book in enumerate(list(books.set_index('book_id').loc[book_user_mat.index].title))}
print("the len is",len(book_to_idx))
book_user_mat_sparse=csr_matrix(book_user_mat.values)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
make_recommendation(model_knn=model_knn,data=book_user_mat_sparse,fav_movie='The Hunger Games (The Hunger Games, #1)',mapper=book_to_idx,n_recommendations=10)
userByBook = pd.pivot_table(ratingsWithTitles, index = ['user_id'], columns = ['title'], values = ['rating'])


# In[101]:


userByBook.shape


# In[102]:


# Removing level in columns
userByBook.columns = userByBook.columns.droplevel(0)


'''# The goal in item-based collaborative filtering is to score the similarity between book pairs, based on the ratings they have received by the pool of users. To obtain a matrix in which each rows and columns will represent a single book and the values in the matrix represent the similarity between them, I have used the corr() function, which computes pairwise correlation of columns, excluding NA values, using Pearson as default method.

# In order to get better recommendations and avoid that not very relevant books will pop up in the Top-N list, I have played with the min_periods parameter within the corr() function, which allows to drop from the correlation computation all the books that are not supported by at least a certain number of people that rated both titles.
# I have set the parameter to 50, 40 and 30 with the intent of having a look at the three of them and see which one looks more reliable.

# In[ ]:
print("hai")

# Attempt number 2, using min_periods = 50
corrMatrix50 = userByBook.corr(method = 'pearson', min_periods = 50)

# In[105]:

def newRecommendations_50(title):
    bookRating = corrMatrix50[[title]][:]
    bookRating = bookRating.dropna()
    df = pd.DataFrame(bookRating.sort_values(by = title, ascending = False))[1:]
    return df.head(10)

print("Hai")
# In[106]:


newRecommendations_50('1984')


# In[107]:


# Attempt number 3, using min_periods = 40
corrMatrix40 = userByBook.corr(method = 'pearson', min_periods = 40)


# In[108]:


def newRecommendations_40(title):
    bookRating = corrMatrix40[[title]][:]
    bookRating = bookRating.dropna()
    df = pd.DataFrame(bookRating.sort_values(by = title, ascending = False))[1:]
    return df.head(10)


# In[109]:


newRecommendations_40('1984')


# In[110]:


# Attempt number 4, using min_periods = 30
corrMatrix30 = userByBook.corr(method = 'pearson', min_periods = 30)


# In[111]:


def newRecommendations_30(title):
    bookRating = corrMatrix30[[title]][:]
    bookRating = bookRating.dropna()
    df = pd.DataFrame(bookRating.sort_values(by = title, ascending = False))[1:]
    return df.head(10)


# In[112]:


newRecommendations_30('1984')


# In[113]:


newRecommendations_40('1984')


# In[114]:


newRecommendations_50('1984')


# In[115]:


newRecommendations_30('Slaughterhouse-Five')


# In[116]:


newRecommendations_40('Slaughterhouse-Five')


# In[117]:


newRecommendations_50('Slaughterhouse-Five')


# In[118]:


newRecommendations_30('Fear and Loathing in Las Vegas')


# In[119]:


newRecommendations_40('Fear and Loathing in Las Vegas')


# In[120]:


# Increasing the min_periods parameters to 40 and 50, some books don't receive any recommendations because
# there are not enough users in our sample who have rated that book.

newRecommendations_50('Fear and Loathing in Las Vegas')


# Given the above results, I have decided to stick with the min_perios = 40 setting, as long as 50 still includes some not satisfying recommendations and 30 remove too many points, not capturing few interesting recommendations.

# #### 3.1 Using item-based correlation matrix to make recommendations to specific users

# Once found the most appropriate way to score similarities between books, I can use the item-based correlation matrix to make recommendations to specific users.
# 
# For the purpose of explanation, I have used my own ratings (user id 55000). First, I isolate only the ratings of the books that the user in question has read, storing them as 'myRatings'. Changing user id in 'myRatings' definition and executing the rest of the code with no further modification, will generate recommendations for any other users.
# 
# After that, I have created an empty series in which I append all the similar movies to each of the ones I have rated, scaling the value by how well I have rated the specific movie.
# In order to give more importance to the books that have high correlation with more than one of the books I have read, I have summed the correlation values per book. The last two steps before getting the final Top-10 recommendations are to sort by aggregated similarity score and remove from the list the book I have already read.

# In[121]:


myRatings = userByBook.loc[55000].dropna()
myRatings


# In[122]:


simCandidates = pd.Series()

for i in range(0, len(myRatings.index)):
    sims = corrMatrix40[myRatings.index[i]].dropna()
    sims = sims.map(lambda x: x * myRatings[i])
    simCandidates =  simCandidates.append(sims)

simCandidates.sort_values(inplace = True, ascending = False)
print(simCandidates.head(10))


# In[123]:


simCandidates = simCandidates.groupby(simCandidates.index).sum()
simCandidates.head(10)


# In[124]:


simCandidates.sort_values(inplace = True, ascending = False)
simCandidates.head(10)


# In[125]:


filteredSims = simCandidates.drop(myRatings.index)
filteredSims.head(10)

'''