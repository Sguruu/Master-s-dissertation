import pandas
import matplotlib.pyplot as plt
import numpy as np

# for i in range(10):
#    data = pandas.read_excel(f"{i+1}.xlsx")
#    data = pandas.get_dummies(data)
#    data.to_excel(f"1_{i+1}.xlsx")
anime = pandas.read_csv("anime.csv")
ganreDt = pandas.read_excel("1_1.xlsx")
print(ganreDt.columns)
ganreDt.drop(["Unnamed: 0"], axis=1, inplace=True)
print(ganreDt.columns)
anime = pandas.concat([anime, ganreDt], axis=1)
# anime.to_excel("tt.xlsx")
print(anime.columns)
anime.drop(["anime_id"], axis=1, inplace=True)
anime.drop(["genre"], axis=1, inplace=True)
anime.drop(anime[(anime["episodes"] == "Unknown")].index, inplace=True)
anime = pandas.get_dummies(anime, columns=["type"])


def clean_dataset(df):
    assert isinstance(df, pandas.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(pandas.np.float64)


anime.dropna(inplace=True)
anime = anime.head(2000)
# anime = clean_dataset(anime)
anime = anime.reset_index()  # Сбрасываем индекс
X = anime.drop("name", axis=1)
print("Посмотри сюда 1 \n ", X.columns)
X = X.drop("index", axis=1)
print("Посмотри сюда 2 \n ", X.columns)
y = anime["name"]

print("Посмотри сюда\n ", y, y.shape)
print(y.head(3))

y.to_excel("tt.xlsx")
# print(X.columns)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200)
model.fit(X, y)
# print("Вывод данных для работы")
# print({col: [0] for col in X.columns})
print("Посмотри сюда\n ", X.columns)
# print({col: [0] for col in X.columns})
exemple = {'episodes': [20], 'rating': [40000], 'members': [100000],
           'genre_Action': [0], 'genre_Adventure': [0],
           'genre_Cars': [0], 'genre_Comedy': [0], 'genre_Dementia': [0], 'genre_Demons': [0], 'genre_Drama': [1],
           'genre_Ecchi': [0], 'genre_Fantasy': [0], 'genre_Game': [0], 'genre_Harem': [0], 'genre_Hentai': [1],
           'genre_Historical': [0], 'genre_Horror': [1], 'genre_Josei': [0], 'genre_Kids': [0], 'genre_Magic': [0],
           'genre_Martial Arts': [0], 'genre_Mecha': [0], 'genre_Military': [0], 'genre_Music': [0],
           'genre_Mystery': [0], 'genre_Parody': [0], 'genre_Police': [0], 'genre_Psychological': [1],
           'genre_Romance': [0], 'genre_Samurai': [0], 'genre_School': [0], 'genre_Sci-Fi': [0], 'genre_Seinen': [0],
           'genre_Shoujo': [0], 'genre_Shounen': [0], 'genre_Slice of Life': [0], 'genre_Space': [0],
           'genre_Sports': [0],
           'genre_Super Power': [0], 'genre_Supernatural': [0], 'genre_Thriller': [1], 'genre_Vampire': [0],
           'genre_Yaoi': [0],
           'type_Movie': [1], 'type_Music': [0], 'type_ONA': [0], 'type_OVA': [0],
           'type_Special': [0], 'type_TV': [0]}

exemple_df = pandas.DataFrame(exemple)
print("Тебе подойдет фильм по ленейной регресии : ", model.predict(exemple_df))

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
model = TruncatedSVD()
model.fit(X_train, y_train)

xtest = model.fit_transform(X_test, y_train)
print(xtest)
