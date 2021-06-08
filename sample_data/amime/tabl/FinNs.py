import pandas
import matplotlib.pyplot as plt
import numpy as np

# Удаление лишних колонок и обработка нужных
ganreDt = pandas.read_excel("1_1.xlsx")
print(ganreDt.columns)
ganreDt.drop(["Unnamed: 0"], axis=1, inplace=True)
print(ganreDt.columns)


anime = pandas.read_csv("anime.csv")
# Удаление строк с неизвестными данными
anime.drop(anime[(anime["episodes"] == "Unknown")].index, inplace=True)

anime["type"].hist()
plt.show()

# Просмотров
anime["members"].plot()
plt.show()

anime = anime.head(1000)

# Вывод диаграмм и графиков
anime["type"].hist()
plt.show()

# Просмотров
anime["members"].plot()
plt.show()

anime = pandas.concat([anime, ganreDt], axis=1)
print(anime.columns)
anime.drop(["anime_id"], axis=1, inplace=True)
anime.drop(["genre"], axis=1, inplace=True)

# преобрахование колонки тип в нужный вид, для входных данных
anime = pandas.get_dummies(anime, columns=["type"])

# Удалние пустых данных
anime.dropna(inplace=True)

anime = anime.reset_index()  # Сбрасываем индекс
# Создание входных данных X, выходных данных y
X = anime.drop("name", axis=1)
X = X.drop("index", axis=1)
y = anime["name"]

print("Посмотри сюда\n ", y, y.shape)
print(y.head(3))

# Тестируем предсказание с помощью линейной регрессии
from sklearn.linear_model import LogisticRegression

# создаем модель и настраиваем ее
model = LogisticRegression(max_iter=200)
# Загружаем в нашу модель данные
model.fit(X, y)

# Создаем данные пользователя для ввода и следующего предсказания
# print({col: [0] for col in X.columns})
exemple = {'episodes': [100], 'rating': [1000], 'members': [120],
           'genre_Action': [0], 'genre_Adventure': [0],
           'genre_Cars': [0], 'genre_Comedy': [0], 'genre_Dementia': [0], 'genre_Demons': [1], 'genre_Drama': [0],
           'genre_Ecchi': [0], 'genre_Fantasy': [0], 'genre_Game': [0], 'genre_Harem': [0], 'genre_Hentai': [0],
           'genre_Historical': [1], 'genre_Horror': [0], 'genre_Josei': [0], 'genre_Kids': [0], 'genre_Magic': [0],
           'genre_Martial Arts': [0], 'genre_Mecha': [0], 'genre_Military': [0], 'genre_Music': [0],
           'genre_Mystery': [0], 'genre_Parody': [0], 'genre_Police': [0], 'genre_Psychological': [1],
           'genre_Romance': [0], 'genre_Samurai': [0], 'genre_School': [0], 'genre_Sci-Fi': [0], 'genre_Seinen': [0],
           'genre_Shoujo': [0], 'genre_Shounen': [0], 'genre_Slice of Life': [0], 'genre_Space': [0],
           'genre_Sports': [0],
           'genre_Super Power': [0], 'genre_Supernatural': [0], 'genre_Thriller': [0], 'genre_Vampire': [0],
           'genre_Yaoi': [0],
           'type_Movie': [1], 'type_Music': [0], 'type_ONA': [0], 'type_OVA': [0],
           'type_Special': [0], 'type_TV': [1]}

exemple_df = pandas.DataFrame(exemple)
print("Тебе подойдет фильм : ", model.predict(exemple_df))

# Подберем наилучшие модели для данных
# Приступим к обучению нашей модели
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error

# 2. Делим на "вход" (Х) и выход (Y)
# X - то, на основе чего мы делаем прогноз
# Y - то, что мы прогнозируем
X = anime.drop(["rating", "name"], axis=1)  # Все кроме колонки Курс
y = anime["rating"]

# Делим на тестовую выборку и обучающую выборку
# Обучить модель - используем обучающую (тренировочная)  выборку (Учебник)
# Проверить модель - проверочная (тестовая) выборка = Экзамен
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# X_train, y_train = Учебник, на нем наша модель учится
# X_test = Экзамен
# y_test = Ответы к экзамену

# Классификация - задача выбора из конечного числа вариантов (выбор города из списка предложенных)
# Регрессия - задача предсказания конкретного числа (значение может быть любым)
from sklearn.linear_model import LinearRegression

model = LinearRegression()  # настройки в скобочках
model.fit(X_train, y_train)  # модель обучилась
prediction = model.predict(X_test)
# проверим на ошибки
mean_absolute_error(y_test, prediction)  # 1 параметр наши ответы, 2 параметр наше предсказание
print("LinearRegression MAE", mean_absolute_error(y_test, prediction))  # в среднем ошибка 39 копеек
# найдем максимальную ошибку
max_error(y_test, prediction)
print("LinearRegression MAX", max_error(y_test, prediction))  # самая худшая ошибка

from sklearn.neural_network import MLPRegressor

model = MLPRegressor(max_iter=15000)  # настройки в скобочках
model.fit(X_train, y_train)  # модель обучилась
prediction = model.predict(X_test)
# проверим на ошибки
mean_absolute_error(y_test, prediction)  # 1 параметр наши ответы, 2 параметр наше предсказание
print("MLPRegressor MAE", mean_absolute_error(y_test, prediction))  # в среднем ошибка 39 копеек
# найдем максимальную ошибку
max_error(y_test, prediction)
print("MLPRegressor MAX", max_error(y_test, prediction))  # самая худшая ошибка

# модель которая ищет примеры, которые примеры похоже друг на друга и делает выводы, что ответы тоже похожи
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(
    n_neighbors=5)  # параметр 1, сколько нужно найти похожих строк, чтобы ответы считать похожие
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("KNeighborsRegressor", "\nMAE", mean_absolute_error(y_test, prediction))
print("MAX", max_error(y_test, prediction))

# Сингулярное разложение матрицы
# Снижение размерности с помощью усеченного SVD (также известного как LSA).
# Этот преобразователь выполняет уменьшение линейной размерности с помощью усеченного сингулярного разложения (SVD).
# В отличие от PCA, этот оценщик не центрирует данные перед вычислением разложения по сингулярным значениям.
# Это означает, что он может эффективно работать с разреженными матрицами.
# Подробнее:
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html?highlight=svd#sklearn.decomposition.TruncatedSVD

from sklearn.decomposition import TruncatedSVD

# Создание входных данных X, выходных данных y
X = anime.drop("name", axis=1)
X = X.drop("index", axis=1)
y = anime["name"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
model = TruncatedSVD()
model.fit(X_train, y_train)

xtest = model.fit_transform(X_test, y_train)
print(xtest)
