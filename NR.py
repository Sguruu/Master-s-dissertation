# Программа создана для решения задач Классификации, выбрать из конечного варианта конкретный  Anime
# Для работы программы необходимо установить следующие библиотеки :
# 1. pandas - для работы с дата сетами
# 2. matplotlib - для красивых графиков
# 3. sklearn - для использования моделей для обучения.

import pandas
import matplotlib.pyplot as plt

data_anime = pandas.read_excel("sample_data/amime/anime.xlsx")
print("Вывод строк и колонок: ", data_anime.shape, data_anime.shape[1])
print("Посмотрим на три первых строчки \n", data_anime.head(3))
print("Построим гистограмму по рейтингу ", data_anime["rating"].hist())
plt.show()
plt.hist(data_anime["rating"], label="rating")
plt.legend()
plt.show()

test_dt = pandas.read_excel("sample_data/amime/animeTest.xlsx")

# Начнем подготавливать наши данные
# ----------------Test----------------
# data_set = pandas.get_dummies(test_dt,  columns=["Title"],)
# data_set = test_dt['Title'].str.split(',', expand=True)  # делит на колонки после запятой но именует столбцы по своему
# data_set = test_dt['Title'].str.split(',', expand=True).rename(columns=lambda x: "string" + str(x + 1)]]])  # + имена
data_set = test_dt['Title'].str.split(',', expand=True).rename(columns=lambda x: "string" + str(x + 1))  # + имена
data_set.to_excel("test_anime1.xlsx")

for i in range(3):
    print(i)
    data_1_3 = data_set[f"string{i + 1}"]
    data_1_3.to_excel(f"data_{i + 1}_{i + 1}_3.xlsx")
    data_1_3 = pandas.get_dummies(data_1_3)
    data_1_3.to_excel(f"data_{i + 1}_3.xlsx")

dm = pandas.read_excel("data_2_3.xlsx")
data_1_3 = pandas.concat([data_1_3, dm], axis=1)
data_1_3.to_excel("dataMerg.xlsx")
# ----------------TestEnd----------------

# 1. Создаем новую DF содержащий колонку которую нужно будет раскидать
# kol_dt = data_anime["genre"]
# kol_dt.to_excel("test_anime3.xlsx")
# 2. Разбиваем этот файл на колонки после запятой вручную черех exel
kol_dt = pandas.read_excel("sample_data/amime/animeColRazbit.xlsx")
kol_dt = pandas.get_dummies(kol_dt, )
kol_dt.to_excel("test_anime4.xlsx")  # жрет много времени комментировать
