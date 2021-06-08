import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# День 2
# Предсказание курса доллара к нефти
# Данные от центробанка: http://cbr.ru/currency_base/dynamics/
# Выбор доллар США, период: 10.01.2017 - 20.05.2021
# Данные по нефти: https://www.eia.gov/dnav/pet/hist/rbrteD.htm

# читаем файл
dollarRate = pandas.read_excel("sample_data/dollar_rate_data.xlsx")
print(dollarRate)

# dollarRate = dollarRate.loc['2017-01-01':'2021-05-19',['data']]

# нарисуем график
dollarRate["curs"].plot()
plt.show()

# читаем файл с нефтью
oilCourse = pandas.read_excel("sample_data/oil_course_data.xls", sheet_name=1
                              , skiprows=2, names=["date", "oil_price"])  # указываем чтение со второго листа,
# пропускаем первые две строки, переименовываем колонки
print(oilCourse)
oilCourse["oil_price"].plot()
plt.show()

# попробуем объеденить таблицы
print(dollarRate.head(3))
print(oilCourse.head(3))

# пишу сначала главную совбпадающую колонку, добавить
dollarRate.set_index("data").join(oilCourse.set_index("date"))  # объединение двух таблиц по дате
oli_dollar_rate = dollarRate.set_index("data").join(oilCourse.set_index("date"))
print(oli_dollar_rate)

# выкидывание лишних колонок
oli_dollar_rate.drop(["nominal", "cdx"], axis=1, inplace=True)  # имя колонок, указываем что имеенно хотим удалить
# колонки, применить к этой таблице

print(oli_dollar_rate)

# уберем пустоты, использую одной из стратегий (берем предыдущее значение)
oli_dollar_rate.fillna(method="ffill", inplace=True)

print(oli_dollar_rate)

oli_dollar_rate.reset_index(inplace=True)  # Сбрасываем индекс

# нарисуем два графика
plt.plot(oli_dollar_rate["data"], oli_dollar_rate["curs"], label="USD/RUB Rate")  # Рисуем линию на курс доллара
plt.plot(oli_dollar_rate["data"], oli_dollar_rate["oil_price"], label="Oil Price")  # Рисусуем линию цены на нефть
plt.legend()  # Вывод подписей к линиям
plt.show()

# добавим новую колонку разбив колонку дата
oli_dollar_rate["year"] = oli_dollar_rate["data"].dt.year  # Создаем колонку Год из поля Дата
oli_dollar_rate["month"] = oli_dollar_rate["data"].dt.month
oli_dollar_rate["weekday"] = oli_dollar_rate["data"].dt.weekday
print(oli_dollar_rate)

# добавим колонки в зависимости от статистики за предыдущие дни
past_day = 7  # Возьмем статистику за 7 дней в прошлом

# напишим цикл для 7 дней
for day in range(past_day):
    d = day + 1
    oli_dollar_rate[f"usd_back_{d}d"] = oli_dollar_rate["curs"].shift(d)  # сдвинуть на d день
    oli_dollar_rate[f"oli_back_{d}d"] = oli_dollar_rate["oil_price"].shift(d)

print(oli_dollar_rate)  # тут можно заметить, что не все данные можно взять с прошлого
# oli_dollar_rate.trail(8) посмотреть 8 эл с конца

# добавим еще колонку, среднее значение за 1 день
oli_dollar_rate["usd_week"] = oli_dollar_rate["curs"].shift(
    1).rolling(window=7).median()  # за 1 деень, скользящее окно 7 дней,
# математическая операция, медиана
oli_dollar_rate["oil_week"] = oli_dollar_rate["oil_price"].shift(1).rolling(window=7).median()
print(oli_dollar_rate)
oli_dollar_rate.shape  # посмотреть форму

# избавляемся от месяцев (дискретных значений) и лет по отдельным колонкам
oli_dollar_rate = pandas.get_dummies(oli_dollar_rate,
                                     columns=["year", "month", "weekday"])  # преобразуем колонки "даты"
# в бинарные (0/1) колонки
oli_dollar_rate.to_excel("testDataSet.xlsx")

# Приступим к обучению нашей модели
# 1. Уберем колонки которые использовать нельзя
oli_dollar_rate.drop(["data", "oil_price"], axis=1, inplace=True)
oli_dollar_rate.dropna(inplace=True)  # отказываемся от первых 7 строк, функция выкидывает все строчки где есть пустота
print(oli_dollar_rate)
# 2. Делим на "вход" (Х) и выход (Y)
# X - то, на основе чего мы делаем прогноз
# Y - то, что мы прогнозируем
X = oli_dollar_rate.drop("curs", axis=1)  # Все кроме колонки Курс
y = oli_dollar_rate["curs"]

# Делим на тестовую выборку и обучающую выборку
# Обучить модель - используем обучающую (тренировочная)  выборку (Учебник)
# Проверить модель - проверочная (тестовая) выборка = Экзамен
from sklearn.model_selection import train_test_split

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
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, prediction)  # 1 параметр наши ответы, 2 параметр наше предсказание
print("MAE", mean_absolute_error(y_test, prediction))  # в среднем ошибка 39 копеек
# найдем максимальную ошибку

from sklearn.metrics import max_error

max_error(y_test, prediction)
print("MAX", max_error(y_test, prediction))  # самая худшая ошибка

from sklearn.ensemble import RandomForestRegressor

# обучении на основе деревьев решений, параметр random_state отвечает за точку обучения, рандома
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("MAE", mean_absolute_error(y_test, prediction))
print("MAX", max_error(y_test, prediction))  # самая худшая ошибка

# попробуем улучшить качество и уменьшит ошибки
# n_estimators - количество деревьер решений
# criterion - на сколько качественные использует метрики
model = RandomForestRegressor(random_state=42, n_estimators=10, criterion="mae", min_samples_split=6)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("MAE", mean_absolute_error(y_test, prediction))
print("MAX", max_error(y_test, prediction))

# модель которая ищет примеры, которые примеры похоже друг на друга и делает выводы, что ответы тоже похожи
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(
    n_neighbors=5)  # параметр 1, сколько нужно найти похожих строк, чтобы ответы считать похожие
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("KNeighborsRegressor", "\nMAE", mean_absolute_error(y_test, prediction))
print("MAX", max_error(y_test, prediction))

# модель основанная на нейронной сети
# hidden_layer_sizes (1,100)количество слоев нейронной сети
# max_iter (200)
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(max_iter=2000, hidden_layer_sizes=(200, 150, 350))  #
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("MLPRegressor", "\nMAE", mean_absolute_error(y_test, prediction))
print("MAX", max_error(y_test, prediction))

from sklearn.model_selection import GridSearchCV

# еще один способ полуавтоматический, работает с моделями  для подборки
model = KNeighborsRegressor()
param_grid = {
    "n_neighbors": range(1, 10)  # () диапазон, [1,2,3] вручную
}
gs = GridSearchCV(model, param_grid, scoring="neg_mean_absolute_error")  # указываем модель с которой работать, у
# казываем параметры которые нужно переберать,
# второй параметр указываем функцию оценки модели

print("GridSearchCV: ", gs.fit(X_train, y_train), " \n Другое: \n", gs.best_params_)
print("Лучшая ошибка: ", gs.best_score_)
gs.best_estimator_  # наши настройки обученной модели которую можно будет использовать

# подберем параметры для другой модели, CV - cross - validation (кросс - валидация),
# способ который позволяет точно определить что наша модель обучилась хорошо
# О - обучающая
# Т - тестовая
# [OOOOOOOOOOOOOTTTTT] пример модели вчера
# [OOOOOOOOOOOOOTTTTTTTTTT]
# перемешать данные
# и обучить с разный выборкой разбиения
# если обучим 5 моделей, сравним их ошибки то модель не очень хорошо работает .
model = RandomForestRegressor()
param_grid = {
    "n_estimators": [5, 1000],
    "criterion": ["mse", "mae"],
    "max_depth": [3, 15]
}
gs = GridSearchCV(model, param_grid, scoring="neg_mean_absolute_error",
                  cv=2)  # параметр cv и отвечает за кросс валидацию

print("GridSearchCV: ", gs.fit(X_train, y_train), " \n Другое: \n", gs.best_params_)
print("Лучшая ошибка: ", gs.best_score_)
gs.best_estimator_

# испытание обученной модели
model = gs.best_estimator_
prediction = model.predict(X_test)
print("MLPRegressor", "\nMAE", mean_absolute_error(y_test, prediction))
print("MAX", max_error(y_test, prediction))
print("model: ", model)



