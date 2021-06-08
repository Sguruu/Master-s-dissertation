import pandas
import matplotlib.pyplot as plt

print("Hello world")
# анализируем Датасет, для этого необходима библиотека pandas
# убираем лишние колонки из датафрейма

trips = pandas.read_excel("sample_data/trips_data.xlsx", index_col=0)
# вывод строк и колонок
print(trips.shape)
# посмотреть на три первых строчки
print(trips.head(3))
print(trips["salary"])
print(trips["salary"].describe())  # считаем разные значения
# построим гистрограмму
print(trips["salary"].hist())
trips["salary"].hist()
# test matplotlib
# x=[1,2,3,4,5,1]
# y=[0,2,1,2,3,4]
# plt.plot()
plt.show()
# фильтрация
# люди которые зарабатывают 250 тысяч рублей, воводится вся инфа
trips["salary"] == 250000
# выводим только верные условия
print(trips[trips["salary"] == 250000])
print(trips[trips["salary"] > 200000])
# выводим возраст
trips["age"].hist()
plt.show()
# посмотрим города
print("Смотрим колонку города")
print(trips["city"].value_counts())  # количество людей в городах
# функция plot() позволяет по данным построить график, где указывается тип графика в аргументах
trips["city"].value_counts().plot(kind="bar")  # столбчатая диаграмма
plt.show()
# что люд предпочитает
trips["vacation_preference"].value_counts().plot(kind="bar")
plt.show()
# транспорт
trips["transport_preference"].value_counts().plot(kind="bar")
plt.show()

# начнем предсказывать куда он политит, задача модели найти закономерности между "входными" (X) колнками и нашей
# "целью" (Y) (колонкой target)
# 1. необходимо подготовить наши данные для модели, постараемся убрать слова
# Неправильный путь: давать каждому слову порядковый номер
# Краснодар - 1
# Екатеренбург - 2
# Томск - 3
# начнет сравнивать города и сделает не верные выводы

# ОДИН ИЗ ПУТЕЙ
# Колонка с Городами => Много колонок, по одной на город (1 - если человек из этого города,
# 0 - если нет).
# то что позвлит это сделать
pandas.get_dummies(trips, columns=["city"])  # создает из строчек колонки сити, колонки
print(pandas.get_dummies(trips, columns=["city"]))
# создаем финальную переменную с правельными датами
final_df = pandas.get_dummies(trips, columns=["city", "vacation_preference", "transport_preference"])
# final_df.to_excel("finakDataSet.xlsx") # сохранить в exel таблицу
# X Входные данные, т.е. то, на основе чего мы делаем предсказание
# Y Выходные данные, т.е. то, что мы пытаемся предсказать
x = final_df.drop("target", axis=1)  # все кроме колонки таргет, axis, что это именно колонка таргет
y = final_df["target"]  # только target
print("Смотри y",y)
from sklearn.linear_model import LogisticRegression  # подключаем библиотеку и загружаем логистическую регрессию,
# подходит для задач классификации

model = LogisticRegression()  # Можно указывають настройки
model.fit(x, y)  # Машинное обучение

# попробуем что то предсказать, для этого необходимо на вход отправить строчку с данными
print("Вывод данных для работы")
print({col: [0] for col in x.columns})
exemple = {'salary': [45000], 'age': [27], 'family_members': [0], 'city_Екатеринбург': [0], 'city_Киев': [0],
           'city_Краснодар': [0], 'city_Минск': [0], 'city_Москва': [1], 'city_Новосибирск': [0], 'city_Омск': [0],
           'city_Петербург': [0], 'city_Томск': [0], 'city_Хабаровск': [0], 'city_Ярославль': [0],
           'vacation_preference_Архитектура': [0], 'vacation_preference_Ночные клубы': [0],
           'vacation_preference_Пляжный отдых': [1], 'vacation_preference_Шоппинг': [1],
           'transport_preference_Автомобиль': [0], 'transport_preference_Космический корабль': [0],
           'transport_preference_Морской транспорт': [0], 'transport_preference_Поезд': [0],
           'transport_preference_Самолет': [1]}
# создаем из нашей строчки табличку
exemple_df = pandas.DataFrame(exemple)
print(exemple_df)
exemple_df.to_excel("exemple_df.xlsx")

# предсказание модели
print(model.predict(exemple_df))  #
