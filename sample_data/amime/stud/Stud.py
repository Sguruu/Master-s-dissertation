import pandas
import matplotlib.pyplot as plt

student = pandas.read_csv("student-por.csv")
# вывод строк и колонок
print(student.shape)
print(student["school"].hist())
plt.show()

print(student["G3"].hist())
plt.show()

print(student["studytime"].plot())
plt.show()

# нарисуем два графика
plt.plot(student["studytime"], label="studytime")
plt.plot(student["G3"], label="G3")
plt.legend()  # Вывод подписей к линиям
plt.show()

# Обрабатываю данные
# failures - количество прошлых сбоев класса (числовое: n, если 1 <= n <3, иначе 4)
# Schoolup - дополнительная образовательная поддержка (двоичная: да или нет)
# famsup - семейная образовательная поддержка (двоичная: да или нет)
# Dalc - потребление алкоголя в течение рабочего дня (числовое значение: от 1 - очень низкий до 5 - очень высокий)
# Walc - потребление алкоголя в выходные дни (числовое значение: от 1 - очень низкий до 5 - очень высокий)

student.drop(["failures", "schoolsup", "famsup", "Dalc", "Walc"], axis=1, inplace=True)
print(student.columns)

# 2
# school - школа ученика (двоичная: GP - Габриэль Перейра или MS - Мусиньо да Силвейра)
# sex - пол студента (двоичные: 'F' - женский или 'M' - мужской)
# age - возраст студента (числовое значение: от 15 до 22)
# address - тип домашнего адреса студента (двоичный: 'U' - городской или 'R' - сельский)
# famsize - размер семьи (двоичное: 'LE3' - меньше или равно 3 или 'GT3' - больше 3)
# Pstatus - статус совместного проживания родителей (двоичный: «T» - проживают вместе или «A» - отдельно)
# Medu образование матери (число: 0 - нет, 1 - начальное образование (4-й класс), 2 - с 5-го по 9-й класс, 3 - среднее
# образование или 4 - высшее образование)
# Fedu - образование отца (число: 0 - нет, 1 - начальное образование (4-й класс), 2 - с 5-го по 9-й класс, 3 - среднее
# образование или 4 - высшее образование)
# Mjob - работа матери (номинальное: «учитель», «медицинское обслуживание», гражданские «службы»
# (например, административные или полицейские), «at_home» или «другое»)
# Fjob - работа отца (номинальное: «учитель», «здравоохранение», гражданские «услуги»
# (например, административные или полицейские), «at_home» или «другое»)
# причина - причина выбрать эту школу (номинальная: близко к «дому», «репутация» школы, «предпочтение по курсу» или
# «другое»)
# reason - причина выбрать эту школу (номинальная: близко к «дому», «репутация» школы, «предпочтение по курсу»
# или «другое»)
# guardian  - опекун ученика (именное: «мать», «отец» или «другой»)
# traveltime - время в пути от дома до школы (числовое значение: 1–1 час)
# studytime -  еженедельное учебное время (числовое значение: 1-10 часов)
# paid - дополнительные платные занятия по предмету курса (математика или португальский) (двоичный: да или нет)
# activities - внеклассные мероприятия (бинарные: да или нет)
# nursery - посещал детский сад (двоичный: да или нет)
# higher - хочет получить высшее образование (двоичное: да или нет)
# internet - доступ в Интернет дома (двоичный: да или нет)
# romantic - с романтическими отношениями (бинарные: да или нет)
# famrel - качество семейных отношений (числовое значение: от 1 - очень плохо до 5 - отлично)
# свободное время - свободное время после школы (числовое значение: от 1 - очень мало до 5 - очень высоко)
# goout - встреча с друзьями (числовые: от 1 - очень низкий, до 5 - очень высокий)
# health - текущее состояние здоровья (числовое значение: от 1 - очень плохо до 5 - очень хорошо)
# absences - количество пропусков в школе (числовое значение: от 0 до 93)
# G1 - первый класс (число: от 0 до 20)
# G2 - второй семестр (число: от 0 до 20)
# G3 - итоговая оценка (числовое значение: от 0 до 20, выходное задание)
student = pandas.get_dummies(student, columns=["school"])
student = pandas.get_dummies(student, columns=["sex"])
student = pandas.get_dummies(student, columns=["address"])
student = pandas.get_dummies(student, columns=["famsize"])
student = pandas.get_dummies(student, columns=["Pstatus"])
student = pandas.get_dummies(student, columns=["Mjob"])
student = pandas.get_dummies(student, columns=["Fjob"])
student = pandas.get_dummies(student, columns=["reason"])
student = pandas.get_dummies(student, columns=["guardian"])
student = pandas.get_dummies(student, columns=["paid"])
student = pandas.get_dummies(student, columns=["activities"])
student = pandas.get_dummies(student, columns=["nursery"])
student = pandas.get_dummies(student, columns=["higher"])
student = pandas.get_dummies(student, columns=["internet"])
student = pandas.get_dummies(student, columns=["romantic"])
print(student.columns)
print(student.shape)

# Приступим к обучению нашей модели
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error

# 2. Делим на "вход" (Х) и выход (Y)
# X - то, на основе чего мы делаем прогноз
# Y - то, что мы прогнозируем
X = student.drop("G3", axis=1)  # Все кроме колонки Курс
y = student["G3"]

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

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()  # настройки в скобочках
model.fit(X_train, y_train)  # модель обучилась
prediction = model.predict(X_test)
# проверим на ошибки
mean_absolute_error(y_test, prediction)  # 1 параметр наши ответы, 2 параметр наше предсказание
print("RandomForestRegressor MAE", mean_absolute_error(y_test, prediction))  # в среднем ошибка 39 копеек
# найдем максимальную ошибку
max_error(y_test, prediction)
print("RandomForestRegressor MAX", max_error(y_test, prediction))  # самая худшая ошибка

from sklearn.neural_network import MLPRegressor

model = MLPRegressor(hidden_layer_sizes=1200, max_iter=20000)  # настройки в скобочках
model.fit(X_train, y_train)  # модель обучилась
prediction = model.predict(X_test)
# проверим на ошибки
mean_absolute_error(y_test, prediction)  # 1 параметр наши ответы, 2 параметр наше предсказание
print("MLPRegressor MAE", mean_absolute_error(y_test, prediction))  # в среднем ошибка 39 копеек
# найдем максимальную ошибку
max_error(y_test, prediction)
print("MLPRegressor MAX", max_error(y_test, prediction))  # самая худшая ошибка

# нарисуем два графика
#plt.plot(prediction, label="Prediction")
plt.plot(y_test.iloc[0], label="Real")
plt.legend()  # Вывод подписей к линиям
plt.show()





# Теперь ставим задачу классификации, и будем определять по парамметрам следующие значения
# paid - дополнительные платные занятия по предмету курса (математика или португальский) (двоичный: да или нет)
X_paid = student.drop(["paid_yes", "paid_no"], axis=1)
y_paid = student["paid_yes"]
model.fit(X_paid, y_paid)

# попробуем что то предсказать, для этого необходимо на вход отправить строчку с данными
# print("Вывод данных для работы")
# print({col: [0] for col in X_paid})

# создаем из нашей строчки табличку
exemple_paid = {'age': [22], 'Medu': [4], 'Fedu': [4], 'traveltime': [1], 'studytime': [3], 'famrel': [2],
                'freetime': [2], 'goout': [3], 'health': [2], 'absences': [20], 'G1': [15], 'G2': [15], 'G3': [20],
                'school_GP': [0], 'school_MS': [1],
                'sex_F': [0], 'sex_M': [1],
                'address_R': [1], 'address_U': [0],
                'famsize_GT3': [0], 'famsize_LE3': [1],
                'Pstatus_A': [1], 'Pstatus_T': [0],
                'Mjob_at_home': [0], 'Mjob_health': [0], 'Mjob_other': [0], 'Mjob_services': [1], 'Mjob_teacher': [0],
                'Fjob_at_home': [0], 'Fjob_health': [0], 'Fjob_other': [0], 'Fjob_services': [1], 'Fjob_teacher': [0],
                'reason_course': [0], 'reason_home': [1], 'reason_other': [0], 'reason_reputation': [0],
                'guardian_father': [0], 'guardian_mother': [1], 'guardian_other': [0],
                'activities_no': [1], 'activities_yes': [0],
                'nursery_no': [1], 'nursery_yes': [0],
                'higher_no': [1], 'higher_yes': [0],
                'internet_no': [1], 'internet_yes': [0],
                'romantic_no': [0], 'romantic_yes': [1]}
exemple_paid = pandas.DataFrame(exemple_paid)

print("Дополнительные платные занятия по прдемету курса : ", model.predict(exemple_paid))

# Medu образование матери (число: 0 - нет, 1 - начальное образование (4-й класс), 2 - с 5-го по 9-й класс, 3 - среднее
X_medu = student.drop("Medu", axis=1)
y_medu = student["Medu"]
model.fit(X_medu, y_medu)
# print({col: [0] for col in X_medu})
exemple_medu = {'age': [17], 'Fedu': [3], 'traveltime': [0.20], 'studytime': [6], 'famrel': [2],
                'freetime': [2], 'goout': [1], 'health': [4], 'absences': [50],
                'G1': [15], 'G2': [10], 'G3': [5],
                'school_GP': [0], 'school_MS': [1],
                'sex_F': [1], 'sex_M': [0],
                'address_R': [1], 'address_U': [0],
                'famsize_GT3': [0], 'famsize_LE3': [1],
                'Pstatus_A': [0], 'Pstatus_T': [1],
                'Mjob_at_home': [0], 'Mjob_health': [0], 'Mjob_other': [0], 'Mjob_services': [1], 'Mjob_teacher': [0],
                'Fjob_at_home': [0], 'Fjob_health': [0], 'Fjob_other': [0], 'Fjob_services': [1], 'Fjob_teacher': [0],
                'reason_course': [0], 'reason_home': [1], 'reason_other': [0], 'reason_reputation': [0],
                'guardian_father': [0], 'guardian_mother': [1], 'guardian_other': [0],
                'paid_no': [0], 'paid_yes': [1],
                'activities_no': [0], 'activities_yes': [1],
                'nursery_no': [0], 'nursery_yes': [1],
                'higher_no': [0], 'higher_yes': [1],
                'internet_no': [0], 'internet_yes': [1],
                'romantic_no': [0], 'romantic_yes': [1]}
exemple_medu = pandas.DataFrame(exemple_medu)
print("Образование матери : ", model.predict(exemple_medu))


# Fedu образование отца (число: 0 - нет, 1 - начальное образование (4-й класс), 2 - с 5-го по 9-й класс, 3 - среднее
X_medu = student.drop("Fedu", axis=1)
y_medu = student["Fedu"]
model.fit(X_medu, y_medu)
# print({col: [0] for col in X_medu})
exemple_medu = {'age': [20], 'Medu': [4], 'traveltime': [0.15], 'studytime': [7], 'famrel': [5],
                'freetime': [7], 'goout': [2], 'health': [4], 'absences': [30],
                'G1': [14], 'G2': [7], 'G3': [2],
                'school_GP': [0], 'school_MS': [1],
                'sex_F': [0], 'sex_M': [1],
                'address_R': [1], 'address_U': [0],
                'famsize_GT3': [1], 'famsize_LE3': [0],
                'Pstatus_A': [0], 'Pstatus_T': [1],
                'Mjob_at_home': [0], 'Mjob_health': [0], 'Mjob_other': [1], 'Mjob_services': [0], 'Mjob_teacher': [0],
                'Fjob_at_home': [0], 'Fjob_health': [0], 'Fjob_other': [0], 'Fjob_services': [1], 'Fjob_teacher': [0],
                'reason_course': [0], 'reason_home': [0], 'reason_other': [0], 'reason_reputation': [1],
                'guardian_father': [0], 'guardian_mother': [1], 'guardian_other': [0],
                'paid_no': [1], 'paid_yes': [0],
                'activities_no': [0], 'activities_yes': [1],
                'nursery_no': [0], 'nursery_yes': [1],
                'higher_no': [1], 'higher_yes': [0],
                'internet_no': [0], 'internet_yes': [1],
                'romantic_no': [1], 'romantic_yes': [0]}
exemple_medu = pandas.DataFrame(exemple_medu)
print("Образование отца : ", model.predict(exemple_medu))


# G3 - итоговая оценка (числовое значение: от 0 до 20, выходное задание)
X_g3 = student.drop("G3", axis=1)
y_g3 = student["G3"]
model.fit(X_g3, y_g3)

# print({col: [0] for col in X_g3})
exemple_g3 = {'age': [22], 'Medu': [4], 'Fedu': [4], 'traveltime': [1], 'studytime': [3], 'famrel': [2],
              'freetime': [2], 'goout': [3], 'health': [2], 'absences': [20],
              'G1': [15], 'G2': [15],
              'school_GP': [0], 'school_MS': [1],
              'sex_F': [0], 'sex_M': [1],
              'address_R': [1], 'address_U': [0],
              'famsize_GT3': [0], 'famsize_LE3': [1],
              'Pstatus_A': [1], 'Pstatus_T': [0],
              'Mjob_at_home': [0], 'Mjob_health': [0], 'Mjob_other': [0], 'Mjob_services': [1], 'Mjob_teacher': [0],
              'Fjob_at_home': [0], 'Fjob_health': [0], 'Fjob_other': [0], 'Fjob_services': [1], 'Fjob_teacher': [0],
              'reason_course': [0], 'reason_home': [1], 'reason_other': [0], 'reason_reputation': [0],
              'guardian_father': [0], 'guardian_mother': [1], 'guardian_other': [0],
              'paid_no': [1], 'paid_yes': [0],
              'activities_no': [1], 'activities_yes': [0],
              'nursery_no': [1], 'nursery_yes': [0],
              'higher_no': [1], 'higher_yes': [0],
              'internet_no': [0], 'internet_yes': [1],
              'romantic_no': [0], 'romantic_yes': [1]}
exemple_g3 = pandas.DataFrame(exemple_g3)
print("Высшая оценка : ", model.predict(exemple_g3))


