import pandas
import matplotlib.pyplot as plt

student = pandas.read_csv("student-por.csv")
student.drop(["failures", "schoolsup", "famsup", "Dalc", "Walc"], axis=1, inplace=True)
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

from sklearn.model_selection import GridSearchCV
# еще один способ полуавтоматический, работает с моделями  для подборки

from sklearn.neural_network import MLPRegressor

model = MLPRegressor(max_iter=20000)
param_grid = {
    # "n_neighbors": range(1, 10)  # () диапазон, [1,2,3] вручную
   # "hidden_layer_sizes": [100, 500, 1000],  # количествой нейронов в скрытом слое
   # "momentum": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
"hidden_layer_sizes": [1000,1100,1200,1300,1400,1500]

}
gs = GridSearchCV(model, param_grid, scoring="neg_mean_absolute_error", cv=3)  # указываем модель с которой работать, у
# казываем параметры которые нужно переберать,
# второй параметр указываем функцию оценки модели
# подберем параметры для другой модели, CV - cross - validation (кросс - валидация),
# способ который позволяет точно определить что наша модель обучилась хорошо
# О - обучающая
# Т - тестовая
# [OOOOOOOOOOOOOTTTTT] пример модели вчера
# [OOOOOOOOOOOOOTTTTTTTTTT]
# перемешать данные
# и обучить с разный выборкой разбиения
# если обучим 5 моделей, сравним их ошибки то модель не очень хорошо работает .

print("GridSearchCV: ", gs.fit(X_train, y_train), " \n Другое: \n", gs.best_params_)
print("Лучшая ошибка: ", gs.best_score_)
gs.best_estimator_
