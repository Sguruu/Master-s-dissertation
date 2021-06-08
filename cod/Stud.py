import pandas
import matplotlib.pyplot as plt
student = pandas.read_csv("student-por.csv")
print(student.shape)
print(student["school"].hist())
plt.show()
print(student["G3"].hist())
plt.show()
print(student["studytime"].plot())
plt.show()
plt.plot(student["studytime"], label="studytime")
plt.plot(student["G3"], label="G3")
plt.legend()
plt.show()
student.drop(["failures", "schoolsup", "famsup", "Dalc", "Walc"], axis=1, inplace=True)
print(student.columns)
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error

X = student.drop("G3", axis=1)
y = student["G3"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
mean_absolute_error(y_test, prediction)
print("LinearRegression MAE", mean_absolute_error(y_test, prediction))
max_error(y_test, prediction)
print("LinearRegression MAX", max_error(y_test, prediction))

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
mean_absolute_error(y_test, prediction)
print("RandomForestRegressor MAE", mean_absolute_error(y_test, prediction))
max_error(y_test, prediction)
print("RandomForestRegressor MAX", max_error(y_test, prediction))

from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=1200, max_iter=20000)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
mean_absolute_error(y_test, prediction)
print("MLPRegressor MAE", mean_absolute_error(y_test, prediction))
max_error(y_test, prediction)
print("MLPRegressor MAX", max_error(y_test, prediction))
X_paid = student.drop(["paid_yes", "paid_no"], axis=1)
y_paid = student["paid_yes"]
model.fit(X_paid, y_paid)
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
exemple_medu = {'age': [22], 'Fedu': [4], 'traveltime': [1], 'studytime': [3], 'famrel': [2],
                'freetime': [2], 'goout': [3], 'health': [2], 'absences': [20],
                'G1': [15], 'G2': [15], 'G3': [20],
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
                'internet_no': [1], 'internet_yes': [0],
                'romantic_no': [0], 'romantic_yes': [1]}
exemple_medu = pandas.DataFrame(exemple_medu)
print("Образование матери : ", model.predict(exemple_medu))
X_g3 = student.drop("G3", axis=1)
y_g3 = student["G3"]
model.fit(X_g3, y_g3)
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
