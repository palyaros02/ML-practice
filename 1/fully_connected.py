# Импортируем основные модули pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Импортируем модули, связанные с компьютерным зрением
from torchvision import datasets
from torchvision.transforms import ToTensor

# Импортируем вспомогательные модули
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# В первой части работы вам предстоит создать и обучить модель на основе полносвязной нейронной сети, аналогичной той,
# что вы уже обучали в первом семестре вручную: сети с одним скрытым слоем и логистической функцией активации.
# В качестве обучающего набора будем использовать уже известный нам набор MNIST.

# Класс полносвязной нейронной сети.
# Необходимо реализовать полносвязную нейронную сеть с одним скрытым слоем
# с логистической функцией активации на скрытом слое и SoftMax на выходном слое.
# Схема сети: Линейный слой -> Логическая функция -> Линейный слой -> SoftMax
# Сеть должна классифицировать черно-белые картинки с цифрами (0-9) размера 28х28 пискселей из набора данных MNIST.
# Подумайте, какую функцию потерь будете использовать при обучении: от этого будет зависеть функция на последнем слое.
# Разберитесь в функциях потерь CrossEntropyLoss, NLLLoss, а также в функцих активации LogSoftmax и Softmax по документации.
# После успешной реализации требуемой сети, попробуйте поиграть с количеством нейронов, слоев, типами функций активации.
# Какой максимальной точности на тестовой выборке удалось достичь?
class FullyConnectedNet(nn.Module):

    def __init__(self):
        super().__init__()
        # ВАШ КОД ЗДЕСЬ

        # ===============

    # Метод для выполнения прямого распространения сигнала, необходимо заполнить
    def forward(self, x):
        # ВАШ КОД ЗДЕСЬ
        pass
        # ===============


# Задаем количество эпох (проходов по всей обучающей выборке) и размер пакета, можно варьировать
EPOCHS = 10
BATCH_SIZE = 256

# Загружаем данные из набора MNIST
train_data = datasets.MNIST(root='./data/train', train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root='./data/test', train=False, download=True, transform=ToTensor())

# DataLoader позволяет разбить выборку на пакеты заданного размера.
# Параметр shuffle отвечает за перемешивание данных в пакете
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# Если графический ускоритель поддерживает обучение на нем, будем использовать его,
# иначе обучать на процессоре.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Готовимся к обучению
model = FullyConnectedNet().to(device)  # создаем модель
optimizer = ...  # оптимизатор, нужно выбрать и настроить
loss_function = ...  # функция потерь, нужно выбрать
loss_history = list()  # список для хранения истории изменения функции стоимости

# Начинаем обучение
for epoch in range(EPOCHS):
    for i, (batch, labels) in enumerate(train_loader):  # разбиваем выборку на пакеты
        # Нужно реализовать один шаг градиентного спуска
        loss = torch.tensor(0, dtype=torch.float32)  # значение функции стоимости на пакете, нужно рассчитать
        # ВАШ КОД ЗДЕСЬ

        # ===============
        loss_history.append(loss.log().item())  # добавляется логарифм стоимости для большей наглядности
    print(f'Эпоха {epoch + 1} завершилась с функцией стоимости на последнем пакете = {loss.item()}')


# Выводим график функции стоимости
plt.title('Зависимость функции стоимости от номера шага градиентного спуска')
plt.xlabel('Номер шага')
plt.ylabel('Функция стоимости')
plt.plot(loss_history)
plt.show()

# Отключаем расчет вычислительного графа для экономии времени и памяти: нам не нужно считать градиенты при тестировании модели
with torch.no_grad():
    # Оцениваем качество модели
    train_data_loader = DataLoader(train_data, batch_size=len(train_data))
    train_features, train_targets = next(iter(train_data_loader))

    train_features = train_features.reshape(len(train_data), -1).to(device)
    train_model_predictions = torch.argmax(model(train_features), dim=1)
    print('Точность (accuracy) на обучающей выборке:', accuracy_score(train_data.targets, train_model_predictions.cpu()))

    test_data_loader = DataLoader(test_data, batch_size=len(test_data))
    test_features, test_targets = next(iter(test_data_loader))

    test_features = test_features.reshape(len(test_data), -1).to(device)
    test_model_predictions = torch.argmax(model(test_features), dim=1)
    print('Точность (accuracy) на тестовой выборке:', accuracy_score(test_data.targets, test_model_predictions.cpu()))
