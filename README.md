# NIR_1_sem_magistracy

Научно-исследовательская работа 1 семестра магистратуры. Модели и методы отображения пространств эмоций различных размерностей. Обучение нейросети функциональному отображению 2-х мерного вектора в 7-ми мерный.

Расширенное содержание пояснительной записки к научно-исследовательской работе - https://disk.yandex.ru/i/2jytTI2zfN7Cbg.

## Инструкция по первичной установке:

1. Для запуска проекта необходима версия Python 3.9.1
2. cd ```<локальная папка репозитория NIR_1_sem_magistracy>```
3. Настроить виртуальное окружение + установить все необходимые пакеты (Для Linux):
```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```
* Если есть проблемы с:
    - установкой пакета pygraphviz
    - работой функцией ```plot_model``` из ```tensorflow.keras.utils``` (актуально для jupyter_notebooks/nn_train/NeuralNetwork_*.ipynb - папка 'architecture_models' с *.png архитектур нейромоделей)
```
sudo apt-get install graphviz graphviz-dev
pip install pygraphviz
```
4. Настроить переменные окружения:
```
export NIR_1_sem_magistracy="<локальная папка репозитория NIR_1_sem_magistracy>"
```
* Если используются модели (или другие папки/файлы) из NIR_3_sem_magistracy, дополнительно настроить:
```
export NIR_3_sem_magistracy="<локальная папка репозитория NIR_3_sem_magistracy>"
```
5. Скачать данные (Data) и модели (saved_models) с диска для папок:
    - Data - https://disk.yandex.ru/d/H1FE-7VO5wYVXQ
    - saved_models - https://disk.yandex.ru/d/_seqTpaSuFdO7Q
6. Запустить Jupyter notebook командой: ```jupyter-notebook```

## Инструкция по проверке успешного завершения установки:

Проверить, что полная установка проекта завершена успешно, можно "прокликав" следующие jupyter notebook'и:
1. Любой .ipynb из папки jupyter_notebooks/Data // кроме Data_4*.ipynb - он для development нужд
2. Любой .ipynb из папки jupyter_notebooks/nn_train // кроме NeuralNetwork_13_*.ipynb - он для development нужд
3. Любой .ipynb из папки jupyter_notebooks/Test // здесь без ограничений

* Если из каждого п. 1)-3) можно "прокликать" полностью .ipynb файл (не возникает ошибок импорта, ошибок с путями, ошибок в которых нет файлов и т.д.) то проект установлен правильно и им можно пользоваться.
Исключение: ошибки типа ```OperationalError: table <table_name> already exists```. Такие ошибки означают, что БД ```<table_name>``` уже была создана. В таком случае, её необходимо удалить и "прокликать" ячейку заново.
* Если же в каком-то .ipynb файле возникают вышеуказанные ошибки импорта, ошибки с путями, ошибки в которых нет файлов и т.д., то нужно понять, какие шаги установки были проделаны неверно/пропущены. И проделать их вновь.

## Инструкция к запуску:

После первичной установки и проверки её успешности, дальнейший запуск проекта включает шаги:
1. cd ```<локальная папка репозитория NIR_1_sem_magistracy>```
2. Активация виртуального окружения: ```source venv/bin/activate```
3. Настройка переменных окружения:
```
export NIR_1_sem_magistracy="<локальная папка репозитория NIR_1_sem_magistracy>"
```
* Если используются модели (или другие папки/файлы) из NIR_3_sem_magistracy, дополнительно настроить:
```
export NIR_3_sem_magistracy="<локальная папка репозитория NIR_3_sem_magistracy>"
```
4. Запуск Jupyter notebook командой: ```jupyter-notebook```
