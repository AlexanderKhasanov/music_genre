import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_pivot_table(data, categories, column):
    '''
    Функция строит сводную таблицу признака по классам.
    В сводную таблицу попадают среднее, медиана, минимальное
    и максимальное значение признака. Также функиця строит столбчатую
    диаграмму по полученным медианам
    Функия принимает на вход:
    data - датасет
    categories - название колонки, содержащей категории
    column - название колонки с признаком
    Функция возвращает полученную сводную таблицу
    '''
    pv = (
        data
        .pivot_table(
            index=categories,
            values=column,
            aggfunc=['mean', 'median', 'min', 'max']
        )
        .reset_index().droplevel(1, axis=1)
        .set_index(categories)
    )
    pv.plot(
        kind='bar',
        y='median',
        title=f'Распределение медианы признака {column} по категорям'
    )
    return pv


def create_violinplot(data, column):
    '''
    Функиця строит диаграмму violinplot для указанного
    признака
    Функия принимает на вход:
    data - датасет
    column - название колонки с признаком
    '''
    _, ax = plt.subplots()
    ax.xaxis.grid(True)
    sns.violinplot(
        data=data,
        x=column, ax=ax
    ).set_title(f'Распределение признака {column}')


def create_category_violinplot(data, categories, column, common=False):
    '''
    Функиця строит диаграммы violinplot для каждой категории
    Функия принимает на вход:
    data - датасет
    categories - название колонки, содержащей категории
    column - название колонки с признаком
    common - будевый признак, отражающие строить ли дополнительный график,
    содержащий графики всех категорий вместе
    '''
    for group_name, group_data in data.groupby(categories)[column]:
        df = pd.DataFrame(group_data.values, columns=[group_name])
        _, ax = plt.subplots()
        sns.violinplot(
            x=df[group_name], ax=ax,
        ).set_title(f'Распределение признака {column} '
                    f'для категории {group_name}')
        ax.xaxis.grid(True)
    if common:
        _, ax_common = plt.subplots(figsize=(20, 30))
        ax_common.xaxis.grid(True)
        sns.violinplot(data=data, x=column, y=categories, ax=ax_common)


def create_countplot(data, column):
    '''
    Функиця строит столбчатую диаграмму для указанного признака
    Функия принимает на вход:
    data - датасет
    column - название колонки с признаком
    '''
    _, ax = plt.subplots()
    ax.xaxis.grid(True)
    sns.countplot(
        data=data,
        x=column, ax=ax
    ).set_title(f'Распределение признака {column}')


def create_category_countplot(data, categories, column):
    '''
    Функиця строит столбчатые диаграммы для каждой категории
    Функия принимает на вход:
    data - датасет
    categories - название колонки, содержащей категории
    column - название колонки с признаком
    '''
    for group_name, group_data in data.groupby(categories)[column]:
        df = pd.DataFrame(group_data.values, columns=[group_name])
        _, ax = plt.subplots()
        sns.countplot(
            x=df[group_name],
            ax=ax
        ).set_title(f'Распределение признака {column} '
                    f'для категории {group_name}')
        ax.xaxis.grid(True)
