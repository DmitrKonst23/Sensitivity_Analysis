import numpy as np


class FuncAnalysis:
    def __init__(self, func) -> None:
        '''
        Класс определяющий функционал для оптимальной генерации набора точек по заданной функции

        :params
            func: callable
                Функция, принимающая один параметр
        '''
        self.main_func = func
    

    def _need_split(self, points, epsilon):
        grads = (self.main_func(points[1:]) - self.main_func(points[:-1])) / (points[1:] - points[:-1])
        grads_diff = abs(np.arctan(grads[1:]) - np.arctan(grads[:-1]))

        need_split = grads_diff >= max(grads_diff) * epsilon

        return need_split


    def get_points(self, x_start, x_stop, num_points=10, epsilon=0.9):
        '''
        Функция, генерирующая оптимальные точки по функции func

        :params
            x_start: float
                Левая граница рассматриваемого отрезка
            x_stop: float
                Правая граница рассматриваемого отрезка
            num_points: int
                Минимальное количество точек, которое необходимо сгенерировать
                ВНИМАНИЕ количество точек может быть и больше указанного числа, но не меньше него
            epsilon: float
                Пороговое значение, при котором происходит разбиение отрезка

        :returns
            points: np.ndarray
                Массив вида (1, n) (n - количество сгенерированных точек)
        '''
        if x_stop <= x_start:
            raise(ValueError('x_stop должен быть больше x_start'))
        if num_points <= 2:
            raise(ValueError('num_points должно быть больше 2'))
        
        points = np.array([x_start, (x_start + x_stop) / 2, x_stop])

        while len(points) < num_points:
            need_split = self._need_split(points, epsilon)
            new_x = [points[0]]
            prev_splitted = False
            for i in range(1, len(points) - 1):
                if need_split[i - 1]:
                    if not prev_splitted:
                        new_x += [(points[i-1] + points[i]) / 2, points[i], (points[i] + points[i+1]) / 2]
                    else:
                        new_x += [points[i], (points[i] + points[i+1]) / 2]
                    prev_splitted = True
                else:
                    new_x += [points[i]]
                    prev_splitted = False
            new_x += [points[-1]]
            points = np.array(new_x)

        return points

