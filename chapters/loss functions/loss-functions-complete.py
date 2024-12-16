import numpy as np
import matplotlib.pyplot as plt

def solve_mse():
    """Решение задачи с MSE для прогнозирования цен на недвижимость"""
    print("\n=== Задача 1: MSE для прогнозирования цен на недвижимость ===")
    
    # Данные
    actual_prices = np.array([5.2, 12.8, 18.5, 22.1, 7.4])
    predicted_prices = np.array([6.1, 12.5, 17.8, 23.0, 8.5])
    
    # Решение
    errors = actual_prices - predicted_prices
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    relative_errors = np.abs(errors) / actual_prices * 100
    
    print("Решение:")
    print(f"MSE = {mse:.2f} млн руб²")
    print(f"RMSE = {rmse:.2f} млн руб")
    print("\nОтносительные ошибки по объектам:")
    for i, error in enumerate(relative_errors):
        print(f"Объект {i+1}: {error:.1f}%")
    
    # Визуализация
    plt.figure(figsize=(12, 5))
    
    # График слева: сравнение цен
    plt.subplot(1, 2, 1)
    x = range(len(actual_prices))
    width = 0.35
    plt.bar(x, actual_prices, width, label='Реальные цены', color='skyblue')
    plt.bar([i + width for i in x], predicted_prices, width, label='Предсказанные цены', color='lightcoral')
    plt.xlabel('Объект')
    plt.ylabel('Цена (млн руб.)')
    plt.title('Сравнение цен на недвижимость')
    plt.legend()
    
    # График справа: квадраты ошибок
    plt.subplot(1, 2, 2)
    squared_errors = errors**2
    plt.bar(x, squared_errors, color='red', alpha=0.6)
    plt.axhline(y=mse, color='black', linestyle='--', label=f'MSE = {mse:.2f}')
    plt.xlabel('Объект')
    plt.ylabel('Квадрат ошибки')
    plt.title('Квадраты ошибок')
    plt.legend()
    
    plt.tight_layout()
    
    plt.savefig('imgs/mse.png')
    plt.show()

def solve_mae():
    """Решение задачи с MAE для прогнозирования потребления электроэнергии"""
    print("\n=== Задача 2: MAE для прогнозирования потребления электроэнергии ===")
    
    # Данные
    actual_consumption = np.array([245, 256, 278, 235, 290])
    predicted_consumption = np.array([250, 262, 275, 230, 285])
    
    # Решение
    errors = actual_consumption - predicted_consumption
    mae = np.mean(np.abs(errors))
    
    print("Решение:")
    print(f"MAE = {mae:.2f} МВт·ч")
    print("\nАбсолютные ошибки по дням:")
    for i, error in enumerate(np.abs(errors)):
        print(f"День {i+1}: {error:.1f} МВт·ч")
    
    # Визуализация
    plt.figure(figsize=(12, 5))
    
    # График слева: линии потребления
    plt.subplot(1, 2, 1)
    days = range(1, len(actual_consumption) + 1)
    plt.plot(days, actual_consumption, 'o-', label='Реальное потребление', color='blue')
    plt.plot(days, predicted_consumption, 'o--', label='Предсказанное потребление', color='red')
    plt.fill_between(days, actual_consumption, predicted_consumption, alpha=0.2, color='gray')
    plt.xlabel('День')
    plt.ylabel('Потребление (МВт·ч)')
    plt.title('Сравнение потребления электроэнергии')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График справа: абсолютные ошибки
    plt.subplot(1, 2, 2)
    abs_errors = np.abs(errors)
    plt.bar(days, abs_errors, color='red', alpha=0.6)
    plt.axhline(y=mae, color='black', linestyle='--', label=f'MAE = {mae:.2f}')
    plt.xlabel('День')
    plt.ylabel('Абсолютная ошибка (МВт·ч)')
    plt.title('Абсолютные ошибки')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    
    plt.savefig('imgs/mae.png')
    plt.show()

def solve_mape():
    """Решение задачи с MAPE для прогнозирования продаж"""
    print("\n=== Задача 3: MAPE для прогнозирования продаж ===")
    
    # Данные
    actual_gum = np.array([1200, 1150, 1300, 1180, 1250])
    actual_tv = np.array([5, 8, 6, 7, 4])
    predicted_gum = np.array([1180, 1200, 1250, 1150, 1300])
    predicted_tv = np.array([6, 7, 5, 8, 5])
    
    # Решение
    mape_gum = np.mean(np.abs((actual_gum - predicted_gum) / actual_gum)) * 100
    mape_tv = np.mean(np.abs((actual_tv - predicted_tv) / actual_tv)) * 100
    
    print("Решение:")
    print(f"MAPE для жевательной резинки: {mape_gum:.1f}%")
    print(f"MAPE для телевизоров: {mape_tv:.1f}%")
    
    # Визуализация
    plt.figure(figsize=(15, 5))
    
    # График для жевательной резинки
    plt.subplot(1, 3, 1)
    days = range(1, len(actual_gum) + 1)
    plt.plot(days, actual_gum, 'o-', label='Реальные продажи', color='blue')
    plt.plot(days, predicted_gum, 'o--', label='Прогноз', color='red')
    plt.title('Продажи жевательной резинки')
    plt.xlabel('День')
    plt.ylabel('Количество (шт.)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График для телевизоров
    plt.subplot(1, 3, 2)
    plt.plot(days, actual_tv, 'o-', label='Реальные продажи', color='blue')
    plt.plot(days, predicted_tv, 'o--', label='Прогноз', color='red')
    plt.title('Продажи телевизоров')
    plt.xlabel('День')
    plt.ylabel('Количество (шт.)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График сравнения MAPE
    plt.subplot(1, 3, 3)
    categories = ['Жев. резинка', 'Телевизоры']
    mape_values = [mape_gum, mape_tv]
    plt.bar(categories, mape_values, color=['skyblue', 'lightcoral'])
    plt.title('Сравнение MAPE')
    plt.ylabel('MAPE (%)')
    
    plt.tight_layout()
    
    plt.savefig('imgs/mape.png')
    plt.show()

def solve_huber():
    """Решение задачи с Huber Loss для времени доставки"""
    print("\n=== Задача 4: Huber Loss для времени доставки ===")
    
    # Данные
    actual_times = np.array([25, 30, 28, 85, 27])
    predicted_times = np.array([28, 32, 30, 35, 29])
    delta = 15  # параметр Huber Loss
    
    # Решение
    errors = actual_times - predicted_times
    huber_losses = np.array([
        0.5 * err**2 if abs(err) <= delta else
        delta * abs(err) - 0.5 * delta**2
        for err in errors
    ])
    mean_huber_loss = np.mean(huber_losses)
    
    print("Решение:")
    print(f"Средняя Huber Loss (δ={delta}): {mean_huber_loss:.2f}")
    print("\nПотери по доставкам:")
    for i, loss in enumerate(huber_losses):
        print(f"Доставка {i+1}: {loss:.2f}")
    
    # Визуализация
    plt.figure(figsize=(12, 5))
    
    # График слева: времена доставки
    plt.subplot(1, 2, 1)
    x = range(1, len(actual_times) + 1)
    plt.plot(x, actual_times, 'o-', label='Реальное время', color='blue')
    plt.plot(x, predicted_times, 'o--', label='Предсказанное время', color='red')
    plt.xlabel('Доставка')
    plt.ylabel('Время (мин)')
    plt.title('Времена доставки')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График справа: сравнение потерь
    plt.subplot(1, 2, 2)
    mse_losses = errors**2
    mae_losses = np.abs(errors)
    
    x_pos = np.arange(len(errors))
    width = 0.25
    
    plt.bar(x_pos - width, mse_losses, width, label='MSE', color='skyblue')
    plt.bar(x_pos, mae_losses, width, label='MAE', color='lightgreen')
    plt.bar(x_pos + width, huber_losses, width, label='Huber', color='lightcoral')
    
    plt.xlabel('Доставка')
    plt.ylabel('Значение функции потерь')
    plt.title('Сравнение функций потерь')
    plt.yscale("log")
    plt.grid()
    plt.legend()
    
    plt.tight_layout()
    
    plt.savefig('imgs/huber.png')
    plt.show()

def solve_fair():
    """Решение задачи с Fair Loss для просмотров контента"""
    print("\n=== Задача 5: Fair Loss для просмотров контента ===")
    
    # Данные
    actual_views = np.array([150, 165, 950, 180, 175])
    predicted_views = np.array([160, 170, 200, 175, 180])
    c = 100  # параметр Fair Loss
    
    # Решение
    errors = np.abs(actual_views - predicted_views)
    fair_losses = c**2 * (errors/c - np.log(1 + errors/c))
    mean_fair_loss = np.mean(fair_losses)
    
    print("Решение:")
    print(f"Средняя Fair Loss (c={c}): {mean_fair_loss:.2f}")
    print("\nОтклонения по эпизодам:")
    for i, (act, pred) in enumerate(zip(actual_views, predicted_views)):
        print(f"Эпизод {i+1}: отклонение {abs(act - pred)} тыс. просмотров")
    
    # Визуализация
    plt.figure(figsize=(12, 5))
    
    # График слева: просмотры
    plt.subplot(1, 2, 1)
    x = range(1, len(actual_views) + 1)
    plt.plot(x, actual_views, 'o-', label='Реальные просмотры', color='blue')
    plt.plot(x, predicted_views, 'o--', label='Предсказанные просмотры', color='red')
    plt.xlabel('Эпизод')
    plt.ylabel('Просмотры (тыс.)')
    plt.title('Просмотры эпизодов')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График справа: Fair Loss
    plt.subplot(1, 2, 2)
    plt.bar(x, fair_losses, color='lightcoral', alpha=0.6)
    plt.axhline(y=mean_fair_loss, color='black', linestyle='--', 
                label=f'Средняя Fair Loss = {mean_fair_loss:.2f}')
    plt.xlabel('Эпизод')
    plt.ylabel('Fair Loss')
    plt.title('Значения Fair Loss')
    plt.yscale("log")
    plt.legend()
    
    plt.tight_layout()
    
    
    plt.savefig('imgs/fair.png')
    plt.show()

def solve_tukey():
    """Решение задачи с Tukey Loss для данных с датчиков"""
    print("\n=== Задача 6: Tukey Loss для данных с датчиков ===")
    # Данные
    actual_temp = np.array([85, 82, 999, 88, 86])  
    predicted_temp = np.array([84, 83, 85, 87, 85])
    c = 10  # параметр Tukey Loss
    
    # Вычисляем потери без нормализации
    errors = actual_temp - predicted_temp
    
    # MSE
    mse_losses = errors**2
    
    # MAE
    mae_losses = np.abs(errors)
    
    # Tukey Loss
    tukey_losses = np.array([
        (c**2/6) * (1 - (1 - min(1, abs(err/c))**2)**3) 
        for err in errors
    ])
    
    # Визуализация
    plt.figure(figsize=(15, 5))
    
    # График 1: Исходные данные
    plt.subplot(1, 3, 1)
    x = range(1, len(actual_temp) + 1)
    plt.plot(x, actual_temp, 'o-', label='Показания датчика', color='blue')
    plt.plot(x, predicted_temp, 'o--', label='Предсказанные значения', color='red')
    plt.xlabel('Измерение')
    plt.ylabel('Температура (°C)')
    plt.title('Исходные данные')
    plt.legend()
    
    # График 2: Абсолютные значения функций потерь
    plt.subplot(1, 3, 2)
    x_pos = np.arange(len(errors))
    width = 0.25
    
    plt.bar(x_pos - width, mse_losses, width, label='MSE')
    plt.bar(x_pos, mae_losses, width, label='MAE')
    plt.bar(x_pos + width, tukey_losses, width, label='Tukey')
    plt.xlabel('Измерение')
    plt.ylabel('Значение функции потерь')
    plt.title('Абсолютные значения потерь\n(логарифмическая шкала)')
    plt.yscale('log')
    plt.legend()
    
    # График 3: Значения функций потерь без выброса
    plt.subplot(1, 3, 3)
    mask = errors != errors[2]  # исключаем выброс
    x_pos = np.arange(sum(mask))
    
    plt.bar(x_pos - width, mse_losses[mask], width, label='MSE')
    plt.bar(x_pos, mae_losses[mask], width, label='MAE')
    plt.bar(x_pos + width, tukey_losses[mask], width, label='Tukey')
    plt.xlabel('Измерение (без выброса)')
    plt.ylabel('Значение функции потерь')
    plt.title('Значения потерь без выброса')
    plt.legend()
    
    plt.tight_layout()
    
    # Вывод значений
    print("\nЗначения функций потерь:")
    for i in range(len(errors)):
        print(f"\nИзмерение {i+1}:")
        print(f"MSE: {mse_losses[i]:.2f}")
        print(f"MAE: {mae_losses[i]:.2f}")
        print(f"Tukey: {tukey_losses[i]:.2f}")
        
    plt.savefig('imgs/tukey.png')
    plt.show()
    
def solve_log_cosh():
    """Решение задачи с Log-Cosh Loss для прогнозирования курса акций"""
    print("\n=== Задача 7: Log-Cosh Loss для прогнозирования курса акций ===")
    
    # Данные
    actual_prices = np.array([145.5, 142.8, 158.2, 144.1, 146.3])
    predicted_prices = np.array([146.0, 143.5, 148.0, 145.0, 145.8])
    
    # Решение
    errors = actual_prices - predicted_prices
    log_cosh_losses = np.log(np.cosh(errors))
    mean_log_cosh = np.mean(log_cosh_losses)
    
    print("Решение:")
    print(f"Средняя Log-Cosh Loss: {mean_log_cosh:.4f}")
    print("\nЗначения потерь по дням:")
    for i, loss in enumerate(log_cosh_losses):
        print(f"День {i+1}: {loss:.4f}")
    
    # Визуализация
    plt.figure(figsize=(12, 5))
    
    # График слева: цены акций
    plt.subplot(1, 2, 1)
    days = range(1, len(actual_prices) + 1)
    plt.plot(days, actual_prices, 'o-', label='Реальные цены', color='blue')
    plt.plot(days, predicted_prices, 'o--', label='Предсказанные цены', color='red')
    plt.fill_between(days, actual_prices, predicted_prices, alpha=0.2, color='gray')
    plt.xlabel('День')
    plt.ylabel('Цена (USD)')
    plt.title('Динамика цен акций')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График справа: сравнение функций потерь
    plt.subplot(1, 2, 2)
    mse_losses = errors**2
    mae_losses = np.abs(errors)
    
    x_pos = np.arange(len(errors))
    width = 0.25
    
    plt.bar(x_pos - width, mse_losses, width, label='MSE', color='skyblue')
    plt.bar(x_pos, mae_losses, width, label='MAE', color='lightgreen')
    plt.bar(x_pos + width, log_cosh_losses, width, label='Log-Cosh', color='lightcoral')
    
    plt.xlabel('День')
    plt.ylabel('Значение функции потерь')
    plt.yscale('log')
    plt.title('Сравнение функций потерь')
    plt.legend()
    
    plt.tight_layout()
    
    plt.savefig('imgs/log_cosh.png')
    plt.show()

def solve_quantile():
    """Решение задачи с Quantile Loss для управления складскими запасами"""
    print("\n=== Задача 8: Quantile Loss для управления складскими запасами ===")
    
    # Данные
    actual_demand = np.array([120, 115, 125, 118, 122])
    predicted_demand = np.array([118, 117, 123, 119, 120])
    tau = 0.7  # квантиль (>0.5 означает, что переоценка предпочтительнее недооценки)
    
    # Решение
    errors = actual_demand - predicted_demand
    quantile_losses = np.array([
        tau * err if err >= 0 else (tau - 1) * err
        for err in errors
    ])
    mean_quantile_loss = np.mean(quantile_losses)
    
    print("Решение:")
    print(f"Средняя Quantile Loss (τ={tau}): {mean_quantile_loss:.2f}")
    print("\nАнализ ошибок прогноза:")
    for i, (act, pred) in enumerate(zip(actual_demand, predicted_demand)):
        error_type = "недооценка" if act > pred else "переоценка"
        print(f"День {i+1}: {error_type} на {abs(act - pred)} единиц")
    
    # Визуализация
    plt.figure(figsize=(12, 5))
    
    # График слева: спрос
    plt.subplot(1, 2, 1)
    days = range(1, len(actual_demand) + 1)
    plt.plot(days, actual_demand, 'o-', label='Реальный спрос', color='blue')
    plt.plot(days, predicted_demand, 'o--', label='Предсказанный спрос', color='red')
    plt.fill_between(days, actual_demand, predicted_demand, 
                    where=(actual_demand >= predicted_demand),
                    color='red', alpha=0.2, label='Недооценка')
    plt.fill_between(days, actual_demand, predicted_demand,
                    where=(actual_demand < predicted_demand),
                    color='blue', alpha=0.2, label='Переоценка')
    plt.xlabel('День')
    plt.ylabel('Спрос (шт.)')
    plt.title(f'Прогноз спроса (τ={tau})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График справа: потери
    plt.subplot(1, 2, 2)
    plt.bar(days, quantile_losses, color=['red' if ql > 0 else 'blue' 
                                        for ql in quantile_losses], alpha=0.6)
    plt.axhline(y=mean_quantile_loss, color='black', linestyle='--',
                label=f'Средняя потеря = {mean_quantile_loss:.2f}')
    plt.xlabel('День')
    plt.ylabel('Quantile Loss')
    plt.title('Значения квантильной функции потерь')
    plt.legend()
    
    plt.tight_layout()
    
    plt.savefig('imgs/quantile.png')
    plt.show()

def solve_huber_improved():
    """Улучшенная визуализация для Huber Loss"""
    print("\n=== Демонстрация свойств Huber Loss ===")
    
    # Создаем данные с разными типами ошибок
    errors = np.array([-20, -5, -2, 0, 2, 5, 20])  # от маленьких до больших ошибок
    delta = 4  # параметр перехода от квадратичной к линейной части
    
    # Вычисляем значения разных функций потерь
    mse = errors**2
    mae = np.abs(errors)
    huber = np.array([
        0.5 * err**2 if abs(err) <= delta else
        delta * abs(err) - 0.5 * delta**2 
        for err in errors
    ])
    
    # Визуализация
    plt.figure(figsize=(15, 5))
    
    # График 1: Сравнение функций на всем диапазоне
    plt.subplot(1, 2, 1)
    x = range(len(errors))
    width = 0.25
    
    plt.bar([i-width for i in x], mse, width, label='MSE', color='skyblue')
    plt.bar([i for i in x], mae, width, label='MAE', color='lightgreen')
    plt.bar([i+width for i in x], huber, width, label='Huber', color='lightcoral')
    
    plt.axvline(x=3, color='gray', linestyle='--', alpha=0.5)
    plt.title('Сравнение функций потерь\nОбратите внимание на масштаб значений')
    plt.xlabel('Ошибка')
    plt.ylabel('Значение функции потерь')
    plt.yscale("log")
    plt.grid()
    plt.legend()
    
    # Добавим подписи значений ошибок
    plt.xticks(x, [str(e) for e in errors])
    
    # График 2: Детальное сравнение для малых ошибок
    plt.subplot(1, 2, 2)
    # Выбираем только малые ошибки
    small_errors_mask = np.abs(errors) <= 5
    small_errors = errors[small_errors_mask]
    small_mse = mse[small_errors_mask]
    small_mae = mae[small_errors_mask]
    small_huber = huber[small_errors_mask]
    
    x = range(len(small_errors))
    plt.bar([i-width for i in x], small_mse, width, label='MSE', color='skyblue')
    plt.bar([i for i in x], small_mae, width, label='MAE', color='lightgreen')
    plt.bar([i+width for i in x], small_huber, width, label='Huber', color='lightcoral')
    
    plt.axvline(x=2, color='gray', linestyle='--', alpha=0.5)
    plt.title(f'Детальный вид для малых ошибок\nδ = {delta}')
    plt.xlabel('Ошибка')
    plt.ylabel('Значение функции потерь')
    plt.yscale("log")
    plt.grid()
    plt.legend()
    
    # Добавим подписи значений ошибок
    plt.xticks(x, [str(e) for e in small_errors])
    
    plt.tight_layout()
    plt.savefig(f"imgs/huber_plus.png")
    plt.show()
    
    # Вывод значений для сравнения
    print("\nСравнение значений функций потерь:")
    print("Ошибка  |   MSE   |   MAE   |  Huber")
    print("-" * 40)
    for e, m1, m2, h in zip(errors, mse, mae, huber):
        print(f"{e:7.1f} | {m1:7.2f} | {m2:7.2f} | {h:7.2f}")

def solve_fair_improved():
    """Улучшенная демонстрация свойств Fair Loss"""
    print("\n=== Демонстрация свойств Fair Loss ===")
    
    # Создаем диапазон ошибок для демонстрации
    errors = np.array([-20, -10, -5, -2, 0, 2, 5, 10, 20])
    c = 3.0  # параметр Fair Loss
    
    # Вычисляем значения разных функций потерь
    mse = errors**2
    mae = np.abs(errors)
    fair = c**2 * (np.abs(errors)/c - np.log(1 + np.abs(errors)/c))
    
    plt.figure(figsize=(15, 5))
    
    # График 1: Сравнение функций потерь
    plt.subplot(1, 2, 1)
    plt.plot(errors, mse/np.max(mse), label='MSE (норм.)', linestyle='-', marker='o')
    plt.plot(errors, mae/np.max(mae), label='MAE (норм.)', linestyle='-', marker='o')
    plt.plot(errors, fair/np.max(fair), label='Fair Loss (норм.)', linestyle='-', marker='o')
    
    plt.title('Сравнение роста функций потерь\n(нормализованные значения)')
    plt.xlabel('Величина ошибки')
    plt.ylabel('Нормализованное значение потерь')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # График 2: Абсолютные значения
    plt.subplot(1, 2, 2)
    plt.plot(errors, mse, label='MSE', linestyle='-', marker='o')
    plt.plot(errors, mae, label='MAE', linestyle='-', marker='o')
    plt.plot(errors, fair, label='Fair Loss', linestyle='-', marker='o')
    
    plt.title(f'Абсолютные значения функций потерь\n(c={c})')
    plt.xlabel('Величина ошибки')
    plt.ylabel('Значение функции потерь')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("imgs/fair_plus.png")
    plt.show()
    
    # Вывод значений для сравнения
    print("\nСравнение значений функций потерь:")
    print("Ошибка  |   MSE   |   MAE   |   Fair")
    print("-" * 40)
    for e, m1, m2, f in zip(errors, mse, mae, fair):
        print(f"{e:7.1f} | {m1:7.2f} | {m2:7.2f} | {f:7.2f}")

# Дополним основную функцию main() новыми решениями
def main():
    """Запуск всех решений"""
    # solve_mse()
    # solve_mae()
    # solve_mape()
    # solve_huber()
    # solve_huber_improved()
    # solve_fair()
    # solve_fair_improved()
    solve_tukey()
    # solve_log_cosh()
    # solve_quantile()

if __name__ == "__main__":
    main()
