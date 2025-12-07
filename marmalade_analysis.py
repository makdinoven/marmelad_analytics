#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сценарный анализ выхода СОАО «Коммунарка» на рынок мармелада

Программа моделирует запуск новой линейки мармелада в условиях конкуренции
с ОАО «Красный пищевик» и оценивает финансовые результаты проекта
при трёх сценариях развития рынка.

Автор: Сценарный анализ
Дата: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMaximize, LpVariable, value, LpStatus

# Настройка matplotlib для корректного отображения кириллицы
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. ИСХОДНЫЕ ПАРАМЕТРЫ
# ============================================================================

# --- Параметры конкурента (ОАО «Красный пищевик») ---
COMPETITOR_CAPACITY = 3924      # Производственная мощность по мармеладу, тонн/год
COMPETITOR_OUTPUT = 2624        # Фактический выпуск в 2024 году, тонн

# --- Параметры новой линии СОАО «Коммунарка» ---
CAPACITY_NEW_LINE = 1000        # Проектная мощность новой линии, тонн/год
PRICE_NEW = 8000                # Отпускная цена мармелада, руб./т
VAR_COST_NEW = 4800             # Переменные затраты на 1 т, руб./т
FIXED_COSTS = 500_000           # Постоянные затраты (амортизация, зарплаты), руб./год

# --- Сценарные значения спроса ---
DEMAND_BASE = 600               # Базовый сценарий, тонн
DEMAND_OPT = 900                # Оптимистичный сценарий, тонн
DEMAND_PESS = 350               # Пессимистичный сценарий, тонн

# --- Корректировки цены и затрат по сценариям (умеренные) ---
# Формат: (коэффициент цены, коэффициент затрат)
ADJUSTMENTS = {
    'base': (1.00, 1.00),       # Без изменений
    'optimistic': (1.05, 0.95), # Цена +5%, затраты -5%
    'pessimistic': (0.95, 1.05) # Цена -5%, затраты +5%
}

# --- Параметры Монте-Карло ---
MC_VARIATION = 0.30             # Диапазон отклонения ±30% (высокая неопределённость)
MC_SIMULATIONS = 50_000         # Количество прогонов

# ============================================================================
# 2. ФОРМИРОВАНИЕ СЦЕНАРИЕВ
# ============================================================================

def create_scenarios():
    """Создаёт словарь со всеми параметрами трёх сценариев."""
    scenarios = {
        'Базовый': {
            'demand': DEMAND_BASE,
            'price': PRICE_NEW * ADJUSTMENTS['base'][0],
            'var_cost': VAR_COST_NEW * ADJUSTMENTS['base'][1],
            'description': 'Умеренный вход на рынок, стабильная конкуренция'
        },
        'Оптимистичный': {
            'demand': DEMAND_OPT,
            'price': PRICE_NEW * ADJUSTMENTS['optimistic'][0],
            'var_cost': VAR_COST_NEW * ADJUSTMENTS['optimistic'][1],
            'description': 'Высокий спрос, ослабление конкурента, рост цен'
        },
        'Пессимистичный': {
            'demand': DEMAND_PESS,
            'price': PRICE_NEW * ADJUSTMENTS['pessimistic'][0],
            'var_cost': VAR_COST_NEW * ADJUSTMENTS['pessimistic'][1],
            'description': 'Низкий спрос, агрессивная конкуренция, снижение цен'
        }
    }
    return scenarios

# ============================================================================
# 3. РАСЧЁТ КЛЮЧЕВЫХ ПОКАЗАТЕЛЕЙ
# ============================================================================

def calculate_indicators(scenarios):
    """
    Рассчитывает ключевые показатели для каждого сценария.
    Возвращает pandas DataFrame.
    """
    results = []
    
    # Общий рынок = выпуск конкурента + выпуск Коммунарки
    for name, params in scenarios.items():
        # Объём выпуска ограничен мощностью линии и спросом
        volume = min(params['demand'], CAPACITY_NEW_LINE)
        
        # Финансовые показатели
        revenue = volume * params['price']
        var_costs = volume * params['var_cost']
        gross_profit = revenue - var_costs
        net_profit = gross_profit - FIXED_COSTS
        
        # Доля рынка (выпуск Коммунарки / общий выпуск рынка)
        total_market = COMPETITOR_OUTPUT + volume
        market_share = (volume / total_market) * 100
        
        # Маржинальность
        margin = ((params['price'] - params['var_cost']) / params['price']) * 100
        
        results.append({
            'Сценарий': name,
            'Спрос (т)': params['demand'],
            'Объём выпуска (т)': volume,
            'Цена (руб./т)': params['price'],
            'Перем. затраты (руб./т)': params['var_cost'],
            'Выручка (руб.)': revenue,
            'Перем. затраты общие (руб.)': var_costs,
            'Валовая прибыль (руб.)': gross_profit,
            'Постоянные затраты (руб.)': FIXED_COSTS,
            'Чистая прибыль (руб.)': net_profit,
            'Доля рынка (%)': market_share,
            'Маржинальность (%)': margin
        })
    
    df = pd.DataFrame(results)
    return df

# ============================================================================
# 4. ИМИТАЦИОННОЕ МОДЕЛИРОВАНИЕ МОНТЕ-КАРЛО
# ============================================================================

def monte_carlo_simulation(scenario_params, n_simulations=MC_SIMULATIONS):
    """
    Выполняет имитационное моделирование методом Монте-Карло.
    Использует треугольное распределение для спроса и цены.
    
    Возвращает массив значений прибыли и статистику.
    """
    np.random.seed(42)  # Для воспроизводимости
    
    base_demand = scenario_params['demand']
    base_price = scenario_params['price']
    base_var_cost = scenario_params['var_cost']
    
    # Границы треугольного распределения (±20%)
    demand_low = base_demand * (1 - MC_VARIATION)
    demand_high = base_demand * (1 + MC_VARIATION)
    
    price_low = base_price * (1 - MC_VARIATION)
    price_high = base_price * (1 + MC_VARIATION)
    
    var_cost_low = base_var_cost * (1 - MC_VARIATION)
    var_cost_high = base_var_cost * (1 + MC_VARIATION)
    
    # Генерация случайных значений (треугольное распределение)
    demands = np.random.triangular(demand_low, base_demand, demand_high, n_simulations)
    prices = np.random.triangular(price_low, base_price, price_high, n_simulations)
    var_costs = np.random.triangular(var_cost_low, base_var_cost, var_cost_high, n_simulations)
    
    # Расчёт прибыли для каждой реализации
    volumes = np.minimum(demands, CAPACITY_NEW_LINE)
    revenues = volumes * prices
    total_var_costs = volumes * var_costs
    profits = revenues - total_var_costs - FIXED_COSTS
    
    # Статистика
    stats = {
        'mean_profit': np.mean(profits),
        'std_profit': np.std(profits),
        'min_profit': np.min(profits),
        'max_profit': np.max(profits),
        'median_profit': np.median(profits),
        'percentile_5': np.percentile(profits, 5),
        'percentile_95': np.percentile(profits, 95),
        'prob_loss': np.mean(profits < 0) * 100,  # Вероятность убытка в %
        'prob_high_profit': np.mean(profits > 1_000_000) * 100  # Вероятность высокой прибыли
    }
    
    return profits, stats

def run_monte_carlo_all_scenarios(scenarios):
    """Запускает Монте-Карло для всех сценариев."""
    mc_results = {}
    for name, params in scenarios.items():
        profits, stats = monte_carlo_simulation(params)
        mc_results[name] = {
            'profits': profits,
            'stats': stats
        }
    return mc_results

# ============================================================================
# 5. ВИЗУАЛИЗАЦИЯ (MATPLOTLIB)
# ============================================================================

def plot_scenario_comparison(df):
    """Строит столбчатые диаграммы сравнения выручки и прибыли по сценариям."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scenarios = df['Сценарий'].tolist()
    x = np.arange(len(scenarios))
    width = 0.5
    
    # Цвета для сценариев
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    # График 1: Выручка и затраты
    ax1 = axes[0]
    revenues = df['Выручка (руб.)'].values / 1_000_000
    var_costs = df['Перем. затраты общие (руб.)'].values / 1_000_000
    fixed = FIXED_COSTS / 1_000_000
    
    bars1 = ax1.bar(x - width/3, revenues, width/1.5, label='Выручка', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/3, var_costs + fixed, width/1.5, label='Общие затраты', color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('Сценарий', fontsize=12)
    ax1.set_ylabel('Сумма (млн руб.)', fontsize=12)
    ax1.set_title('Сравнение выручки и затрат по сценариям', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # График 2: Чистая прибыль
    ax2 = axes[1]
    profits = df['Чистая прибыль (руб.)'].values / 1_000_000
    
    bar_colors = ['#2ecc71' if p >= 0 else '#e74c3c' for p in profits]
    bars3 = ax2.bar(scenarios, profits, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Сценарий', fontsize=12)
    ax2.set_ylabel('Чистая прибыль (млн руб.)', fontsize=12)
    ax2.set_title('Чистая прибыль по сценариям', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar, profit in zip(bars3, profits):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 3 if height >= 0 else -10
        ax2.annotate(f'{profit:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset), textcoords="offset points",
                    ha='center', va=va, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('scenario_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✓ График сохранён: scenario_comparison.png")

def plot_monte_carlo_histogram(mc_results, scenario_name='Базовый'):
    """Строит гистограмму распределения прибыли по результатам Монте-Карло."""
    profits = mc_results[scenario_name]['profits'] / 1_000_000
    stats = mc_results[scenario_name]['stats']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Гистограмма
    n, bins, patches = ax.hist(profits, bins=50, edgecolor='white', alpha=0.7)
    
    # Раскрашиваем столбцы: красные для убытков, зелёные для прибыли
    for i, patch in enumerate(patches):
        if bins[i] < 0:
            patch.set_facecolor('#e74c3c')
        else:
            patch.set_facecolor('#2ecc71')
    
    # Вертикальные линии для статистик
    mean_profit = stats['mean_profit'] / 1_000_000
    ax.axvline(mean_profit, color='#2980b9', linestyle='--', linewidth=2, 
               label=f'Средняя: {mean_profit:.2f} млн')
    ax.axvline(0, color='black', linestyle='-', linewidth=1.5, label='Точка безубыточности')
    
    # Заливка области убытков
    ax.axvspan(ax.get_xlim()[0], 0, alpha=0.1, color='red')
    
    ax.set_xlabel('Чистая прибыль (млн руб.)', fontsize=12)
    ax.set_ylabel('Частота', fontsize=12)
    ax.set_title(f'Распределение прибыли (Монте-Карло, {MC_SIMULATIONS:,} прогонов)\n'
                f'Сценарий: {scenario_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Текстовый блок со статистикой
    textstr = '\n'.join([
        f'Средняя прибыль: {stats["mean_profit"]/1e6:.2f} млн',
        f'Стд. отклонение: {stats["std_profit"]/1e6:.2f} млн',
        f'Мин / Макс: {stats["min_profit"]/1e6:.2f} / {stats["max_profit"]/1e6:.2f} млн',
        f'5% / 95% перцентиль: {stats["percentile_5"]/1e6:.2f} / {stats["percentile_95"]/1e6:.2f} млн',
        f'Вероятность убытка: {stats["prob_loss"]:.1f}%'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('monte_carlo_histogram.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✓ График сохранён: monte_carlo_histogram.png")

def plot_market_share(df):
    """Строит диаграмму долей рынка."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = df['Сценарий'].tolist()
    kommunarka_share = df['Доля рынка (%)'].values
    competitor_share = 100 - kommunarka_share
    
    x = np.arange(len(scenarios))
    width = 0.6
    
    ax.bar(x, competitor_share, width, label='Красный пищевик', color='#e74c3c', alpha=0.8)
    ax.bar(x, kommunarka_share, width, bottom=competitor_share, label='Коммунарка', color='#3498db', alpha=0.8)
    
    # Добавляем подписи процентов
    for i, (k, c) in enumerate(zip(kommunarka_share, competitor_share)):
        ax.text(i, c + k/2, f'{k:.1f}%', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        ax.text(i, c/2, f'{c:.1f}%', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    ax.set_xlabel('Сценарий', fontsize=12)
    ax.set_ylabel('Доля рынка (%)', fontsize=12)
    ax.set_title('Распределение долей рынка мармелада', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('market_share.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✓ График сохранён: market_share.png")

# ============================================================================
# 6. ОПТИМИЗАЦИОННАЯ МОДЕЛЬ (PuLP)
# ============================================================================

def optimize_production(scenario_name, scenario_params):
    """
    Решает оптимизационную задачу максимизации прибыли.
    
    Переменная: объём выпуска (volume)
    Целевая функция: max (price - var_cost) * volume - fixed_costs
    Ограничения:
        - volume <= capacity_new_line
        - volume <= demand
        - volume >= 0
    """
    # Создаём задачу максимизации
    prob = LpProblem(f"Profit_Maximization_{scenario_name}", LpMaximize)
    
    # Переменная решения: объём выпуска
    volume = LpVariable("volume", lowBound=0, upBound=CAPACITY_NEW_LINE, cat='Continuous')
    
    # Параметры сценария
    price = scenario_params['price']
    var_cost = scenario_params['var_cost']
    demand = scenario_params['demand']
    
    # Целевая функция: максимизация прибыли
    # Прибыль = (цена - переменные затраты) * объём - постоянные затраты
    margin_per_unit = price - var_cost
    prob += margin_per_unit * volume - FIXED_COSTS, "Total_Profit"
    
    # Ограничение по спросу
    prob += volume <= demand, "Demand_Constraint"
    
    # Решаем задачу
    prob.solve()
    
    # Результаты
    optimal_volume = value(volume)
    optimal_profit = value(prob.objective)
    status = LpStatus[prob.status]
    
    return {
        'scenario': scenario_name,
        'status': status,
        'optimal_volume': optimal_volume,
        'optimal_profit': optimal_profit,
        'margin_per_ton': margin_per_unit,
        'revenue': optimal_volume * price,
        'total_var_costs': optimal_volume * var_cost,
        'utilization': (optimal_volume / CAPACITY_NEW_LINE) * 100
    }

def run_optimization_all_scenarios(scenarios):
    """Запускает оптимизацию для всех сценариев."""
    opt_results = []
    for name, params in scenarios.items():
        result = optimize_production(name, params)
        opt_results.append(result)
    return opt_results

# ============================================================================
# 7. ИТОГОВЫЙ ВЫВОД И РЕКОМЕНДАЦИИ
# ============================================================================

def print_summary(df, mc_results, opt_results):
    """Формирует итоговый текстовый вывод с рекомендациями."""
    print("\n" + "="*80)
    print("                    ИТОГОВЫЙ АНАЛИЗ И РЕКОМЕНДАЦИИ")
    print("="*80)
    
    # Сравнение сценариев
    print("\n📊 СРАВНЕНИЕ СЦЕНАРИЕВ ПО КЛЮЧЕВЫМ ПОКАЗАТЕЛЯМ:")
    print("-" * 60)
    
    best_profit_idx = df['Чистая прибыль (руб.)'].idxmax()
    worst_profit_idx = df['Чистая прибыль (руб.)'].idxmin()
    
    for idx, row in df.iterrows():
        profit = row['Чистая прибыль (руб.)']
        marker = "★" if idx == best_profit_idx else "  "
        print(f"{marker} {row['Сценарий']:15} | Прибыль: {profit/1e6:>7.2f} млн руб. | "
              f"Доля рынка: {row['Доля рынка (%)']:>5.1f}% | Объём: {row['Объём выпуска (т)']:>4.0f} т")
    
    best_scenario = df.loc[best_profit_idx, 'Сценарий']
    best_profit = df.loc[best_profit_idx, 'Чистая прибыль (руб.)']
    
    print(f"\n✅ Наиболее выгодный сценарий: {best_scenario}")
    print(f"   Ожидаемая прибыль: {best_profit/1e6:.2f} млн руб.")
    
    # Анализ рисков (Монте-Карло)
    print("\n📈 АНАЛИЗ РИСКОВ (по результатам Монте-Карло):")
    print("-" * 60)
    
    for name, data in mc_results.items():
        stats = data['stats']
        risk_level = "низкий" if stats['prob_loss'] < 10 else "средний" if stats['prob_loss'] < 30 else "высокий"
        print(f"  {name:15} | Вероятность убытка: {stats['prob_loss']:>5.1f}% ({risk_level})")
        print(f"                  | Средняя прибыль: {stats['mean_profit']/1e6:>6.2f} млн ± {stats['std_profit']/1e6:.2f} млн")
    
    # Результаты оптимизации
    print("\n🎯 РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ (максимизация прибыли):")
    print("-" * 60)
    
    for opt in opt_results:
        print(f"  {opt['scenario']:15} | Оптимальный объём: {opt['optimal_volume']:>6.0f} т | "
              f"Макс. прибыль: {opt['optimal_profit']/1e6:>6.2f} млн руб. | "
              f"Загрузка линии: {opt['utilization']:>5.1f}%")
    
    # Итоговая рекомендация
    print("\n" + "="*80)
    print("                         РЕКОМЕНДАЦИЯ")
    print("="*80)
    
    base_stats = mc_results['Базовый']['stats']
    base_profit = df[df['Сценарий'] == 'Базовый']['Чистая прибыль (руб.)'].values[0]
    
    if base_profit > 0 and base_stats['prob_loss'] < 20:
        recommendation = "РЕКОМЕНДУЕТСЯ"
        reason = "положительную прибыль в базовом сценарии и приемлемый уровень риска"
    elif base_profit > 0 and base_stats['prob_loss'] < 40:
        recommendation = "РЕКОМЕНДУЕТСЯ С ОГОВОРКАМИ"
        reason = "положительную прибыль, но повышенный уровень риска"
    else:
        recommendation = "НЕ РЕКОМЕНДУЕТСЯ"
        reason = "высокий риск убытков"
    
    print(f"""
    Запуск линии мармелада на СОАО «Коммунарка»: {recommendation}
    
    Обоснование:
    • Базовый сценарий показывает {reason}.
    • Вероятность убытка в базовом сценарии: {base_stats['prob_loss']:.1f}%
    • Ожидаемая прибыль в базовом сценарии: {base_profit/1e6:.2f} млн руб.
    
    Рекомендуемый объём производства:
    • При базовом сценарии: {DEMAND_BASE} тонн/год
    • Оптимальная загрузка линии: {(min(DEMAND_BASE, CAPACITY_NEW_LINE)/CAPACITY_NEW_LINE)*100:.0f}%
    
    Ключевые факторы успеха:
    1. Снижение выпуска конкурента (Красный пищевик) создаёт «окно возможностей»
    2. Маржинальность продукта ({(PRICE_NEW-VAR_COST_NEW)/PRICE_NEW*100:.0f}%) обеспечивает запас прочности
    3. При оптимистичном сценарии прибыль может достичь {df[df['Сценарий']=='Оптимистичный']['Чистая прибыль (руб.)'].values[0]/1e6:.2f} млн руб.
    
    Риски:
    • При пессимистичном сценарии возможен убыток до {abs(df[df['Сценарий']=='Пессимистичный']['Чистая прибыль (руб.)'].values[0])/1e6:.2f} млн руб.
    • Вероятность убытка при пессимистичном сценарии: {mc_results['Пессимистичный']['stats']['prob_loss']:.1f}%
    """)
    
    print("="*80)

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Основная функция программы."""
    print("="*80)
    print("     СЦЕНАРНЫЙ АНАЛИЗ ВЫХОДА СОАО «КОММУНАРКА» НА РЫНОК МАРМЕЛАДА")
    print("="*80)
    
    # 1. Вывод исходных параметров
    print("\n📋 ИСХОДНЫЕ ПАРАМЕТРЫ:")
    print("-" * 40)
    print(f"  Мощность новой линии:      {CAPACITY_NEW_LINE:,} т/год")
    print(f"  Отпускная цена:            {PRICE_NEW:,} руб./т")
    print(f"  Переменные затраты:        {VAR_COST_NEW:,} руб./т")
    print(f"  Постоянные затраты:        {FIXED_COSTS:,} руб./год")
    print(f"  Маржа на тонну:            {PRICE_NEW - VAR_COST_NEW:,} руб./т")
    print(f"\n  Конкурент (Красный пищевик):")
    print(f"    Мощность:                {COMPETITOR_CAPACITY:,} т/год")
    print(f"    Выпуск 2024:             {COMPETITOR_OUTPUT:,} т")
    
    # 2. Формирование сценариев
    print("\n" + "="*80)
    print("                         СЦЕНАРИИ РАЗВИТИЯ")
    print("="*80)
    scenarios = create_scenarios()
    
    for name, params in scenarios.items():
        print(f"\n  {name}:")
        print(f"    Описание: {params['description']}")
        print(f"    Спрос: {params['demand']} т | Цена: {params['price']:.0f} руб./т | "
              f"Затраты: {params['var_cost']:.0f} руб./т")
    
    # 3. Расчёт показателей
    print("\n" + "="*80)
    print("                    КЛЮЧЕВЫЕ ПОКАЗАТЕЛИ ПО СЦЕНАРИЯМ")
    print("="*80)
    df = calculate_indicators(scenarios)
    
    # Форматирование для вывода
    df_display = df.copy()
    for col in df_display.columns:
        if 'руб' in col:
            df_display[col] = df_display[col].apply(lambda x: f"{x:,.0f}")
        elif '%' in col:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.1f}%")
    
    print("\n", df_display.to_string(index=False))
    
    # 4. Монте-Карло моделирование
    print("\n" + "="*80)
    print("              ИМИТАЦИОННОЕ МОДЕЛИРОВАНИЕ (МОНТЕ-КАРЛО)")
    print("="*80)
    print(f"\n  Параметры: {MC_SIMULATIONS:,} прогонов, диапазон ±{MC_VARIATION*100:.0f}%")
    
    mc_results = run_monte_carlo_all_scenarios(scenarios)
    
    print("\n  Результаты моделирования:")
    print("-" * 70)
    print(f"  {'Сценарий':15} | {'Сред. прибыль':>14} | {'Стд. откл.':>12} | {'P(убыток)':>10}")
    print("-" * 70)
    for name, data in mc_results.items():
        stats = data['stats']
        print(f"  {name:15} | {stats['mean_profit']/1e6:>11.2f} млн | "
              f"{stats['std_profit']/1e6:>9.2f} млн | {stats['prob_loss']:>8.1f}%")
    
    # 5. Оптимизация
    print("\n" + "="*80)
    print("                    ОПТИМИЗАЦИЯ ОБЪЁМА ВЫПУСКА (PuLP)")
    print("="*80)
    
    opt_results = run_optimization_all_scenarios(scenarios)
    
    print("\n  Результаты оптимизации (максимизация прибыли):")
    print("-" * 80)
    for opt in opt_results:
        print(f"\n  {opt['scenario']}:")
        print(f"    Статус решения:     {opt['status']}")
        print(f"    Оптимальный объём:  {opt['optimal_volume']:.0f} т")
        print(f"    Максимальная прибыль: {opt['optimal_profit']/1e6:.2f} млн руб.")
        print(f"    Загрузка линии:     {opt['utilization']:.1f}%")
    
    # 6. Визуализация
    print("\n" + "="*80)
    print("                         ПОСТРОЕНИЕ ГРАФИКОВ")
    print("="*80)
    
    plot_scenario_comparison(df)
    plot_monte_carlo_histogram(mc_results, 'Базовый')
    plot_market_share(df)
    
    # 7. Итоговый вывод
    print_summary(df, mc_results, opt_results)
    
    # Сохранение результатов в CSV
    df.to_csv('scenario_results.csv', index=False, encoding='utf-8-sig')
    print("\n✓ Результаты сохранены: scenario_results.csv")
    
    return df, mc_results, opt_results

# ============================================================================
# ЗАПУСК ПРОГРАММЫ
# ============================================================================

if __name__ == "__main__":
    df, mc_results, opt_results = main()

