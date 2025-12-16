#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Веб-приложение для сценарного анализа выхода СОАО «Коммунарка» на рынок мармелада
Flask + Jinja2 интерфейс
"""

import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Используем не-интерактивный бэкенд
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from pulp import LpProblem, LpMaximize, LpVariable, value, LpStatus

# Настройка matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)

# ============================================================================
# ФУНКЦИИ РАСЧЁТА (адаптированы из marmalade_analysis.py)
# ============================================================================

def create_scenarios(params):
    """Создаёт словарь со всеми параметрами трёх сценариев."""
    scenarios = {
        'Базовый': {
            'demand': params['demand_base'],
            'price': params['price_new'] * params['price_adj_base'],
            'var_cost': params['var_cost_new'] * params['cost_adj_base'],
            'description': 'Умеренный вход на рынок, стабильная конкуренция'
        },
        'Оптимистичный': {
            'demand': params['demand_opt'],
            'price': params['price_new'] * params['price_adj_opt'],
            'var_cost': params['var_cost_new'] * params['cost_adj_opt'],
            'description': 'Высокий спрос, ослабление конкурента, рост цен'
        },
        'Пессимистичный': {
            'demand': params['demand_pess'],
            'price': params['price_new'] * params['price_adj_pess'],
            'var_cost': params['var_cost_new'] * params['cost_adj_pess'],
            'description': 'Низкий спрос, агрессивная конкуренция, снижение цен'
        }
    }
    return scenarios


def calculate_indicators(scenarios, params):
    """Рассчитывает ключевые показатели для каждого сценария."""
    results = []
    
    for name, sc in scenarios.items():
        volume = min(sc['demand'], params['capacity_new_line'])
        revenue = volume * sc['price']
        var_costs = volume * sc['var_cost']
        gross_profit = revenue - var_costs
        net_profit = gross_profit - params['fixed_costs']
        total_market = params['competitor_output'] + volume
        market_share = (volume / total_market) * 100
        margin = ((sc['price'] - sc['var_cost']) / sc['price']) * 100
        
        results.append({
            'Сценарий': name,
            'Спрос (т)': sc['demand'],
            'Объём выпуска (т)': volume,
            'Цена (руб./т)': sc['price'],
            'Перем. затраты (руб./т)': sc['var_cost'],
            'Выручка (руб.)': revenue,
            'Перем. затраты общие (руб.)': var_costs,
            'Валовая прибыль (руб.)': gross_profit,
            'Постоянные затраты (руб.)': params['fixed_costs'],
            'Чистая прибыль (руб.)': net_profit,
            'Доля рынка (%)': market_share,
            'Маржинальность (%)': margin
        })
    
    return pd.DataFrame(results)


def monte_carlo_simulation(scenario_params, params, n_simulations=10000):
    """Выполняет имитационное моделирование методом Монте-Карло."""
    np.random.seed(42)
    
    base_demand = scenario_params['demand']
    base_price = scenario_params['price']
    base_var_cost = scenario_params['var_cost']
    variation = params['mc_variation']
    
    demand_low = base_demand * (1 - variation)
    demand_high = base_demand * (1 + variation)
    price_low = base_price * (1 - variation)
    price_high = base_price * (1 + variation)
    var_cost_low = base_var_cost * (1 - variation)
    var_cost_high = base_var_cost * (1 + variation)
    
    demands = np.random.triangular(demand_low, base_demand, demand_high, n_simulations)
    prices = np.random.triangular(price_low, base_price, price_high, n_simulations)
    var_costs = np.random.triangular(var_cost_low, base_var_cost, var_cost_high, n_simulations)
    
    volumes = np.minimum(demands, params['capacity_new_line'])
    revenues = volumes * prices
    total_var_costs = volumes * var_costs
    profits = revenues - total_var_costs - params['fixed_costs']
    
    stats = {
        'mean_profit': np.mean(profits),
        'std_profit': np.std(profits),
        'min_profit': np.min(profits),
        'max_profit': np.max(profits),
        'median_profit': np.median(profits),
        'percentile_5': np.percentile(profits, 5),
        'percentile_95': np.percentile(profits, 95),
        'prob_loss': np.mean(profits < 0) * 100,
    }
    
    return profits, stats


def run_monte_carlo_all_scenarios(scenarios, params):
    """Запускает Монте-Карло для всех сценариев."""
    mc_results = {}
    for name, sc_params in scenarios.items():
        profits, stats = monte_carlo_simulation(sc_params, params, params['mc_simulations'])
        mc_results[name] = {'profits': profits, 'stats': stats}
    return mc_results


def optimize_production(scenario_name, scenario_params, params):
    """Решает оптимизационную задачу максимизации прибыли."""
    prob = LpProblem(f"Profit_Max_{scenario_name}", LpMaximize)
    volume = LpVariable("volume", lowBound=0, upBound=params['capacity_new_line'], cat='Continuous')
    
    price = scenario_params['price']
    var_cost = scenario_params['var_cost']
    demand = scenario_params['demand']
    margin_per_unit = price - var_cost
    
    prob += margin_per_unit * volume - params['fixed_costs'], "Total_Profit"
    prob += volume <= demand, "Demand_Constraint"
    prob.solve()
    
    optimal_volume = value(volume)
    optimal_profit = value(prob.objective)
    
    return {
        'scenario': scenario_name,
        'status': LpStatus[prob.status],
        'optimal_volume': optimal_volume,
        'optimal_profit': optimal_profit,
        'utilization': (optimal_volume / params['capacity_new_line']) * 100
    }


def run_optimization_all_scenarios(scenarios, params):
    """Запускает оптимизацию для всех сценариев."""
    return [optimize_production(name, sc, params) for name, sc in scenarios.items()]


# ============================================================================
# ТОЧКА БЕЗУБЫТОЧНОСТИ
# ============================================================================

def calculate_break_even(params):
    """Рассчитывает точку безубыточности."""
    price = params['price_new']
    var_cost = params['var_cost_new']
    fixed_costs = params['fixed_costs']
    
    # Маржинальная прибыль на единицу
    margin_per_unit = price - var_cost
    
    # Точка безубыточности в натуральном выражении
    if margin_per_unit > 0:
        break_even_volume = fixed_costs / margin_per_unit
    else:
        break_even_volume = float('inf')
    
    # Точка безубыточности в денежном выражении
    break_even_revenue = break_even_volume * price
    
    # Запас прочности для базового сценария
    base_volume = min(params['demand_base'], params['capacity_new_line'])
    if base_volume > 0:
        safety_margin = ((base_volume - break_even_volume) / base_volume) * 100
    else:
        safety_margin = 0
    
    # Коэффициент покрытия (доля маржинальной прибыли в выручке)
    coverage_ratio = (margin_per_unit / price) * 100 if price > 0 else 0
    
    return {
        'break_even_volume': break_even_volume,
        'break_even_revenue': break_even_revenue,
        'margin_per_unit': margin_per_unit,
        'safety_margin': safety_margin,
        'coverage_ratio': coverage_ratio,
        'base_volume': base_volume
    }


# ============================================================================
# АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ
# ============================================================================

def sensitivity_analysis(params):
    """Анализ чувствительности: как изменение параметров влияет на прибыль."""
    base_volume = min(params['demand_base'], params['capacity_new_line'])
    base_price = params['price_new']
    base_var_cost = params['var_cost_new']
    fixed_costs = params['fixed_costs']
    
    # Базовая прибыль
    base_profit = base_volume * (base_price - base_var_cost) - fixed_costs
    
    # Диапазон изменения параметров (±30%)
    variation = 0.30
    
    results = []
    
    # Влияние цены
    price_low = base_price * (1 - variation)
    price_high = base_price * (1 + variation)
    profit_price_low = base_volume * (price_low - base_var_cost) - fixed_costs
    profit_price_high = base_volume * (price_high - base_var_cost) - fixed_costs
    results.append({
        'parameter': 'Цена',
        'low_value': price_low,
        'high_value': price_high,
        'profit_low': profit_price_low,
        'profit_high': profit_price_high,
        'impact': abs(profit_price_high - profit_price_low)
    })
    
    # Влияние переменных затрат
    var_cost_low = base_var_cost * (1 - variation)
    var_cost_high = base_var_cost * (1 + variation)
    profit_var_low = base_volume * (base_price - var_cost_high) - fixed_costs
    profit_var_high = base_volume * (base_price - var_cost_low) - fixed_costs
    results.append({
        'parameter': 'Перем. затраты',
        'low_value': var_cost_low,
        'high_value': var_cost_high,
        'profit_low': profit_var_low,
        'profit_high': profit_var_high,
        'impact': abs(profit_var_high - profit_var_low)
    })
    
    # Влияние спроса/объёма
    volume_low = base_volume * (1 - variation)
    volume_high = min(base_volume * (1 + variation), params['capacity_new_line'])
    profit_vol_low = volume_low * (base_price - base_var_cost) - fixed_costs
    profit_vol_high = volume_high * (base_price - base_var_cost) - fixed_costs
    results.append({
        'parameter': 'Объём продаж',
        'low_value': volume_low,
        'high_value': volume_high,
        'profit_low': profit_vol_low,
        'profit_high': profit_vol_high,
        'impact': abs(profit_vol_high - profit_vol_low)
    })
    
    # Влияние постоянных затрат
    fixed_low = fixed_costs * (1 - variation)
    fixed_high = fixed_costs * (1 + variation)
    profit_fixed_low = base_volume * (base_price - base_var_cost) - fixed_high
    profit_fixed_high = base_volume * (base_price - base_var_cost) - fixed_low
    results.append({
        'parameter': 'Пост. затраты',
        'low_value': fixed_low,
        'high_value': fixed_high,
        'profit_low': profit_fixed_low,
        'profit_high': profit_fixed_high,
        'impact': abs(profit_fixed_high - profit_fixed_low)
    })
    
    # Сортируем по влиянию
    results.sort(key=lambda x: x['impact'], reverse=True)
    
    return {
        'base_profit': base_profit,
        'factors': results,
        'variation': variation
    }


def wafer_sensitivity_analysis(params):
    """Анализ чувствительности для вафельной линии."""
    base_volume = min(params['wafer_demand_base'], params['wafer_capacity'])
    base_price = params['wafer_price']
    base_var_cost = params['wafer_var_cost']
    fixed_costs = params['wafer_fixed_costs']
    
    # Базовая прибыль
    base_profit = base_volume * (base_price - base_var_cost) - fixed_costs
    
    # Диапазон изменения параметров (±30%)
    variation = 0.30
    
    results = []
    
    # Влияние цены
    price_low = base_price * (1 - variation)
    price_high = base_price * (1 + variation)
    profit_price_low = base_volume * (price_low - base_var_cost) - fixed_costs
    profit_price_high = base_volume * (price_high - base_var_cost) - fixed_costs
    results.append({
        'parameter': 'Цена',
        'low_value': price_low,
        'high_value': price_high,
        'profit_low': profit_price_low,
        'profit_high': profit_price_high,
        'impact': abs(profit_price_high - profit_price_low)
    })
    
    # Влияние переменных затрат
    var_cost_low = base_var_cost * (1 - variation)
    var_cost_high = base_var_cost * (1 + variation)
    profit_var_low = base_volume * (base_price - var_cost_high) - fixed_costs
    profit_var_high = base_volume * (base_price - var_cost_low) - fixed_costs
    results.append({
        'parameter': 'Перем. затраты',
        'low_value': var_cost_low,
        'high_value': var_cost_high,
        'profit_low': profit_var_low,
        'profit_high': profit_var_high,
        'impact': abs(profit_var_high - profit_var_low)
    })
    
    # Влияние спроса/объёма
    volume_low = base_volume * (1 - variation)
    volume_high = min(base_volume * (1 + variation), params['wafer_capacity'])
    profit_vol_low = volume_low * (base_price - base_var_cost) - fixed_costs
    profit_vol_high = volume_high * (base_price - base_var_cost) - fixed_costs
    results.append({
        'parameter': 'Объём продаж',
        'low_value': volume_low,
        'high_value': volume_high,
        'profit_low': profit_vol_low,
        'profit_high': profit_vol_high,
        'impact': abs(profit_vol_high - profit_vol_low)
    })
    
    # Влияние постоянных затрат
    fixed_low = fixed_costs * (1 - variation)
    fixed_high = fixed_costs * (1 + variation)
    profit_fixed_low = base_volume * (base_price - base_var_cost) - fixed_high
    profit_fixed_high = base_volume * (base_price - base_var_cost) - fixed_low
    results.append({
        'parameter': 'Пост. затраты',
        'low_value': fixed_low,
        'high_value': fixed_high,
        'profit_low': profit_fixed_low,
        'profit_high': profit_fixed_high,
        'impact': abs(profit_fixed_high - profit_fixed_low)
    })
    
    # Сортируем по влиянию
    results.sort(key=lambda x: x['impact'], reverse=True)
    
    return {
        'base_profit': base_profit,
        'factors': results,
        'variation': variation
    }


# ============================================================================
# ДИНАМИКА ПО ГОДАМ
# ============================================================================

def calculate_yearly_dynamics(params):
    """Рассчитывает динамику показателей по годам с учётом случайных факторов."""
    np.random.seed(42)  # Для воспроизводимости, но можно убрать для полного рандома
    
    years = params['horizon_years']
    initial_investment = params['initial_investment']
    demand_growth = params['demand_growth_rate']
    inflation = params['inflation_rate']
    price_growth = params['price_growth_rate']
    
    base_demand = params['demand_base']
    base_price = params['price_new']
    base_var_cost = params['var_cost_new']
    base_fixed_costs = params['fixed_costs']
    capacity = params['capacity_new_line']
    
    yearly_data = []
    cumulative_profit = -initial_investment  # Начинаем с инвестиций
    cumulative_cashflow = -initial_investment
    payback_year = None
    
    # Генерация случайных факторов на весь горизонт
    # 1. Волатильность спроса (±12% случайные колебания)
    demand_noise = np.random.normal(0, 0.08, years)  # стд. откл. 8%
    
    # 2. Экономический цикл (синусоида + шум)
    economic_cycle = 0.05 * np.sin(np.linspace(0, 2*np.pi, years)) + np.random.normal(0, 0.02, years)
    
    # 3. Конкурентное давление (случайные годы с ценовой войной)
    price_pressure = np.random.choice([0, -0.03, -0.05, 0.02], years, p=[0.6, 0.2, 0.1, 0.1])
    
    # 4. Непредвиденные расходы (ремонт, модернизация, штрафы)
    unexpected_costs = np.zeros(years)
    # В случайные годы добавляем непредвиденные расходы
    for i in range(years):
        if np.random.random() < 0.25:  # 25% шанс каждый год
            unexpected_costs[i] = base_fixed_costs * np.random.uniform(0.1, 0.3)
    
    # 5. Сезонный фактор (некоторые годы лучше/хуже)
    seasonal_factor = np.random.uniform(0.95, 1.08, years)
    
    for year in range(1, years + 1):
        idx = year - 1
        
        # Базовый рост параметров
        trend_demand = base_demand * ((1 + demand_growth) ** idx)
        trend_price = base_price * ((1 + price_growth) ** idx)
        trend_var_cost = base_var_cost * ((1 + inflation) ** idx)
        trend_fixed_costs = base_fixed_costs * ((1 + inflation) ** idx)
        
        # Применение случайных факторов
        # Спрос: тренд + волатильность + экономический цикл + сезонность
        demand = trend_demand * (1 + demand_noise[idx] + economic_cycle[idx]) * seasonal_factor[idx]
        demand = max(demand, base_demand * 0.5)  # Минимум 50% от базового
        
        # Цена: тренд + конкурентное давление
        price = trend_price * (1 + price_pressure[idx])
        
        # Затраты: тренд + небольшая волатильность
        var_cost = trend_var_cost * (1 + np.random.uniform(-0.02, 0.04))
        fixed_costs = trend_fixed_costs + unexpected_costs[idx]
        
        # Объём ограничен мощностью
        volume = min(demand, capacity)
        
        # Финансовые показатели
        revenue = volume * price
        total_var_costs = volume * var_cost
        gross_profit = revenue - total_var_costs
        net_profit = gross_profit - fixed_costs
        
        # Накопленные показатели
        cumulative_profit += net_profit
        cumulative_cashflow += net_profit
        
        # Срок окупаемости
        if payback_year is None and cumulative_cashflow >= 0:
            payback_year = year
        
        yearly_data.append({
            'year': year,
            'demand': demand,
            'volume': volume,
            'price': price,
            'var_cost': var_cost,
            'fixed_costs': fixed_costs,
            'revenue': revenue,
            'total_var_costs': total_var_costs,
            'gross_profit': gross_profit,
            'net_profit': net_profit,
            'cumulative_profit': cumulative_profit,
            'cumulative_cashflow': cumulative_cashflow,
            # Дополнительная информация для анализа
            'demand_factor': 1 + demand_noise[idx] + economic_cycle[idx],
            'price_factor': 1 + price_pressure[idx],
            'unexpected_costs': unexpected_costs[idx]
        })
    
    # Расчёт NPV (ставка дисконтирования = инфляция + 5% премия за риск)
    discount_rate = inflation + 0.05
    npv = -initial_investment
    for i, year_data in enumerate(yearly_data):
        npv += year_data['net_profit'] / ((1 + discount_rate) ** (i + 1))
    
    # Расчёт IRR (простое приближение)
    total_profit = sum(y['net_profit'] for y in yearly_data)
    if initial_investment > 0:
        simple_roi = (total_profit / initial_investment) * 100
        avg_annual_roi = simple_roi / years
    else:
        simple_roi = 0
        avg_annual_roi = 0
    
    return {
        'yearly_data': yearly_data,
        'initial_investment': initial_investment,
        'total_profit': total_profit,
        'npv': npv,
        'discount_rate': discount_rate,
        'payback_year': payback_year,
        'simple_roi': simple_roi,
        'avg_annual_roi': avg_annual_roi
    }


# ============================================================================
# СРАВНЕНИЕ ПРОДУКТОВЫХ ЛИНИЙ (МАРМЕЛАД vs ВАФЛИ)
# ============================================================================

def create_wafer_scenarios(params):
    """Создаёт словарь со всеми параметрами трёх сценариев для вафельной линии."""
    scenarios = {
        'Базовый': {
            'demand': params['wafer_demand_base'],
            'price': params['wafer_price'] * params['price_adj_base'],
            'var_cost': params['wafer_var_cost'] * params['cost_adj_base'],
            'description': 'Умеренный вход на рынок, стабильная конкуренция'
        },
        'Оптимистичный': {
            'demand': params['wafer_demand_opt'],
            'price': params['wafer_price'] * params['price_adj_opt'],
            'var_cost': params['wafer_var_cost'] * params['cost_adj_opt'],
            'description': 'Высокий спрос, рост цен'
        },
        'Пессимистичный': {
            'demand': params['wafer_demand_pess'],
            'price': params['wafer_price'] * params['price_adj_pess'],
            'var_cost': params['wafer_var_cost'] * params['cost_adj_pess'],
            'description': 'Низкий спрос, снижение цен'
        }
    }
    return scenarios


def calculate_wafer_indicators(scenarios, params):
    """Рассчитывает ключевые показатели для каждого сценария вафельной линии."""
    results = []
    
    for name, sc in scenarios.items():
        volume = min(sc['demand'], params['wafer_capacity'])
        revenue = volume * sc['price']
        var_costs = volume * sc['var_cost']
        gross_profit = revenue - var_costs
        net_profit = gross_profit - params['wafer_fixed_costs']
        margin = ((sc['price'] - sc['var_cost']) / sc['price']) * 100
        
        results.append({
            'Сценарий': name,
            'Спрос (т)': sc['demand'],
            'Объём выпуска (т)': volume,
            'Цена (руб./т)': sc['price'],
            'Перем. затраты (руб./т)': sc['var_cost'],
            'Выручка (руб.)': revenue,
            'Перем. затраты общие (руб.)': var_costs,
            'Валовая прибыль (руб.)': gross_profit,
            'Постоянные затраты (руб.)': params['wafer_fixed_costs'],
            'Чистая прибыль (руб.)': net_profit,
            'Маржинальность (%)': margin
        })
    
    return pd.DataFrame(results)


def wafer_monte_carlo_simulation(scenario_params, params, n_simulations=10000):
    """Выполняет имитационное моделирование методом Монте-Карло для вафельной линии."""
    np.random.seed(43)  # Другой seed для независимости
    
    base_demand = scenario_params['demand']
    base_price = scenario_params['price']
    base_var_cost = scenario_params['var_cost']
    variation = params['mc_variation']
    
    demand_low = base_demand * (1 - variation)
    demand_high = base_demand * (1 + variation)
    price_low = base_price * (1 - variation)
    price_high = base_price * (1 + variation)
    var_cost_low = base_var_cost * (1 - variation)
    var_cost_high = base_var_cost * (1 + variation)
    
    demands = np.random.triangular(demand_low, base_demand, demand_high, n_simulations)
    prices = np.random.triangular(price_low, base_price, price_high, n_simulations)
    var_costs = np.random.triangular(var_cost_low, base_var_cost, var_cost_high, n_simulations)
    
    volumes = np.minimum(demands, params['wafer_capacity'])
    revenues = volumes * prices
    total_var_costs = volumes * var_costs
    profits = revenues - total_var_costs - params['wafer_fixed_costs']
    
    stats = {
        'mean_profit': np.mean(profits),
        'std_profit': np.std(profits),
        'min_profit': np.min(profits),
        'max_profit': np.max(profits),
        'median_profit': np.median(profits),
        'percentile_5': np.percentile(profits, 5),
        'percentile_95': np.percentile(profits, 95),
        'prob_loss': np.mean(profits < 0) * 100,
    }
    
    return profits, stats


def run_wafer_monte_carlo_all_scenarios(scenarios, params):
    """Запускает Монте-Карло для всех сценариев вафельной линии."""
    mc_results = {}
    for name, sc_params in scenarios.items():
        profits, stats = wafer_monte_carlo_simulation(sc_params, params, params['mc_simulations'])
        mc_results[name] = {'profits': profits, 'stats': stats}
    return mc_results


def calculate_wafer_break_even(params):
    """Рассчитывает точку безубыточности для вафельной линии."""
    price = params['wafer_price']
    var_cost = params['wafer_var_cost']
    fixed_costs = params['wafer_fixed_costs']
    
    margin_per_unit = price - var_cost
    
    if margin_per_unit > 0:
        break_even_volume = fixed_costs / margin_per_unit
    else:
        break_even_volume = float('inf')
    
    break_even_revenue = break_even_volume * price
    
    base_volume = min(params['wafer_demand_base'], params['wafer_capacity'])
    if base_volume > 0:
        safety_margin = ((base_volume - break_even_volume) / base_volume) * 100
    else:
        safety_margin = 0
    
    coverage_ratio = (margin_per_unit / price) * 100 if price > 0 else 0
    
    return {
        'break_even_volume': break_even_volume,
        'break_even_revenue': break_even_revenue,
        'margin_per_unit': margin_per_unit,
        'safety_margin': safety_margin,
        'coverage_ratio': coverage_ratio,
        'base_volume': base_volume
    }


def calculate_wafer_yearly_dynamics(params):
    """Рассчитывает динамику показателей по годам для вафельной линии."""
    np.random.seed(43)
    
    years = params['horizon_years']
    initial_investment = params['wafer_initial_investment']
    demand_growth = params['demand_growth_rate']
    inflation = params['inflation_rate']
    price_growth = params['price_growth_rate']
    
    base_demand = params['wafer_demand_base']
    base_price = params['wafer_price']
    base_var_cost = params['wafer_var_cost']
    base_fixed_costs = params['wafer_fixed_costs']
    capacity = params['wafer_capacity']
    
    yearly_data = []
    cumulative_profit = -initial_investment
    cumulative_cashflow = -initial_investment
    payback_year = None
    
    demand_noise = np.random.normal(0, 0.08, years)
    economic_cycle = 0.05 * np.sin(np.linspace(0, 2*np.pi, years)) + np.random.normal(0, 0.02, years)
    price_pressure = np.random.choice([0, -0.03, -0.05, 0.02], years, p=[0.6, 0.2, 0.1, 0.1])
    unexpected_costs = np.zeros(years)
    for i in range(years):
        if np.random.random() < 0.25:
            unexpected_costs[i] = base_fixed_costs * np.random.uniform(0.1, 0.3)
    seasonal_factor = np.random.uniform(0.95, 1.08, years)
    
    for year in range(1, years + 1):
        idx = year - 1
        
        trend_demand = base_demand * ((1 + demand_growth) ** idx)
        trend_price = base_price * ((1 + price_growth) ** idx)
        trend_var_cost = base_var_cost * ((1 + inflation) ** idx)
        trend_fixed_costs = base_fixed_costs * ((1 + inflation) ** idx)
        
        demand = trend_demand * (1 + demand_noise[idx] + economic_cycle[idx]) * seasonal_factor[idx]
        demand = max(demand, base_demand * 0.5)
        
        price = trend_price * (1 + price_pressure[idx])
        var_cost = trend_var_cost * (1 + np.random.uniform(-0.02, 0.04))
        fixed_costs = trend_fixed_costs + unexpected_costs[idx]
        
        volume = min(demand, capacity)
        
        revenue = volume * price
        total_var_costs = volume * var_cost
        gross_profit = revenue - total_var_costs
        net_profit = gross_profit - fixed_costs
        
        cumulative_profit += net_profit
        cumulative_cashflow += net_profit
        
        if payback_year is None and cumulative_cashflow >= 0:
            payback_year = year
        
        yearly_data.append({
            'year': year,
            'demand': demand,
            'volume': volume,
            'price': price,
            'var_cost': var_cost,
            'fixed_costs': fixed_costs,
            'revenue': revenue,
            'total_var_costs': total_var_costs,
            'gross_profit': gross_profit,
            'net_profit': net_profit,
            'cumulative_profit': cumulative_profit,
            'cumulative_cashflow': cumulative_cashflow,
            'demand_factor': 1 + demand_noise[idx] + economic_cycle[idx],
            'price_factor': 1 + price_pressure[idx],
            'unexpected_costs': unexpected_costs[idx]
        })
    
    discount_rate = inflation + 0.05
    npv = -initial_investment
    for i, year_data in enumerate(yearly_data):
        npv += year_data['net_profit'] / ((1 + discount_rate) ** (i + 1))
    
    total_profit = sum(y['net_profit'] for y in yearly_data)
    if initial_investment > 0:
        simple_roi = (total_profit / initial_investment) * 100
        avg_annual_roi = simple_roi / years
    else:
        simple_roi = 0
        avg_annual_roi = 0
    
    return {
        'yearly_data': yearly_data,
        'initial_investment': initial_investment,
        'total_profit': total_profit,
        'npv': npv,
        'discount_rate': discount_rate,
        'payback_year': payback_year,
        'simple_roi': simple_roi,
        'avg_annual_roi': avg_annual_roi
    }


def compare_product_lines(marmalade_df, wafer_df, marmalade_mc, wafer_mc, 
                          marmalade_be, wafer_be, marmalade_dyn, wafer_dyn, params):
    """Создаёт сводное сравнение двух продуктовых линий."""
    
    # Базовые показатели
    marm_base = marmalade_df[marmalade_df['Сценарий'] == 'Базовый'].iloc[0]
    wafer_base = wafer_df[wafer_df['Сценарий'] == 'Базовый'].iloc[0]
    
    comparison = {
        'marmalade': {
            'name': 'Мармелад',
            'capacity': params['capacity_new_line'],
            'investment': params['initial_investment'],
            'base_profit': marm_base['Чистая прибыль (руб.)'],
            'base_revenue': marm_base['Выручка (руб.)'],
            'base_volume': marm_base['Объём выпуска (т)'],
            'margin': marm_base['Маржинальность (%)'],
            'break_even': marmalade_be['break_even_volume'],
            'safety_margin': marmalade_be['safety_margin'],
            'prob_loss': marmalade_mc['Базовый']['stats']['prob_loss'],
            'npv': marmalade_dyn['npv'],
            'roi': marmalade_dyn['simple_roi'],
            'payback': marmalade_dyn['payback_year'],
            'mc_mean': marmalade_mc['Базовый']['stats']['mean_profit'],
            'mc_std': marmalade_mc['Базовый']['stats']['std_profit'],
        },
        'wafer': {
            'name': 'Вафли',
            'capacity': params['wafer_capacity'],
            'investment': params['wafer_initial_investment'],
            'base_profit': wafer_base['Чистая прибыль (руб.)'],
            'base_revenue': wafer_base['Выручка (руб.)'],
            'base_volume': wafer_base['Объём выпуска (т)'],
            'margin': wafer_base['Маржинальность (%)'],
            'break_even': wafer_be['break_even_volume'],
            'safety_margin': wafer_be['safety_margin'],
            'prob_loss': wafer_mc['Базовый']['stats']['prob_loss'],
            'npv': wafer_dyn['npv'],
            'roi': wafer_dyn['simple_roi'],
            'payback': wafer_dyn['payback_year'],
            'mc_mean': wafer_mc['Базовый']['stats']['mean_profit'],
            'mc_std': wafer_mc['Базовый']['stats']['std_profit'],
        }
    }
    
    # Определяем лучший вариант по разным критериям
    comparison['best'] = {
        'profit': 'marmalade' if comparison['marmalade']['base_profit'] > comparison['wafer']['base_profit'] else 'wafer',
        'roi': 'marmalade' if comparison['marmalade']['roi'] > comparison['wafer']['roi'] else 'wafer',
        'risk': 'marmalade' if comparison['marmalade']['prob_loss'] < comparison['wafer']['prob_loss'] else 'wafer',
        'npv': 'marmalade' if comparison['marmalade']['npv'] > comparison['wafer']['npv'] else 'wafer',
        'payback': 'marmalade' if (comparison['marmalade']['payback'] or 999) < (comparison['wafer']['payback'] or 999) else 'wafer',
    }
    
    return comparison


# ============================================================================
# ФУНКЦИИ ПОСТРОЕНИЯ ГРАФИКОВ (возвращают base64)
# ============================================================================

def fig_to_base64(fig):
    """Конвертирует matplotlib figure в base64 строку."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def plot_scenario_comparison(df, params):
    """Строит столбчатые диаграммы сравнения выручки и прибыли."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scenarios = df['Сценарий'].tolist()
    x = np.arange(len(scenarios))
    width = 0.5
    
    ax1 = axes[0]
    revenues = df['Выручка (руб.)'].values / 1_000_000
    var_costs = df['Перем. затраты общие (руб.)'].values / 1_000_000
    fixed = params['fixed_costs'] / 1_000_000
    
    bars1 = ax1.bar(x - width/3, revenues, width/1.5, label='Выручка', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/3, var_costs + fixed, width/1.5, label='Общие затраты', color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('Сценарий', fontsize=12)
    ax1.set_ylabel('Сумма (млн руб.)', fontsize=12)
    ax1.set_title('Сравнение выручки и затрат по сценариям', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    
    ax2 = axes[1]
    profits = df['Чистая прибыль (руб.)'].values / 1_000_000
    bar_colors = ['#2ecc71' if p >= 0 else '#e74c3c' for p in profits]
    bars3 = ax2.bar(scenarios, profits, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Сценарий', fontsize=12)
    ax2.set_ylabel('Чистая прибыль (млн руб.)', fontsize=12)
    ax2.set_title('Чистая прибыль по сценариям', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, profit in zip(bars3, profits):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 3 if height >= 0 else -10
        ax2.annotate(f'{profit:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset), textcoords="offset points", ha='center', va=va, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_monte_carlo_histogram(mc_results, scenario_name='Базовый', title_prefix=''):
    """Строит гистограмму распределения прибыли."""
    profits = mc_results[scenario_name]['profits'] / 1_000_000
    stats = mc_results[scenario_name]['stats']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    n, bins, patches = ax.hist(profits, bins=50, edgecolor='white', alpha=0.7)
    
    for i, patch in enumerate(patches):
        if bins[i] < 0:
            patch.set_facecolor('#e74c3c')
        else:
            patch.set_facecolor('#2ecc71')
    
    mean_profit = stats['mean_profit'] / 1_000_000
    ax.axvline(mean_profit, color='#2980b9', linestyle='--', linewidth=2, 
               label=f'Средняя: {mean_profit:.2f} млн')
    ax.axvline(0, color='black', linestyle='-', linewidth=1.5, label='Точка безубыточности')
    ax.axvspan(ax.get_xlim()[0], 0, alpha=0.1, color='red')
    
    ax.set_xlabel('Чистая прибыль (млн руб.)', fontsize=12)
    ax.set_ylabel('Частота', fontsize=12)
    ax.set_title(f'{title_prefix}Распределение прибыли (Монте-Карло)\nСценарий: {scenario_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    textstr = '\n'.join([
        f'Средняя прибыль: {stats["mean_profit"]/1e6:.2f} млн',
        f'Стд. отклонение: {stats["std_profit"]/1e6:.2f} млн',
        f'Мин / Макс: {stats["min_profit"]/1e6:.2f} / {stats["max_profit"]/1e6:.2f} млн',
        f'5% / 95%: {stats["percentile_5"]/1e6:.2f} / {stats["percentile_95"]/1e6:.2f} млн',
        f'Вероятность убытка: {stats["prob_loss"]:.1f}%'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig_to_base64(fig)


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
    return fig_to_base64(fig)


def plot_all_monte_carlo(mc_results):
    """Строит сравнительные boxplot для всех сценариев."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = [mc_results[name]['profits'] / 1_000_000 for name in ['Базовый', 'Оптимистичный', 'Пессимистичный']]
    labels = ['Базовый', 'Оптимистичный', 'Пессимистичный']
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Точка безубыточности')
    ax.set_xlabel('Сценарий', fontsize=12)
    ax.set_ylabel('Чистая прибыль (млн руб.)', fontsize=12)
    ax.set_title('Сравнение распределений прибыли по сценариям (Монте-Карло)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_break_even(break_even, params):
    """Строит график точки безубыточности."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Диапазон объёмов
    max_volume = params['capacity_new_line'] * 1.2
    volumes = np.linspace(0, max_volume, 100)
    
    # Линии
    revenues = volumes * params['price_new']
    total_costs = volumes * params['var_cost_new'] + params['fixed_costs']
    fixed_line = np.full_like(volumes, params['fixed_costs'])
    
    ax.plot(volumes, revenues / 1e6, 'b-', linewidth=2.5, label='Выручка')
    ax.plot(volumes, total_costs / 1e6, 'r-', linewidth=2.5, label='Общие затраты')
    ax.plot(volumes, fixed_line / 1e6, 'r--', linewidth=1.5, alpha=0.7, label='Постоянные затраты')
    
    # Точка безубыточности
    be_vol = break_even['break_even_volume']
    be_rev = break_even['break_even_revenue']
    ax.plot(be_vol, be_rev / 1e6, 'go', markersize=15, zorder=5, label=f'Точка безубыточности: {be_vol:.0f} т')
    
    # Заливка зон прибыли/убытка
    ax.fill_between(volumes, revenues / 1e6, total_costs / 1e6, 
                    where=(revenues >= total_costs), color='green', alpha=0.2, label='Зона прибыли')
    ax.fill_between(volumes, revenues / 1e6, total_costs / 1e6, 
                    where=(revenues < total_costs), color='red', alpha=0.2, label='Зона убытка')
    
    # Вертикальная линия базового объёма
    base_vol = break_even['base_volume']
    ax.axvline(base_vol, color='#9b59b6', linestyle='--', linewidth=2, 
               label=f'Базовый объём: {base_vol:.0f} т')
    
    # Вертикальная линия мощности
    ax.axvline(params['capacity_new_line'], color='orange', linestyle=':', linewidth=2, 
               label=f'Мощность: {params["capacity_new_line"]:.0f} т')
    
    ax.set_xlabel('Объём продаж (тонн)', fontsize=12)
    ax.set_ylabel('Сумма (млн руб.)', fontsize=12)
    ax.set_title('Анализ точки безубыточности', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_volume)
    ax.set_ylim(0, None)
    
    # Текстовый блок
    textstr = '\n'.join([
        f'Точка безубыточности: {be_vol:.0f} т',
        f'Выручка в ТБ: {be_rev/1e6:.2f} млн',
        f'Маржа на тонну: {break_even["margin_per_unit"]:,.0f} руб.',
        f'Запас прочности: {break_even["safety_margin"]:.1f}%'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_tornado(sensitivity, title_prefix=''):
    """Строит торнадо-диаграмму анализа чувствительности."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    factors = sensitivity['factors']
    base_profit = sensitivity['base_profit'] / 1e6
    
    y_pos = np.arange(len(factors))
    labels = [f['parameter'] for f in factors]
    
    # Отклонения от базовой прибыли
    left_bars = [(f['profit_low'] - sensitivity['base_profit']) / 1e6 for f in factors]
    right_bars = [(f['profit_high'] - sensitivity['base_profit']) / 1e6 for f in factors]
    
    # Рисуем бары
    for i, (left, right, factor) in enumerate(zip(left_bars, right_bars, factors)):
        # Определяем цвета (красный для негативного, зелёный для позитивного)
        ax.barh(i, left, height=0.6, color='#e74c3c', alpha=0.8, left=0)
        ax.barh(i, right, height=0.6, color='#2ecc71', alpha=0.8, left=0)
    
    # Базовая линия
    ax.axvline(0, color='black', linewidth=2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Изменение прибыли (млн руб.)', fontsize=12)
    ax.set_title(f'{title_prefix}Анализ чувствительности (±{sensitivity["variation"]*100:.0f}%)\n'
                f'Базовая прибыль: {base_profit:.2f} млн руб.', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Легенда
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', alpha=0.8, label='Снижение параметра'),
                      Patch(facecolor='#2ecc71', alpha=0.8, label='Увеличение параметра')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_yearly_dynamics(dynamics):
    """Строит графики динамики по годам с учётом случайных факторов."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    years = [d['year'] for d in dynamics['yearly_data']]
    
    # График 1: Выручка и затраты (с областью прибыли)
    ax1 = axes[0, 0]
    revenues = [d['revenue'] / 1e6 for d in dynamics['yearly_data']]
    total_costs = [(d['total_var_costs'] + d['fixed_costs']) / 1e6 for d in dynamics['yearly_data']]
    
    # Линии с маркерами
    ax1.plot(years, revenues, 'b-o', linewidth=2.5, markersize=10, label='Выручка', zorder=3)
    ax1.plot(years, total_costs, 'r-s', linewidth=2.5, markersize=10, label='Общие затраты', zorder=3)
    
    # Заливка между линиями (прибыль)
    for i in range(len(years)):
        color = '#2ecc71' if revenues[i] > total_costs[i] else '#e74c3c'
        if i < len(years) - 1:
            ax1.fill_between([years[i], years[i+1]], 
                           [revenues[i], revenues[i+1]], 
                           [total_costs[i], total_costs[i+1]], 
                           alpha=0.25, color=color)
    
    ax1.set_xlabel('Год', fontsize=11)
    ax1.set_ylabel('Сумма (млн руб.)', fontsize=11)
    ax1.set_title('📊 Выручка vs Затраты', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(years)
    
    # График 2: Чистая прибыль с индикаторами факторов
    ax2 = axes[0, 1]
    profits = [d['net_profit'] / 1e6 for d in dynamics['yearly_data']]
    unexpected = [d.get('unexpected_costs', 0) / 1e6 for d in dynamics['yearly_data']]
    
    # Бары прибыли
    colors = ['#27ae60' if p >= 0 else '#c0392b' for p in profits]
    bars = ax2.bar(years, profits, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
    ax2.axhline(0, color='black', linewidth=1.5)
    
    # Отметки непредвиденных расходов
    for i, (year, unexp) in enumerate(zip(years, unexpected)):
        if unexp > 0:
            ax2.annotate('⚠️', xy=(year, profits[i]), 
                        xytext=(0, -25 if profits[i] >= 0 else 10), 
                        textcoords='offset points', ha='center', fontsize=14)
    
    ax2.set_xlabel('Год', fontsize=11)
    ax2.set_ylabel('Чистая прибыль (млн руб.)', fontsize=11)
    ax2.set_title('💰 Чистая прибыль (⚠️ = непредв. расходы)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_xticks(years)
    
    # Значения над барами
    for year, profit in zip(years, profits):
        offset = 8 if profit >= 0 else -15
        ax2.annotate(f'{profit:.2f}', xy=(year, profit), 
                    xytext=(0, offset), textcoords='offset points',
                    ha='center', fontsize=10, fontweight='bold',
                    color='darkgreen' if profit >= 0 else 'darkred')
    
    # График 3: Накопленный денежный поток
    ax3 = axes[1, 0]
    cumulative = [d['cumulative_cashflow'] / 1e6 for d in dynamics['yearly_data']]
    years_with_zero = [0] + years
    cumulative_with_zero = [-dynamics['initial_investment'] / 1e6] + cumulative
    
    # Градиентная линия
    ax3.plot(years_with_zero, cumulative_with_zero, color='#9b59b6', linewidth=3, 
             marker='o', markersize=10, markerfacecolor='white', markeredgewidth=2, zorder=3)
    
    # Заливка зон
    ax3.fill_between(years_with_zero, cumulative_with_zero, 0, 
                    where=[c >= 0 for c in cumulative_with_zero], 
                    color='#2ecc71', alpha=0.4, label='Зона прибыли')
    ax3.fill_between(years_with_zero, cumulative_with_zero, 0, 
                    where=[c < 0 for c in cumulative_with_zero], 
                    color='#e74c3c', alpha=0.4, label='Зона убытка')
    ax3.axhline(0, color='black', linewidth=2, linestyle='-')
    
    # Срок окупаемости
    if dynamics['payback_year']:
        ax3.axvline(dynamics['payback_year'], color='#27ae60', linewidth=2.5, linestyle='--',
                   label=f'Окупаемость: год {dynamics["payback_year"]}')
    
    ax3.set_xlabel('Год', fontsize=11)
    ax3.set_ylabel('Накопленный CF (млн руб.)', fontsize=11)
    ax3.set_title('📈 Окупаемость инвестиций', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xticks(years_with_zero)
    
    # График 4: Объём продаж и факторы влияния
    ax4 = axes[1, 1]
    volumes = [d['volume'] for d in dynamics['yearly_data']]
    demands = [d['demand'] for d in dynamics['yearly_data']]
    demand_factors = [d.get('demand_factor', 1) for d in dynamics['yearly_data']]
    price_factors = [d.get('price_factor', 1) for d in dynamics['yearly_data']]
    
    # Объём продаж (бары)
    ax4.bar(years, volumes, color='#3498db', alpha=0.7, label='Объём продаж (т)', edgecolor='#2980b9', linewidth=1.5)
    
    # Спрос (линия)
    ax4.plot(years, demands, 'g--', linewidth=2, marker='D', markersize=6, label='Спрос (т)')
    
    # Вторая ось для факторов
    ax4_twin = ax4.twinx()
    ax4_twin.plot(years, demand_factors, 'orange', linewidth=2, marker='^', markersize=6, 
                  label='Фактор спроса', alpha=0.8)
    ax4_twin.axhline(1.0, color='gray', linewidth=1, linestyle=':')
    ax4_twin.set_ylabel('Фактор влияния', fontsize=10, color='orange')
    ax4_twin.tick_params(axis='y', labelcolor='orange')
    ax4_twin.set_ylim(0.8, 1.2)
    
    ax4.set_xlabel('Год', fontsize=11)
    ax4.set_ylabel('Объём / Спрос (т)', fontsize=11)
    ax4.set_title('📦 Объём продаж и рыночные факторы', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4_twin.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xticks(years)
    
    plt.tight_layout()
    return fig_to_base64(fig)


# ============================================================================
# ГРАФИКИ СРАВНЕНИЯ ПРОДУКТОВЫХ ЛИНИЙ
# ============================================================================

def plot_lines_profit_comparison(comparison):
    """Строит сравнительную диаграмму прибыли и выручки двух линий."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    products = ['Мармелад', 'Вафли']
    x = np.arange(len(products))
    width = 0.35
    
    # График 1: Прибыль по сценариям
    ax1 = axes[0]
    profits = [comparison['marmalade']['base_profit'] / 1e6, 
               comparison['wafer']['base_profit'] / 1e6]
    colors = ['#3498db', '#e67e22']
    bars = ax1.bar(products, profits, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_ylabel('Чистая прибыль (млн руб.)', fontsize=12)
    ax1.set_title('Сравнение прибыли (базовый сценарий)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, profit in zip(bars, profits):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 5 if height >= 0 else -15
        color = '#27ae60' if height >= 0 else '#c0392b'
        ax1.annotate(f'{profit:.2f} млн', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset), textcoords="offset points", ha='center', va=va,
                    fontsize=12, fontweight='bold', color=color)
    
    # График 2: Инвестиции vs NPV
    ax2 = axes[1]
    investments = [comparison['marmalade']['investment'] / 1e6, 
                   comparison['wafer']['investment'] / 1e6]
    npvs = [comparison['marmalade']['npv'] / 1e6, 
            comparison['wafer']['npv'] / 1e6]
    
    x_pos = np.arange(len(products))
    bars1 = ax2.bar(x_pos - width/2, investments, width, label='Инвестиции', color='#e74c3c', alpha=0.8)
    bars2 = ax2.bar(x_pos + width/2, npvs, width, label='NPV', color='#2ecc71', alpha=0.8)
    
    ax2.set_ylabel('Сумма (млн руб.)', fontsize=12)
    ax2.set_title('Инвестиции vs NPV', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(products)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(0, color='black', linewidth=1)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.1:
                ax2.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3 if height >= 0 else -12), textcoords="offset points",
                            ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_lines_risk_comparison(comparison, marmalade_mc, wafer_mc):
    """Строит сравнительную диаграмму рисков двух линий."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # График 1: Вероятность убытка по сценариям
    ax1 = axes[0]
    scenarios = ['Базовый', 'Оптимистичный', 'Пессимистичный']
    x = np.arange(len(scenarios))
    width = 0.35
    
    marm_probs = [marmalade_mc[s]['stats']['prob_loss'] for s in scenarios]
    wafer_probs = [wafer_mc[s]['stats']['prob_loss'] for s in scenarios]
    
    bars1 = ax1.bar(x - width/2, marm_probs, width, label='Мармелад', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, wafer_probs, width, label='Вафли', color='#e67e22', alpha=0.8)
    
    ax1.set_ylabel('Вероятность убытка (%)', fontsize=12)
    ax1.set_title('Сравнение рисков по сценариям', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Зона высокого риска
    ax1.axhspan(30, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 30 else 100, 
                alpha=0.1, color='red', label='Высокий риск')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    # График 2: Boxplot распределения прибыли
    ax2 = axes[1]
    data = [
        marmalade_mc['Базовый']['profits'] / 1e6,
        wafer_mc['Базовый']['profits'] / 1e6
    ]
    labels = ['Мармелад', 'Вафли']
    
    bp = ax2.boxplot(data, labels=labels, patch_artist=True)
    colors = ['#3498db', '#e67e22']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Точка безубыточности')
    ax2.set_ylabel('Чистая прибыль (млн руб.)', fontsize=12)
    ax2.set_title('Распределение прибыли (Монте-Карло)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_lines_roi_comparison(comparison, marmalade_dyn, wafer_dyn):
    """Строит сравнительную диаграмму ROI и окупаемости."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    products = ['Мармелад', 'Вафли']
    colors = ['#3498db', '#e67e22']
    
    # График 1: ROI и маржинальность
    ax1 = axes[0]
    x = np.arange(len(products))
    width = 0.35
    
    roi = [comparison['marmalade']['roi'], comparison['wafer']['roi']]
    margin = [comparison['marmalade']['margin'], comparison['wafer']['margin']]
    
    bars1 = ax1.bar(x - width/2, roi, width, label='ROI (%)', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, margin, width, label='Маржинальность (%)', color='#9b59b6', alpha=0.8)
    
    ax1.set_ylabel('Процент (%)', fontsize=12)
    ax1.set_title('ROI и маржинальность', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(products)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
    
    # График 2: Накопленный денежный поток
    ax2 = axes[1]
    years = [0] + [d['year'] for d in marmalade_dyn['yearly_data']]
    
    marm_cf = [-marmalade_dyn['initial_investment'] / 1e6] + \
              [d['cumulative_cashflow'] / 1e6 for d in marmalade_dyn['yearly_data']]
    wafer_cf = [-wafer_dyn['initial_investment'] / 1e6] + \
               [d['cumulative_cashflow'] / 1e6 for d in wafer_dyn['yearly_data']]
    
    ax2.plot(years, marm_cf, 'o-', color='#3498db', linewidth=2.5, markersize=8, label='Мармелад')
    ax2.plot(years, wafer_cf, 's-', color='#e67e22', linewidth=2.5, markersize=8, label='Вафли')
    
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5)
    ax2.fill_between(years, marm_cf, 0, where=[c >= 0 for c in marm_cf], color='#3498db', alpha=0.1)
    ax2.fill_between(years, wafer_cf, 0, where=[c >= 0 for c in wafer_cf], color='#e67e22', alpha=0.1)
    
    # Отметка окупаемости
    if marmalade_dyn['payback_year']:
        ax2.axvline(marmalade_dyn['payback_year'], color='#3498db', linestyle=':', alpha=0.7)
    if wafer_dyn['payback_year']:
        ax2.axvline(wafer_dyn['payback_year'], color='#e67e22', linestyle=':', alpha=0.7)
    
    ax2.set_xlabel('Год', fontsize=12)
    ax2.set_ylabel('Накопленный CF (млн руб.)', fontsize=12)
    ax2.set_title('Динамика окупаемости', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(years)
    
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_lines_break_even_comparison(marmalade_be, wafer_be, params):
    """Строит сравнительный график точек безубыточности."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Диапазон объёмов (общий для обеих линий)
    max_volume = max(params['capacity_new_line'], params['wafer_capacity']) * 1.2
    volumes = np.linspace(0, max_volume, 100)
    
    # Мармелад
    marm_revenues = volumes * params['price_new']
    marm_costs = volumes * params['var_cost_new'] + params['fixed_costs']
    
    # Вафли  
    wafer_revenues = volumes * params['wafer_price']
    wafer_costs = volumes * params['wafer_var_cost'] + params['wafer_fixed_costs']
    
    # Линии мармелада
    ax.plot(volumes, marm_revenues / 1e6, 'b-', linewidth=2, label='Выручка (мармелад)')
    ax.plot(volumes, marm_costs / 1e6, 'b--', linewidth=2, label='Затраты (мармелад)', alpha=0.7)
    
    # Линии вафель
    ax.plot(volumes, wafer_revenues / 1e6, color='#e67e22', linestyle='-', linewidth=2, label='Выручка (вафли)')
    ax.plot(volumes, wafer_costs / 1e6, color='#e67e22', linestyle='--', linewidth=2, label='Затраты (вафли)', alpha=0.7)
    
    # Точки безубыточности
    ax.plot(marmalade_be['break_even_volume'], marmalade_be['break_even_revenue'] / 1e6, 
            'bo', markersize=15, zorder=5, label=f'ТБ мармелад: {marmalade_be["break_even_volume"]:.0f} т')
    ax.plot(wafer_be['break_even_volume'], wafer_be['break_even_revenue'] / 1e6, 
            'o', color='#e67e22', markersize=15, zorder=5, label=f'ТБ вафли: {wafer_be["break_even_volume"]:.0f} т')
    
    # Мощности
    ax.axvline(params['capacity_new_line'], color='#3498db', linestyle=':', linewidth=2, alpha=0.5)
    ax.axvline(params['wafer_capacity'], color='#e67e22', linestyle=':', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Объём продаж (тонн)', fontsize=12)
    ax.set_ylabel('Сумма (млн руб.)', fontsize=12)
    ax.set_title('Сравнение точек безубыточности', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_volume)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    return fig_to_base64(fig)


# ============================================================================
# МАРШРУТЫ FLASK
# ============================================================================

def get_default_params():
    """Возвращает параметры по умолчанию."""
    return {
        # Конкурент (мармелад)
        'competitor_capacity': 3924,
        'competitor_output': 2624,
        # Коммунарка - линия мармелада
        'capacity_new_line': 1000,
        'price_new': 8000,
        'var_cost_new': 4800,
        'fixed_costs': 500000,
        # Сценарии спроса (мармелад)
        'demand_base': 600,
        'demand_opt': 900,
        'demand_pess': 350,
        # Корректировки цены (множители)
        'price_adj_base': 1.00,
        'price_adj_opt': 1.05,
        'price_adj_pess': 0.95,
        # Корректировки затрат (множители)
        'cost_adj_base': 1.00,
        'cost_adj_opt': 0.95,
        'cost_adj_pess': 1.05,
        # Монте-Карло
        'mc_variation': 0.30,
        'mc_simulations': 50000,
        # Динамика по годам (мармелад)
        'horizon_years': 5,
        'initial_investment': 36_000_000,  # Инвестиции в линию (нижние границы)
        'demand_growth_rate': 0.05,        # Рост спроса 5% в год
        'inflation_rate': 0.072,           # Инфляция 7.2%
        'price_growth_rate': 0.072,        # Рост цены (соразмерно инфляции)
        
        # === ВАФЕЛЬНАЯ ЛИНИЯ ===
        # Производственные параметры
        'wafer_capacity': 800,             # Мощность линии (т/год)
        'wafer_price': 6200,               # Отпускная цена (руб./т)
        'wafer_var_cost': 4600,            # Переменные затраты (руб./т)
        'wafer_fixed_costs': 430000,       # Постоянные затраты (руб./год)
        # Сценарии спроса (вафли)
        'wafer_demand_base': 500,          # 62.5% от мощности
        'wafer_demand_opt': 650,           # 81.3% от мощности
        'wafer_demand_pess': 350,          # 43.8% от мощности
        # Инвестиции (вафли)
        'wafer_initial_investment': 20_000_000,  # Инвестиции в реконструкцию
    }


def get_params_from_form(form):
    """Извлекает параметры из формы."""
    params = get_default_params()
    int_keys = ['competitor_capacity', 'competitor_output', 'capacity_new_line', 
                'demand_base', 'demand_opt', 'demand_pess', 'mc_simulations',
                'horizon_years', 'initial_investment',
                # Вафельная линия
                'wafer_capacity', 'wafer_demand_base', 'wafer_demand_opt', 
                'wafer_demand_pess', 'wafer_initial_investment']
    for key in params:
        if key in form:
            try:
                params[key] = float(form[key])
                if key in int_keys:
                    params[key] = int(params[key])
            except ValueError:
                pass
    return params


def generate_recommendation(df, mc_results, params):
    """Генерирует текстовую рекомендацию."""
    base_stats = mc_results['Базовый']['stats']
    base_profit = df[df['Сценарий'] == 'Базовый']['Чистая прибыль (руб.)'].values[0]
    opt_profit = df[df['Сценарий'] == 'Оптимистичный']['Чистая прибыль (руб.)'].values[0]
    pess_profit = df[df['Сценарий'] == 'Пессимистичный']['Чистая прибыль (руб.)'].values[0]
    
    if base_profit > 0 and base_stats['prob_loss'] < 20:
        status = 'success'
        recommendation = 'РЕКОМЕНДУЕТСЯ'
        reason = 'положительную прибыль в базовом сценарии и приемлемый уровень риска'
    elif base_profit > 0 and base_stats['prob_loss'] < 40:
        status = 'warning'
        recommendation = 'РЕКОМЕНДУЕТСЯ С ОГОВОРКАМИ'
        reason = 'положительную прибыль, но повышенный уровень риска'
    else:
        status = 'danger'
        recommendation = 'НЕ РЕКОМЕНДУЕТСЯ'
        reason = 'высокий риск убытков'
    
    margin = (params['price_new'] - params['var_cost_new']) / params['price_new'] * 100
    
    return {
        'status': status,
        'recommendation': recommendation,
        'reason': reason,
        'base_profit': base_profit,
        'opt_profit': opt_profit,
        'pess_profit': pess_profit,
        'prob_loss_base': base_stats['prob_loss'],
        'prob_loss_pess': mc_results['Пессимистичный']['stats']['prob_loss'],
        'margin': margin,
        'optimal_load': min(params['demand_base'], params['capacity_new_line']) / params['capacity_new_line'] * 100
    }


@app.route('/', methods=['GET', 'POST'])
def index():
    """Главная страница."""
    if request.method == 'POST':
        params = get_params_from_form(request.form)
    else:
        params = get_default_params()
    
    # === РАСЧЁТЫ ДЛЯ МАРМЕЛАДА ===
    scenarios = create_scenarios(params)
    df = calculate_indicators(scenarios, params)
    mc_results = run_monte_carlo_all_scenarios(scenarios, params)
    opt_results = run_optimization_all_scenarios(scenarios, params)
    break_even = calculate_break_even(params)
    sensitivity = sensitivity_analysis(params)
    wafer_sensitivity = wafer_sensitivity_analysis(params)
    dynamics = calculate_yearly_dynamics(params)
    
    # === РАСЧЁТЫ ДЛЯ ВАФЕЛЬНОЙ ЛИНИИ ===
    wafer_scenarios = create_wafer_scenarios(params)
    wafer_df = calculate_wafer_indicators(wafer_scenarios, params)
    wafer_mc_results = run_wafer_monte_carlo_all_scenarios(wafer_scenarios, params)
    wafer_break_even = calculate_wafer_break_even(params)
    wafer_dynamics = calculate_wafer_yearly_dynamics(params)
    
    # === СРАВНЕНИЕ ЛИНИЙ ===
    lines_comparison = compare_product_lines(
        df, wafer_df, mc_results, wafer_mc_results,
        break_even, wafer_break_even, dynamics, wafer_dynamics, params
    )
    
    # === ГРАФИКИ МАРМЕЛАДА ===
    chart_comparison = plot_scenario_comparison(df, params)
    chart_mc_base = plot_monte_carlo_histogram(mc_results, 'Базовый')
    chart_mc_opt = plot_monte_carlo_histogram(mc_results, 'Оптимистичный')
    chart_mc_pess = plot_monte_carlo_histogram(mc_results, 'Пессимистичный')
    chart_market_share = plot_market_share(df)
    chart_mc_comparison = plot_all_monte_carlo(mc_results)
    chart_break_even = plot_break_even(break_even, params)
    chart_tornado = plot_tornado(sensitivity)
    chart_wafer_tornado = plot_tornado(wafer_sensitivity, title_prefix='Вафли: ')
    chart_dynamics = plot_yearly_dynamics(dynamics)
    
    # === ГРАФИКИ ВАФЕЛЬНОЙ ЛИНИИ ===
    chart_wafer_mc_base = plot_monte_carlo_histogram(wafer_mc_results, 'Базовый', title_prefix='Вафли: ')
    chart_wafer_mc_opt = plot_monte_carlo_histogram(wafer_mc_results, 'Оптимистичный', title_prefix='Вафли: ')
    chart_wafer_mc_pess = plot_monte_carlo_histogram(wafer_mc_results, 'Пессимистичный', title_prefix='Вафли: ')
    
    # === ГРАФИКИ СРАВНЕНИЯ ЛИНИЙ ===
    chart_lines_profit = plot_lines_profit_comparison(lines_comparison)
    chart_lines_risk = plot_lines_risk_comparison(lines_comparison, mc_results, wafer_mc_results)
    chart_lines_roi = plot_lines_roi_comparison(lines_comparison, dynamics, wafer_dynamics)
    chart_lines_break_even = plot_lines_break_even_comparison(break_even, wafer_break_even, params)
    
    # Рекомендация
    recommendation = generate_recommendation(df, mc_results, params)
    
    # Форматирование таблицы
    df_display = df.copy()
    df_display['Цена (руб./т)'] = df_display['Цена (руб./т)'].apply(lambda x: f'{x:,.0f}')
    df_display['Перем. затраты (руб./т)'] = df_display['Перем. затраты (руб./т)'].apply(lambda x: f'{x:,.0f}')
    df_display['Выручка (руб.)'] = df_display['Выручка (руб.)'].apply(lambda x: f'{x:,.0f}')
    df_display['Перем. затраты общие (руб.)'] = df_display['Перем. затраты общие (руб.)'].apply(lambda x: f'{x:,.0f}')
    df_display['Валовая прибыль (руб.)'] = df_display['Валовая прибыль (руб.)'].apply(lambda x: f'{x:,.0f}')
    df_display['Постоянные затраты (руб.)'] = df_display['Постоянные затраты (руб.)'].apply(lambda x: f'{x:,.0f}')
    df_display['Чистая прибыль (руб.)'] = df_display['Чистая прибыль (руб.)'].apply(lambda x: f'{x:,.0f}')
    df_display['Доля рынка (%)'] = df_display['Доля рынка (%)'].apply(lambda x: f'{x:.1f}%')
    df_display['Маржинальность (%)'] = df_display['Маржинальность (%)'].apply(lambda x: f'{x:.1f}%')
    
    # MC статистика для таблицы
    mc_stats_table = []
    for name in ['Базовый', 'Оптимистичный', 'Пессимистичный']:
        stats = mc_results[name]['stats']
        mc_stats_table.append({
            'scenario': name,
            'mean': f"{stats['mean_profit']/1e6:.2f} млн",
            'std': f"{stats['std_profit']/1e6:.2f} млн",
            'min': f"{stats['min_profit']/1e6:.2f} млн",
            'max': f"{stats['max_profit']/1e6:.2f} млн",
            'p5': f"{stats['percentile_5']/1e6:.2f} млн",
            'p95': f"{stats['percentile_95']/1e6:.2f} млн",
            'prob_loss': f"{stats['prob_loss']:.1f}%"
        })
    
    # Форматирование таблицы вафельной линии
    wafer_df_display = wafer_df.copy()
    wafer_df_display['Цена (руб./т)'] = wafer_df_display['Цена (руб./т)'].apply(lambda x: f'{x:,.0f}')
    wafer_df_display['Перем. затраты (руб./т)'] = wafer_df_display['Перем. затраты (руб./т)'].apply(lambda x: f'{x:,.0f}')
    wafer_df_display['Выручка (руб.)'] = wafer_df_display['Выручка (руб.)'].apply(lambda x: f'{x:,.0f}')
    wafer_df_display['Перем. затраты общие (руб.)'] = wafer_df_display['Перем. затраты общие (руб.)'].apply(lambda x: f'{x:,.0f}')
    wafer_df_display['Валовая прибыль (руб.)'] = wafer_df_display['Валовая прибыль (руб.)'].apply(lambda x: f'{x:,.0f}')
    wafer_df_display['Постоянные затраты (руб.)'] = wafer_df_display['Постоянные затраты (руб.)'].apply(lambda x: f'{x:,.0f}')
    wafer_df_display['Чистая прибыль (руб.)'] = wafer_df_display['Чистая прибыль (руб.)'].apply(lambda x: f'{x:,.0f}')
    wafer_df_display['Маржинальность (%)'] = wafer_df_display['Маржинальность (%)'].apply(lambda x: f'{x:.1f}%')
    
    # MC статистика для вафельной линии
    wafer_mc_stats_table = []
    for name in ['Базовый', 'Оптимистичный', 'Пессимистичный']:
        stats = wafer_mc_results[name]['stats']
        wafer_mc_stats_table.append({
            'scenario': name,
            'mean': f"{stats['mean_profit']/1e6:.2f} млн",
            'std': f"{stats['std_profit']/1e6:.2f} млн",
            'min': f"{stats['min_profit']/1e6:.2f} млн",
            'max': f"{stats['max_profit']/1e6:.2f} млн",
            'p5': f"{stats['percentile_5']/1e6:.2f} млн",
            'p95': f"{stats['percentile_95']/1e6:.2f} млн",
            'prob_loss': f"{stats['prob_loss']:.1f}%"
        })
    
    return render_template('index.html',
                          params=params,
                          # Мармелад
                          df=df_display.to_dict('records'),
                          df_columns=df_display.columns.tolist(),
                          opt_results=opt_results,
                          mc_stats=mc_stats_table,
                          recommendation=recommendation,
                          break_even=break_even,
                          sensitivity=sensitivity,
                          wafer_sensitivity=wafer_sensitivity,
                          dynamics=dynamics,
                          # Вафли
                          wafer_df=wafer_df_display.to_dict('records'),
                          wafer_df_columns=wafer_df_display.columns.tolist(),
                          wafer_mc_stats=wafer_mc_stats_table,
                          wafer_break_even=wafer_break_even,
                          wafer_dynamics=wafer_dynamics,
                          # Сравнение линий
                          lines_comparison=lines_comparison,
                          # Графики мармелада
                          chart_comparison=chart_comparison,
                          chart_mc_base=chart_mc_base,
                          chart_mc_opt=chart_mc_opt,
                          chart_mc_pess=chart_mc_pess,
                          chart_market_share=chart_market_share,
                          chart_mc_comparison=chart_mc_comparison,
                          chart_break_even=chart_break_even,
                          chart_tornado=chart_tornado,
                          chart_wafer_tornado=chart_wafer_tornado,
                          chart_dynamics=chart_dynamics,
                          # Графики Монте-Карло для вафель
                          chart_wafer_mc_base=chart_wafer_mc_base,
                          chart_wafer_mc_opt=chart_wafer_mc_opt,
                          chart_wafer_mc_pess=chart_wafer_mc_pess,
                          # Графики сравнения линий
                          chart_lines_profit=chart_lines_profit,
                          chart_lines_risk=chart_lines_risk,
                          chart_lines_roi=chart_lines_roi,
                          chart_lines_break_even=chart_lines_break_even)


if __name__ == '__main__':
    import os
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=debug, host='0.0.0.0', port=port)

