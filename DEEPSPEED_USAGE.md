# DeepSpeed Configuration Guide

Этот проект настроен для использования DeepSpeed с тремя различными стратегиями ZeRO для оптимизации памяти и ускорения обучения.

## Конфигурации DeepSpeed

### 1. ZeRO-1 (ds_zero1_config.json)
**Рекомендуется для**: Небольших моделей, когда GPU память достаточна
- Оптимизирует только состояния оптимизатора
- Минимальные накладные расходы на коммуникацию
- Лучшая производительность при достаточной памяти

### 2. ZeRO-2 (ds_zero2_config.json) 
**Рекомендуется для**: Средних моделей, ограниченная GPU память
- Оптимизирует состояния оптимизатора + градиенты
- Сбалансированное соотношение памяти и производительности
- Включает CPU offloading для дополнительной экономии памяти

### 3. ZeRO-3 (ds_zero3_config.json)
**Рекомендуется для**: Больших моделей, критическая нехватка GPU памяти  
- Оптимизирует параметры модели + градиенты + состояния оптимизатора
- Максимальная экономия памяти
- Некоторые накладные расходы на коммуникацию

## Команды для запуска

### ⚡ Автоматическое обнаружение GPU (рекомендуется):
```bash
# Основное обучение - автоматически использует все доступные GPU
make craken_ds1_auto  # ZeRO-1
make craken_ds2_auto  # ZeRO-2 (рекомендуется)
make craken_ds3_auto  # ZeRO-3

# Тестовое обучение
make test_craken_ds1_auto
make test_craken_ds2_auto  
make test_craken_ds3_auto
```

### 📋 С использованием config файлов:
```bash
# Основное обучение
make craken_ds1   # ZeRO-1
make craken_ds2   # ZeRO-2
make craken_ds3   # ZeRO-3

# Тестовое обучение
make test_craken_ds1
make test_craken_ds2
make test_craken_ds3
```

## Оптимизация Accelerate

### Ключевые настройки:
- **mixed_precision: bf16** - Использует bfloat16 для экономии памяти
- **gradient_accumulation_steps: 8** - Синхронизировано с trainer конфигом
- **CPU offloading** - Перегружает оптимизатор и параметры на CPU
- **gradient_clipping: 0.3** - Стабилизирует обучение

### Рекомендации по выбору конфига:

1. **Для моделей до 7B параметров**: Используйте ZeRO-1
2. **Для моделей 7B-13B**: Используйте ZeRO-2  
3. **Для моделей 13B+**: Используйте ZeRO-3

### Мониторинг и отладка:

- Используйте `wall_clock_breakdown: false` для production
- Установите `wall_clock_breakdown: true` для профилирования
- `steps_per_print: 10` показывает прогресс каждые 10 шагов

### Настройка для вашего оборудования:

1. **Если у вас несколько GPU**, измените `num_processes` в accelerate конфигах
2. **Для multi-node**, обновите `num_machines` и `machine_rank`
3. **Для оптимизации пропускной способности**, настройте bucket sizes в DeepSpeed конфигах

### Troubleshooting:

- **OOM ошибки**: Переключитесь на более высокий ZeRO stage
- **Медленная скорость**: Уменьшите CPU offloading или используйте меньший ZeRO stage  
- **NaN losses**: Проверьте gradient clipping и learning rate

## Интеграция с существующим кодом

Ваш код уже готов для DeepSpeed! Accelerate автоматически:
- Обертывает модель в DeepSpeed engine
- Настраивает оптимизатор согласно ZeRO конфигурации
- Управляет gradient accumulation и clipping
- Обрабатывает checkpoint saving/loading 