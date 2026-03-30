# 📖 ПОЛНАЯ ИНСТРУКЦИЯ ПО СБОРКЕ И НАСТРОЙКЕ

## Роботизированная платформа слежения TRACKING PLATFORM

---

## 📋 СОДЕРЖАНИЕ

1. [Список оборудования](#1-список-оборудования)
2. [Сборка оборудования](#2-сборка-оборудования)
3. [Настройка Driver Board моторов Cubemars](#3-настройка-driver-board)
4. [Настройка Raspberry Pi 5](#4-настройка-raspberry-pi-5)
5. [Установка Hailo AI HAT+](#5-установка-hailo-ai-hat)
6. [Установка Waveshare CAN HAT](#6-установка-waveshare-can-hat)
7. [Подключение дальномера](#7-подключение-дальномера)
8. [Подключение камеры и монитора](#8-подключение-камеры-и-монитора)
9. [Установка ПО](#9-установка-по)
10. [Тестирование подсистем](#10-тестирование-подсистем)
11. [Калибровка и первый запуск](#11-калибровка-и-первый-запуск)
12. [Диагностика проблем](#12-диагностика-проблем)

---

## 1. СПИСОК ОБОРУДОВАНИЯ

| # | Компонент | Модель | Кол-во |
|---|-----------|--------|--------|
| 1 | Одноплатный компьютер | Raspberry Pi 5 16 ГБ | 1 |
| 2 | AI-ускоритель | Raspberry Pi AI HAT+ 2 (Hailo-10H, 40 TOPS) | 1 |
| 3 | CAN-адаптер | Waveshare 2-CH CAN FD HAT | 1 |
| 4 | Мотор горизонта (Yaw) | Cubemars AK80-64 (24V) | 1 |
| 5 | Мотор вертикали (Pitch) | Cubemars AK80-64 (24V) | 1 |
| 6 | Драйвер мотора | Cubemars Driver Board | 2 |
| 7 | Камера | Arducam 5MP 1080p Pan-Tilt-Zoom | 1 |
| 8 | Лазерный дальномер | Meskernel TS1224 | 1 |
| 9 | Джойстик | Logitech X56 HOTAS (правая ручка) | 1 |
| 10 | Монитор | 9" автомобильный HDMI | 1 |
| 11 | Блок питания 24V | Для моторов (мин. 20А) | 1 |
| 12 | Блок питания 5V 5A | Для Raspberry Pi 5 (USB-C PD) | 1 |
| 13 | Провода CAN | Витая пара (CAN_H, CAN_L, GND) | ~2 м |
| 14 | Резистор 120 Ом | Терминирующий для CAN-шины | 1 |
| 15 | MicroSD карта | 64 ГБ+ (класс A2) | 1 |

---

## 2. СБОРКА ОБОРУДОВАНИЯ

### 2.1 Механическая сборка платформы

```
                    ┌─────────────┐
                    │   Камера    │
                    │  Arducam    │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │   Изделие   │
                    │ (КОРД/ПКТ)  │
                    └──────┬──────┘
                           │ ← Плечо ЦМ: 18.6 мм
              ┌────────────┴────────────┐
              │   Мотор Pitch (CAN ID 2)│ ← AK80-64
              │   Вертикальная ось      │
              └────────────┬────────────┘
                           │
              ┌────────────┴────────────┐
              │   Мотор Yaw (CAN ID 1)  │ ← AK80-64
              │   Горизонтальная ось    │
              └────────────┬────────────┘
                           │
                    ┌──────┴──────┐
                    │  Основание  │
                    │  платформы  │
                    └─────────────┘
```

**Важно:**
- Центр масс изделия по горизонтали (Yaw) должен быть **на оси вращения** мотора
- По вертикали (Pitch) плечо ЦМ = **18.6 мм** от оси вращения
- Моторы крепятся **напрямую** к осям (без редукторов/ремней)
- Камера и дальномер крепятся соосно с изделием

---

## 3. НАСТРОЙКА DRIVER BOARD

### 3.1 Подключение Driver Board к мотору

Каждый мотор AK80-64 подключается к своему Driver Board кабелем мотора (идёт в комплекте).

### 3.2 Настройка CAN ID

**КРИТИЧЕСКИ ВАЖНО:** Каждый мотор должен иметь уникальный CAN ID!

По умолчанию оба мотора имеют CAN ID = 1. Нужно изменить:
- **Мотор Yaw (горизонт):** CAN ID = **1**
- **Мотор Pitch (вертикаль):** CAN ID = **2**

#### Способ 1: Через утилиту Cubemars (Windows)

1. Подключите Driver Board к компьютеру через USB-CAN адаптер
2. Скачайте **Cubemars Motor Configuration Tool** с [cubemars.com](https://www.cubemars.com)
3. Подключитесь к мотору
4. В поле **CAN ID** установите нужный номер (1 или 2)
5. Нажмите **Write** для сохранения
6. Повторите для второго мотора

#### Способ 2: Через CAN-шину с RPi (после установки ПО)

```bash
# Подключите ТОЛЬКО ОДИН мотор к CAN-шине
# Поднимите CAN
sudo ip link set can0 up type can bitrate 1000000

# Отправьте команду смены CAN ID
# Формат зависит от прошивки Driver Board
# Обычно: cansend can0 001#XX (где XX — новый ID)
```

### 3.3 Настройка режима работы Driver Board

Driver Board должен быть настроен на **MIT Mode** (режим по умолчанию для AK80-64).

Проверьте в утилите Cubemars:
- **Control Mode:** MIT Mode
- **CAN Bitrate:** 1 Mbit/s (1000000)
- **Motor Direction:** Forward (по умолчанию)

### 3.4 Подключение питания Driver Board

```
Блок питания 24V ──┬── Driver Board 1 (Yaw)  ── V+, V-, GND
                    └── Driver Board 2 (Pitch) ── V+, V-, GND
```

⚠️ **ВНИМАНИЕ:**
- Используйте блок питания **24V минимум 20А** (пиковый ток AK80-64 до 18А)
- **НЕ** питайте моторы от Raspberry Pi!
- Проверьте полярность перед подключением!
- Подключайте питание **ПОСЛЕ** настройки CAN ID

---

## 4. НАСТРОЙКА RASPBERRY PI 5

### 4.1 Установка ОС

1. Скачайте **Raspberry Pi Imager** с [raspberrypi.com](https://www.raspberrypi.com/software/)
2. Вставьте microSD карту в компьютер
3. В Raspberry Pi Imager:
   - OS: **Raspberry Pi OS (64-bit) Bookworm**
   - Настройки:
     - Hostname: `tracking-platform`
     - Username: `pi`
     - Password: ваш пароль
     - Wi-Fi: настройте для первоначальной настройки
     - SSH: включите
4. Запишите образ на карту

### 4.2 Первый запуск

1. Вставьте microSD в RPi5
2. Подключите клавиатуру, мышь, монитор
3. Подключите питание USB-C PD (5V 5A)
4. Дождитесь загрузки и настройте систему

### 4.3 Базовая настройка

```bash
# Обновление системы
sudo apt-get update && sudo apt-get upgrade -y

# Включение SPI (для CAN HAT)
sudo raspi-config
# → Interface Options → SPI → Enable

# Включение I2C (для камеры)
sudo raspi-config
# → Interface Options → I2C → Enable

# Включение Serial Port (для дальномера)
sudo raspi-config
# → Interface Options → Serial Port
#   → Login shell over serial: NO
#   → Serial port hardware enabled: YES

# Перезагрузка
sudo reboot
```

---

## 5. УСТАНОВКА HAILO AI HAT+

### 5.1 Физическая установка

1. **Выключите** Raspberry Pi и отключите питание
2. Аккуратно установите Hailo AI HAT+ 2 на GPIO-разъём RPi5
3. Совместите все 40 пинов и надавите равномерно
4. Закрепите стойками (если есть в комплекте)

### 5.2 Установка драйверов Hailo

```bash
# Обновление firmware RPi5 (важно для Hailo)
sudo rpi-update
sudo reboot

# Установка Hailo Runtime
sudo apt-get update
sudo apt-get install -y hailo-all

# Проверка установки
hailortcli fw-control identify
# Должно показать: Hailo-10H, firmware version, etc.

# Если hailo-all недоступен, скачайте с hailo.ai:
# https://hailo.ai/developer-zone/
# sudo dpkg -i hailort_*.deb
```

### 5.3 Проверка Hailo

```bash
# Проверка что устройство видно
lspci | grep Hailo
# Должно показать: Hailo Technologies Ltd. Hailo-10H

# Тест производительности
hailortcli run-power-measurement
```

---

## 6. УСТАНОВКА CAN HAT

> **Примечание:** Инструкция написана для Waveshare 2-CH CAN FD HAT, но подойдёт
> любая CAN HAT для Raspberry Pi, совместимая с SocketCAN (MCP2515, MCP2518FD и др.).
> Код использует стандартный интерфейс `socketcan` → `can0` и не зависит от конкретной модели HAT.
> При использовании другой HAT — скорректируйте dtoverlay в `/boot/firmware/config.txt`
> согласно документации вашей HAT.

### 6.1 Физическая установка

⚠️ Если Hailo HAT уже установлен, используйте **GPIO стакер** (40-pin stacker header) чтобы поставить CAN HAT поверх Hailo HAT.

1. Установите стакер на Hailo HAT
2. Установите Waveshare CAN HAT на стакер
3. Закрепите стойками

### 6.2 Подключение моторов к CAN HAT

```
Waveshare CAN HAT (Channel 0 — can0)
    ┌─────────────────────────────────────┐
    │  CAN0_H ──┬── Driver Board 1 CAN_H │ (Мотор Yaw, ID=1)
    │           └── Driver Board 2 CAN_H │ (Мотор Pitch, ID=2)
    │                                     │
    │  CAN0_L ──┬── Driver Board 1 CAN_L │
    │           └── Driver Board 2 CAN_L │
    │                                     │
    │  GND    ──┬── Driver Board 1 GND   │
    │           └── Driver Board 2 GND   │
    └─────────────────────────────────────┘
```

**Оба мотора на одной CAN-шине** (различаются по CAN ID).

### 6.3 Терминирующий резистор

**ОБЯЗАТЕЛЬНО!** На CAN-шине должен быть терминирующий резистор **120 Ом** между CAN_H и CAN_L.

- Waveshare CAN HAT имеет встроенный переключатель/джампер для терминации — **ВКЛЮЧИТЕ его**
- Или припаяйте резистор 120 Ом между CAN_H и CAN_L на последнем устройстве шины

### 6.4 Настройка CAN overlay

```bash
# Редактируем config.txt
sudo nano /boot/firmware/config.txt

# Добавьте в конец файла:
dtparam=spi=on
dtoverlay=mcp2515-can0,oscillator=12000000,interrupt=25,spimaxfrequency=2000000

# Сохраните (Ctrl+O, Enter, Ctrl+X)
# Перезагрузите
sudo reboot
```

### 6.5 Проверка CAN

```bash
# Поднять CAN-интерфейс
sudo ip link set can0 up type can bitrate 1000000

# Проверить статус
ip -details link show can0
# Должно показать: state UP, bitrate 1000000

# Прослушать CAN-шину (включите питание моторов)
candump can0
# При включении моторов могут появиться пакеты

# Отправить тестовый пакет
cansend can0 001#FFFFFFFFFFFFFF
```

---

## 7. ПОДКЛЮЧЕНИЕ ДАЛЬНОМЕРА

### 7.1 Схема подключения Meskernel TS1224

```
Meskernel TS1224          Raspberry Pi 5 (GPIO)
    TX  ──────────────── GPIO 15 (RXD) — Pin 10
    RX  ──────────────── GPIO 14 (TXD) — Pin 8
    GND ──────────────── GND           — Pin 6
    VCC ──────────────── 5V            — Pin 2 (или внешний 5V)
```

### 7.2 Проверка UART

```bash
# Проверить что UART доступен
ls -la /dev/ttyAMA0
# Должен существовать

# Тест связи (minicom)
sudo apt-get install -y minicom
minicom -D /dev/ttyAMA0 -b 115200
# Ctrl+A, затем X для выхода
```

---

## 8. ПОДКЛЮЧЕНИЕ КАМЕРЫ И МОНИТОРА

### 8.1 Arducam PTZ камера

1. Подключите камеру через **USB** в любой USB-порт RPi5
2. Проверьте:
```bash
# Список видеоустройств
v4l2-ctl --list-devices
# Должно показать /dev/video0

# Проверка разрешений
v4l2-ctl -d /dev/video0 --list-formats-ext
# Должно показать 1920x1080 @ 30fps
```

### 8.2 HDMI монитор

1. Подключите 9" монитор к **micro-HDMI** порту RPi5 (порт 0 — ближайший к USB-C)
2. Монитор должен автоматически определиться

### 8.3 Джойстик Logitech X56

1. Подключите **только правую ручку (РУС)** через USB
2. Проверьте:
```bash
# Список input-устройств
ls /dev/input/js*
# Должен быть /dev/input/js0

# Тест джойстика
sudo apt-get install -y jstest-gtk
jstest /dev/input/js0
```

---

## 9. УСТАНОВКА ПО

### 9.1 Клонирование проекта

```bash
cd ~
git clone https://github.com/VladimirAndreev1985/tracking_platform.git
cd tracking_platform
```

### 9.2 Автоматическая установка

```bash
bash install.sh
```

Скрипт выполнит:
1. Обновление системы
2. Установку системных зависимостей (SDL2, OpenCV, can-utils)
3. Проверку Hailo SDK
4. Создание виртуального окружения Python
5. Установку Python-библиотек (YOLO26, OpenCV, python-can, pygame, pyserial)
6. Настройку CAN overlay и UART
7. Скачивание модели YOLO26n
8. Создание скриптов запуска

### 9.3 Перезагрузка

```bash
sudo reboot
```

### 9.4 Поднятие CAN после перезагрузки

```bash
cd ~/tracking_platform
./setup_can.sh
```

---

## 10. ТЕСТИРОВАНИЕ ПОДСИСТЕМ

### 10.1 Тест джойстика

```bash
./run_joystick_test.sh
```

Двигайте стик и нажимайте кнопки. Запишите номера осей и кнопок.
Если номера отличаются от конфига — отредактируйте `config/settings.yaml`:

```yaml
joystick:
  axis_yaw: 0      # ← номер оси горизонта
  axis_pitch: 1     # ← номер оси вертикали
  axis_zoom: 5      # ← номер оси колёсика зума
```

### 10.2 Тест моторов

⚠️ **Перед тестом:**
- Убедитесь что питание 24V подключено к Driver Board
- CAN-шина поднята (`./setup_can.sh`)
- Платформа надёжно закреплена
- Рядом нет людей в зоне вращения

```bash
./run_motor_test.sh
```

Каждый мотор сделает движение +5° и обратно. Проверьте:
- Оба мотора отвечают
- Направление вращения правильное (если нет — измените `direction: -1` в конфиге)

### 10.3 Тест камеры

```bash
cd ~/tracking_platform
source venv/bin/activate
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print(f'Камера: {frame.shape[1]}x{frame.shape[0]}' if ret else 'ОШИБКА!')
cap.release()
"
```

### 10.4 Тест дальномера

```bash
cd ~/tracking_platform
source venv/bin/activate
python3 -c "
import serial
ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
print(f'Дальномер: порт открыт' if ser.is_open else 'ОШИБКА!')
ser.close()
"
```

---

## 11. КАЛИБРОВКА И ПЕРВЫЙ ЗАПУСК

### 11.1 Начальное положение

1. Вручную установите платформу в **центральное положение** (изделие смотрит прямо вперёд)
2. Включите питание 24V моторов

### 11.2 Первый запуск

```bash
./run.sh
```

При первом запуске:
- Текущее положение моторов устанавливается как **ноль (0°, 0°)**
- HUD отображается на мониторе
- Управляйте джойстиком

### 11.3 Настройка MIT Mode gains

Если платформа **вибрирует** — уменьшите `kp`, увеличьте `kd`:
```yaml
motors:
  horizontal:
    kp: 80.0    # ← уменьшить
    kd: 5.0     # ← увеличить
```

Если платформа **медленно реагирует** — увеличьте `kp`:
```yaml
motors:
  horizontal:
    kp: 120.0   # ← увеличить
```

### 11.4 Настройка гравитационной компенсации

Если вертикальная ось **проседает** под весом изделия:
```yaml
motors:
  vertical:
    gravity_compensation_nm: 8.0  # ← увеличить (было 7.3)
```

Если вертикальная ось **задирается вверх**:
```yaml
motors:
  vertical:
    gravity_compensation_nm: 6.5  # ← уменьшить
```

---

## 12. ДИАГНОСТИКА ПРОБЛЕМ

### CAN-шина не работает

```bash
# Проверить overlay
dmesg | grep -i can
dmesg | grep -i mcp2515

# Проверить интерфейс
ip link show can0

# Перезапустить CAN
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000

# Прослушать шину
candump can0
```

### Мотор не отвечает

1. Проверьте питание 24V на Driver Board (LED должен гореть)
2. Проверьте CAN-проводку (CAN_H, CAN_L, GND)
3. Проверьте терминирующий резистор 120 Ом
4. Проверьте CAN ID в конфиге совпадает с настройкой Driver Board
5. Попробуйте `candump can0` — есть ли ответы

### Камера не определяется

```bash
v4l2-ctl --list-devices
ls /dev/video*
# Если /dev/video0 нет — переподключите USB камеры
```

### Джойстик не найден

```bash
lsusb | grep -i logitech
ls /dev/input/js*
# Если нет — переподключите USB джойстика
```

### Hailo не определяется

```bash
lspci | grep -i hailo
hailortcli fw-control identify
# Если не видно — проверьте физическое подключение HAT
# и обновите firmware: sudo rpi-update
```

### Дальномер не отвечает

```bash
# Проверить UART
ls /dev/ttyAMA0
# Проверить проводку TX→RX, RX→TX, GND→GND
# Проверить baudrate (115200)
```

---

## 📌 КРАТКАЯ ШПАРГАЛКА

```bash
# После каждой перезагрузки RPi:
cd ~/tracking_platform
./setup_can.sh          # Поднять CAN
./run.sh                # Запустить платформу

# Управление (Logitech X56 HOTAS — правая ручка):
# Стик X/Y        → наведение платформы
# Колёсико         → оптический зум камеры
# Триггер (кн. 0) → ОГОНЬ (усиленные gains для гашения отдачи)
# Кнопка A (кн. 1) → АВАРИЙНАЯ ОСТАНОВКА
# Кнопка B (кн. 2) → захват / сброс цели
# Кнопка C (кн. 3) → замер дальности лазером
# Кнопка D (кн. 4) → КОРД ↔ ПКТ
# Pinky    (кн. 5) → РУЧНОЙ ↔ АВТО
# Hat Push (кн. 6) → возврат в центр
# ESC              → выход
```
