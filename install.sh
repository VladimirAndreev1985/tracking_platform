#!/bin/bash
# ================================================================
# TRACKING PLATFORM — Полный скрипт установки
# Для Raspberry Pi 5 + Hailo AI HAT+ 2
#
# Порядок установки и настройки:
# ──────────────────────────────────────────────────
# 1. Подключить оборудование:
#    - Hailo AI HAT+ 2 → на GPIO разъём RPi5
#    - Waveshare CAN HAT → на GPIO (поверх Hailo или через стакер)
#    - Arducam PTZ камера → USB
#    - Meskernel TS1224 дальномер → UART (GPIO 14/15)
#    - Logitech X56 HOTAS → USB
#    - HDMI монитор → micro-HDMI
#    - Питание моторов 24V → Driver Board → CAN HAT
#
# 2. Запустить этот скрипт:
#    bash install.sh
#
# 3. Перезагрузить RPi:
#    sudo reboot
#
# 4. Поднять CAN-интерфейс:
#    ./setup_can.sh
#
# 5. Запустить платформу:
#    ./run.sh
# ================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  TRACKING PLATFORM — Полная установка              ${NC}"
echo -e "${CYAN}  Raspberry Pi 5 + Hailo-10H + Cubemars AK80-64    ${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ════════════════════════════════════════════════
# 1. Обновление системы
# ════════════════════════════════════════════════
echo -e "${YELLOW}[1/8] Обновление системы...${NC}"

sudo apt-get update -qq
sudo apt-get upgrade -y -qq

echo -e "${GREEN}  ✓ Система обновлена${NC}"

# ════════════════════════════════════════════════
# 2. Системные зависимости
# ════════════════════════════════════════════════
echo -e "${YELLOW}[2/8] Системные зависимости...${NC}"

sudo apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    can-utils \
    libsdl2-dev \
    libsdl2-mixer-dev \
    libsdl2-image-dev \
    libsdl2-ttf-dev \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libv4l-dev \
    v4l-utils \
    git \
    curl \
    wget

echo -e "${GREEN}  ✓ Системные зависимости установлены${NC}"

# ════════════════════════════════════════════════
# 3. Hailo AI HAT+ 2 (Hailo-10H NPU)
# ════════════════════════════════════════════════
echo -e "${YELLOW}[3/8] Hailo AI HAT+ 2...${NC}"

# Проверяем, установлен ли Hailo Runtime
if dpkg -l | grep -q hailo; then
    echo -e "${GREEN}  ✓ Hailo Runtime уже установлен${NC}"
else
    echo -e "${YELLOW}  Hailo Runtime не найден.${NC}"
    echo -e "${YELLOW}  Для установки Hailo SDK на RPi5:${NC}"
    echo ""
    echo "  1. Включите Hailo репозиторий:"
    echo "     sudo apt-get install hailo-all"
    echo ""
    echo "  2. Или скачайте .deb пакеты с https://hailo.ai/developer-zone/"
    echo "     и установите:"
    echo "     sudo dpkg -i hailort_*.deb"
    echo "     sudo dpkg -i hailo-firmware_*.deb"
    echo ""
    echo -e "${YELLOW}  Пропускаю (можно установить позже).${NC}"
fi

# ════════════════════════════════════════════════
# 4. Виртуальное окружение Python
# ════════════════════════════════════════════════
echo -e "${YELLOW}[4/8] Виртуальное окружение Python...${NC}"

if [ ! -d "venv" ]; then
    python3 -m venv venv --system-site-packages
    echo -e "${GREEN}  ✓ Создано (с доступом к системным пакетам)${NC}"
else
    echo -e "${GREEN}  ✓ Уже существует${NC}"
fi

source venv/bin/activate

# ════════════════════════════════════════════════
# 5. Python-зависимости
# ════════════════════════════════════════════════
echo -e "${YELLOW}[5/8] Python-зависимости...${NC}"

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo -e "${GREEN}  ✓ Python-зависимости установлены:${NC}"
echo "    - PyYAML (конфигурация)"
echo "    - opencv-python (видео, HUD)"
echo "    - numpy (математика)"
echo "    - python-can (CAN-шина, MIT Mode)"
echo "    - pygame (джойстик X56)"
echo "    - pyserial (дальномер UART)"
echo "    - ultralytics (YOLO26)"

# ════════════════════════════════════════════════
# 6. Настройка CAN-интерфейса
# ════════════════════════════════════════════════
echo -e "${YELLOW}[6/8] Настройка CAN-интерфейса...${NC}"

CONFIG_FILE="/boot/firmware/config.txt"
if [ ! -f "$CONFIG_FILE" ]; then
    CONFIG_FILE="/boot/config.txt"
fi

if grep -q "mcp2515" "$CONFIG_FILE" 2>/dev/null; then
    echo -e "${GREEN}  ✓ CAN overlay уже настроен в $CONFIG_FILE${NC}"
else
    echo -e "${YELLOW}  Добавляю CAN overlay...${NC}"
    sudo bash -c "cat >> $CONFIG_FILE" << 'CAN_EOF'

# ── CAN Bus (Waveshare 2-CH CAN FD HAT для Cubemars AK80-64) ──
dtparam=spi=on
dtoverlay=mcp2515-can0,oscillator=12000000,interrupt=25,spimaxfrequency=2000000
CAN_EOF
    echo -e "${GREEN}  ✓ CAN overlay добавлен (НУЖНА ПЕРЕЗАГРУЗКА!)${NC}"
fi

# Настройка UART для дальномера
if grep -q "enable_uart=1" "$CONFIG_FILE" 2>/dev/null; then
    echo -e "${GREEN}  ✓ UART уже включён${NC}"
else
    echo -e "${YELLOW}  Включаю UART для дальномера...${NC}"
    sudo bash -c "echo 'enable_uart=1' >> $CONFIG_FILE"
    echo -e "${GREEN}  ✓ UART включён${NC}"
fi

# Скрипт поднятия CAN
cat > setup_can.sh << 'SETUP_CAN_EOF'
#!/bin/bash
# ── Поднять CAN-интерфейс для Cubemars AK80-64 ──
# Bitrate: 1 Мбит/с (стандарт Cubemars)
# Запускать после каждой перезагрузки RPi

echo "Настройка CAN0 (1 Мбит/с)..."
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 up type can bitrate 1000000
echo "CAN0 поднят:"
ip -details link show can0 | grep -E "can|state"
echo ""
echo "Проверка: candump can0 (Ctrl+C для выхода)"
SETUP_CAN_EOF
chmod +x setup_can.sh

echo -e "${GREEN}  ✓ setup_can.sh создан${NC}"

# ════════════════════════════════════════════════
# 7. Директории и модели
# ════════════════════════════════════════════════
echo -e "${YELLOW}[7/8] Директории и модели...${NC}"

mkdir -p logs
mkdir -p models

# Скачиваем модель YOLO26n если её нет
if [ ! -f "models/yolo26n.pt" ]; then
    echo -e "${YELLOW}  Скачиваю модель YOLO26n...${NC}"
    # Ultralytics автоматически скачает модель при первом запуске
    # Но можно скачать заранее:
    python3 -c "
from ultralytics import YOLO
model = YOLO('yolo26n.pt')
import shutil
shutil.move('yolo26n.pt', 'models/yolo26n.pt')
print('Модель YOLO26n скачана в models/')
" 2>/dev/null || echo -e "${YELLOW}  Модель будет скачана при первом запуске${NC}"
fi

echo -e "${GREEN}  ✓ Директории созданы${NC}"

# ════════════════════════════════════════════════
# 8. Скрипты запуска
# ════════════════════════════════════════════════
echo -e "${YELLOW}[8/8] Скрипты запуска...${NC}"

# Главный скрипт запуска
cat > run.sh << 'RUN_EOF'
#!/bin/bash
# ── Запуск платформы слежения ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source venv/bin/activate

echo "╔══════════════════════════════════════════════╗"
echo "║     TRACKING PLATFORM — Платформа слежения   ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Проверка CAN
if ! ip link show can0 2>/dev/null | grep -q "UP"; then
    echo "⚠ CAN0 не поднят! Запускаю setup_can.sh..."
    bash setup_can.sh
fi

python3 main.py "$@"
RUN_EOF
chmod +x run.sh

# Тест джойстика
cat > run_joystick_test.sh << 'JOY_EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source venv/bin/activate
python3 main.py --test-joystick
JOY_EOF
chmod +x run_joystick_test.sh

# Тест моторов
cat > run_motor_test.sh << 'MOT_EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source venv/bin/activate

# Проверка CAN
if ! ip link show can0 2>/dev/null | grep -q "UP"; then
    echo "⚠ CAN0 не поднят! Запускаю setup_can.sh..."
    bash setup_can.sh
fi

python3 main.py --test-motors
MOT_EOF
chmod +x run_motor_test.sh

echo -e "${GREEN}  ✓ Скрипты запуска созданы${NC}"

# ════════════════════════════════════════════════
# Готово!
# ════════════════════════════════════════════════
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  ✅ Установка завершена!                           ${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Порядок запуска:${NC}"
echo ""
echo "  1. ${YELLOW}sudo reboot${NC}"
echo "     (если CAN/UART overlay были только что добавлены)"
echo ""
echo "  2. ${YELLOW}./setup_can.sh${NC}"
echo "     (поднять CAN-интерфейс после перезагрузки)"
echo ""
echo "  3. ${YELLOW}./run_joystick_test.sh${NC}"
echo "     (проверить джойстик — оси, кнопки)"
echo ""
echo "  4. ${YELLOW}./run_motor_test.sh${NC}"
echo "     (проверить моторы — малые движения ±5°)"
echo ""
echo "  5. ${YELLOW}./run.sh${NC}"
echo "     (запуск платформы слежения)"
echo ""
echo -e "${GREEN}Управление:${NC}"
echo "  Стик X/Y      → наведение платформы"
echo "  Колёсико       → оптический зум камеры"
echo "  Кнопка 5       → переключение РУЧНОЙ ↔ АВТО"
echo "  Кнопка 3       → захват / сброс цели"
echo "  Кнопка 4       → переключение КОРД ↔ ПКТ"
echo "  Триггер        → замер дальности лазером"
echo "  Кнопка 1       → АВАРИЙНАЯ ОСТАНОВКА"
echo "  Кнопка 2       → возврат в центр"
echo "  ESC / Ctrl+C   → выход"
echo ""
echo -e "${CYAN}Оборудование:${NC}"
echo "  Моторы:     Cubemars AK80-64 × 2 (MIT Mode, 24V)"
echo "  Камера:     Arducam 5MP PTZ (1080p, оптический зум)"
echo "  Дальномер:  Meskernel TS1224 (UART, до 1200 м)"
echo "  AI:         Hailo-10H 40 TOPS (YOLO26)"
echo "  Джойстик:   Logitech X56 HOTAS (правая ручка)"
echo ""
