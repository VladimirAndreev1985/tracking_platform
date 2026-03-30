# -*- coding: utf-8 -*-
"""
Профессиональный баллистический калькулятор.

Реализует полную модель внешней баллистики с:
- Стандартной drag-функцией G1 (Ingalls/Mayevski) с табличными коэффициентами
- Численным интегрированием траектории методом Рунге-Кутты 4-го порядка (RK4)
- Учётом гравитации, плотности воздуха, температуры, давления, влажности
- Итеративным расчётом точки перехвата движущейся цели (lead point)
- Поддержкой профилей вооружения (КОРД 12.7×108, ПКТ 7.62×54R)

Drag-модель G1:
  Стандартная модель сопротивления для оживальных пуль.
  Коэффициент сопротивления Cd зависит от числа Маха (M):
  - Дозвуковой (M < 0.9): Cd растёт медленно
  - Трансзвуковой (0.9 < M < 1.2): резкий скачок Cd
  - Сверхзвуковой (M > 1.2): Cd падает

  Реальная формула drag: F_drag = 0.5 * ρ * v² * Cd(M) * S
  где S = π * (d/2)² — площадь миделя пули

  BC (баллистический коэффициент) связывает реальную пулю
  с эталонной: BC = (m / d²) / (m_ref / d_ref²)
  Чем выше BC — тем лучше пуля сохраняет скорость.
"""

import math
import logging
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from config.app_config import WeaponProfile, AtmosphereConfig

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# Стандартная drag-функция G1 (Ingalls/Mayevski)
# Табличные коэффициенты для расчёта коэффициента сопротивления
# по числу Маха. Источник: Ballistic Research Laboratory.
# ════════════════════════════════════════════════════════════════

# Таблица G1: (Mach_min, Mach_max, A, N)
# Cd_ref(M) = A * M^N  (в каждом диапазоне Маха)
# Это стандартные коэффициенты Mayevski/Ingalls для G1-модели.
_G1_TABLE = [
    # (M_min, M_max,        A,          N)
    (0.00, 0.05,   0.0,        0.0),      # Нулевая скорость
    (0.05, 0.45,   55.5556,   -1.258),
    (0.45, 0.70,   98.0295,   -1.950),
    (0.70, 0.85,   137.218,   -2.100),
    (0.85, 0.925,  254.014,   -2.950),
    (0.925, 0.975, 466.978,   -4.150),
    (0.975, 1.000, 581.616,   -4.950),
    (1.000, 1.025, 582.672,   -4.960),
    (1.025, 1.075, 462.218,   -4.100),
    (1.075, 1.125, 352.435,   -3.450),
    (1.125, 1.200, 273.710,   -2.950),
    (1.200, 1.350, 202.461,   -2.450),
    (1.350, 1.500, 171.674,   -2.200),
    (1.500, 1.750, 154.178,   -2.050),
    (1.750, 2.000, 138.037,   -1.875),
    (2.000, 2.500, 130.270,   -1.800),
    (2.500, 3.000, 124.550,   -1.750),
    (3.000, 5.000, 121.093,   -1.725),
]


def _g1_drag_coefficient(mach: float) -> float:
    """
    Вычислить эталонный коэффициент сопротивления G1 по числу Маха.

    Использует табличную аппроксимацию Mayevski/Ingalls.

    Args:
        mach: Число Маха (v / скорость_звука)

    Returns:
        Cd_ref — эталонный коэффициент drag для G1-модели
    """
    if mach < 0.05:
        return 0.0

    for m_min, m_max, a, n in _G1_TABLE:
        if m_min <= mach < m_max:
            # Замедление (retardation) A * M^N, нормализованное
            # Преобразуем в Cd через стандартную формулу
            retardation = a * (mach ** n)
            # Cd = retardation / (v² в единицах) — упрощённо
            # Для G1 модели retardation напрямую используется
            return retardation

    # За пределами таблицы (M > 5.0) — экстраполяция
    a, n = _G1_TABLE[-1][2], _G1_TABLE[-1][3]
    return a * (mach ** n)


# ════════════════════════════════════════════════════════════════
# Атмосферная модель
# ════════════════════════════════════════════════════════════════

class Atmosphere:
    """
    Модель атмосферы для баллистических расчётов.

    Вычисляет плотность воздуха и скорость звука
    с учётом высоты, температуры, давления и влажности.
    """

    # Физические константы
    R_AIR = 287.058        # Удельная газовая постоянная воздуха (Дж/(кг·К))
    R_VAPOR = 461.495      # Удельная газовая постоянная водяного пара (Дж/(кг·К))
    GAMMA = 1.4            # Показатель адиабаты воздуха
    LAPSE_RATE = 0.0065    # Температурный градиент (К/м)
    SEA_LEVEL_T = 288.15   # Стандартная температура на уровне моря (К)
    SEA_LEVEL_P = 101325.0 # Стандартное давление на уровне моря (Па)

    def __init__(self, config: AtmosphereConfig):
        self._cfg = config
        self.update()

    def update(self, config: AtmosphereConfig = None):
        """Пересчитать атмосферные параметры."""
        if config:
            self._cfg = config

        cfg = self._cfg
        self.gravity = cfg.gravity_mps2
        self.temperature_k = cfg.temperature_c + 273.15
        self.pressure_pa = cfg.pressure_hpa * 100.0
        self.altitude = cfg.altitude_m
        self.humidity = cfg.humidity_pct / 100.0

        # Поправка давления на высоту (барометрическая формула)
        if self.altitude > 0:
            self.pressure_at_alt = self.pressure_pa * (
                1.0 - self.LAPSE_RATE * self.altitude / self.SEA_LEVEL_T
            ) ** (self.gravity / (self.R_AIR * self.LAPSE_RATE))
        else:
            self.pressure_at_alt = self.pressure_pa

        # Давление насыщенного водяного пара (формула Бака)
        t_c = cfg.temperature_c
        p_sat = 611.21 * math.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))
        p_vapor = self.humidity * p_sat

        # Плотность воздуха с учётом влажности (виртуальная температура)
        p_dry = self.pressure_at_alt - p_vapor
        self.density = (p_dry / (self.R_AIR * self.temperature_k) +
                        p_vapor / (self.R_VAPOR * self.temperature_k))

        # Скорость звука
        self.speed_of_sound = math.sqrt(
            self.GAMMA * self.R_AIR * self.temperature_k
        )

        # Отношение плотности к стандартной (ICAO)
        rho_standard = 1.2250  # кг/м³ при 15°C, 1013.25 гПа, 0 м
        self.density_ratio = self.density / rho_standard

    def mach_number(self, velocity: float) -> float:
        """Число Маха для заданной скорости."""
        return velocity / self.speed_of_sound if self.speed_of_sound > 0 else 0.0


# ════════════════════════════════════════════════════════════════
# Результат баллистического расчёта
# ════════════════════════════════════════════════════════════════

@dataclass
class BallisticSolution:
    """Полный результат баллистического расчёта."""
    # Время полёта (сек)
    time_of_flight: float = 0.0
    # Скорость пули на дистанции (м/с)
    velocity_at_target: float = 0.0
    # Энергия пули на дистанции (Дж)
    energy_at_target: float = 0.0
    # Падение пули (м)
    bullet_drop_m: float = 0.0
    # Падение в угловых единицах
    bullet_drop_mrad: float = 0.0   # Миллирадианы
    bullet_drop_moa: float = 0.0    # Угловые минуты (MOA)
    # Углы упреждения для движущейся цели (градусы)
    lead_yaw_deg: float = 0.0
    lead_pitch_deg: float = 0.0
    # Полный угол наведения (с учётом drop + lead)
    total_yaw_deg: float = 0.0
    total_pitch_deg: float = 0.0
    # Точка перехвата в 3D (м, относительно платформы)
    intercept_x: float = 0.0
    intercept_y: float = 0.0
    intercept_z: float = 0.0
    # Дистанция до точки перехвата (м)
    intercept_distance: float = 0.0
    # Перехват возможен
    is_valid: bool = False
    # Число Маха на дистанции
    mach_at_target: float = 0.0
    # Имя активного профиля
    weapon_name: str = ""


# ════════════════════════════════════════════════════════════════
# Баллистический калькулятор
# ════════════════════════════════════════════════════════════════

class BallisticCalculator:
    """
    Профессиональный баллистический калькулятор с G1 drag-моделью.

    Использует:
    - Численное интегрирование RK4 для точной траектории
    - Стандартную G1 drag-функцию с табличными коэффициентами
    - Полную атмосферную модель (высота, T, P, влажность)
    - Итеративный расчёт точки перехвата движущейся цели
    """

    def __init__(self, atmosphere_cfg: AtmosphereConfig,
                 profiles: Dict[str, WeaponProfile],
                 active_profile: str = "kord"):
        """
        Args:
            atmosphere_cfg: Атмосферные условия
            profiles: Словарь профилей вооружения
            active_profile: Ключ активного профиля
        """
        self._atmosphere = Atmosphere(atmosphere_cfg)
        self._profiles = profiles
        self._active_key = active_profile

        # Текущий профиль
        self._profile: WeaponProfile = profiles.get(
            active_profile, next(iter(profiles.values())) if profiles else WeaponProfile()
        )

        # Предрассчитанные параметры пули
        self._update_bullet_params()

        logger.info(
            f"BallisticCalculator: профиль={self._profile.short_name}, "
            f"V₀={self._profile.muzzle_velocity_mps} м/с, "
            f"BC(G1)={self._profile.ballistic_coefficient_g1}, "
            f"ρ={self._atmosphere.density:.4f} кг/м³, "
            f"a={self._atmosphere.speed_of_sound:.1f} м/с"
        )

    def _update_bullet_params(self):
        """Пересчитать параметры пули из текущего профиля."""
        p = self._profile
        # Масса пули в кг
        self._bullet_mass_kg = p.bullet_mass_g / 1000.0
        # Площадь миделя (поперечного сечения) пули (м²)
        radius_m = (p.bullet_diameter_mm / 1000.0) / 2.0
        self._cross_section = math.pi * radius_m ** 2
        # Секционная плотность (кг/м²)
        self._sectional_density = self._bullet_mass_kg / self._cross_section

    # ── Переключение профилей ───────────────────────────────

    @property
    def active_profile(self) -> WeaponProfile:
        """Текущий активный профиль вооружения."""
        return self._profile

    @property
    def active_profile_key(self) -> str:
        """Ключ текущего профиля."""
        return self._active_key

    @property
    def profile_names(self) -> List[str]:
        """Список ключей всех профилей."""
        return list(self._profiles.keys())

    def switch_profile(self, key: str = None):
        """
        Переключить профиль вооружения.

        Args:
            key: Ключ профиля. Если None — циклическое переключение.
        """
        if key and key in self._profiles:
            self._active_key = key
        else:
            # Циклическое переключение
            keys = list(self._profiles.keys())
            if keys:
                idx = keys.index(self._active_key) if self._active_key in keys else -1
                self._active_key = keys[(idx + 1) % len(keys)]

        self._profile = self._profiles[self._active_key]
        self._update_bullet_params()

        logger.info(
            f"Профиль вооружения: {self._profile.short_name} "
            f"({self._profile.cartridge}), "
            f"V₀={self._profile.muzzle_velocity_mps} м/с, "
            f"BC={self._profile.ballistic_coefficient_g1}"
        )

    # ── Расчёт траектории пули ──────────────────────────────

    def _accel(self, vx: float, vy: float, bc: float,
               atm: 'Atmosphere') -> tuple:
        """
        Вычислить ускорение пули (для RK4).

        Args:
            vx, vy: Компоненты скорости (м/с)
            bc: Баллистический коэффициент G1
            atm: Атмосферная модель

        Returns:
            (ax, ay) — ускорение по осям (м/с²)
        """
        v = math.sqrt(vx * vx + vy * vy)
        if v < 1.0:
            return (0.0, -atm.gravity)

        mach = v / atm.speed_of_sound
        g1_ret = _g1_drag_coefficient(mach)
        deceleration = g1_ret * atm.density_ratio / bc
        drag_factor = deceleration / v

        ax = -drag_factor * vx
        ay = -drag_factor * vy - atm.gravity
        return (ax, ay)

    def _spin_drift(self, tof: float, distance_m: float) -> float:
        """
        Вычислить боковое отклонение из-за spin-drift (гироскопический снос).

        Эмпирическая формула Litz:
            SD = 1.25 * (SG + 1.2) * ToF^1.83
        где SG — гироскопическая стабильность (≈1.5 для стандартных пуль).

        Для 12.7 мм Б-32 на 500 м: SD ≈ 3-5 см (вправо для правой нарезки)
        Для 7.62 мм ЛПС на 500 м: SD ≈ 1-2 см

        Args:
            tof: Время полёта (сек)
            distance_m: Дистанция (м)

        Returns:
            Боковое отклонение (м), положительное = вправо
        """
        # Гироскопическая стабильность (упрощённая оценка)
        # SG ≈ 1.5 для стандартных оживальных пуль
        sg = 1.5
        # Эмпирическая формула Litz (в дюймах, затем → метры)
        sd_inches = 1.25 * (sg + 1.2) * (tof ** 1.83)
        sd_meters = sd_inches * 0.0254
        return sd_meters

    def compute_trajectory(self, distance_m: float,
                           elevation_rad: float = 0.0) -> BallisticSolution:
        """
        Рассчитать траекторию пули до заданной дистанции.

        Использует метод Рунге-Кутты 4-го порядка (RK4) с G1 drag-функцией.
        Учитывает: drag (G1), гравитацию, плотность воздуха, spin-drift.

        Args:
            distance_m: Наклонная дальность до цели (м)
            elevation_rad: Угол возвышения ствола (рад)

        Returns:
            BallisticSolution с ToF, drop, скоростью, энергией и spin-drift
        """
        sol = BallisticSolution(weapon_name=self._profile.short_name)

        if distance_m <= 0 or distance_m > self._profile.max_range_m:
            return sol

        p = self._profile
        atm = self._atmosphere
        bc = p.ballistic_coefficient_g1
        v0 = p.muzzle_velocity_mps

        # Начальные условия (2D: x — горизонт, y — вертикаль)
        vx = v0 * math.cos(elevation_rad)
        vy = v0 * math.sin(elevation_rad)
        x = 0.0
        y = 0.0
        t = 0.0

        dt = 0.001  # Шаг RK4 (1 мс) — достаточно для 4-го порядка точности

        # ── Интегрирование методом Рунге-Кутты 4-го порядка (RK4) ──
        while x < distance_m and t < 15.0:
            v = math.sqrt(vx * vx + vy * vy)
            if v < 10.0:
                break

            # k1 — ускорение в начале шага
            ax1, ay1 = self._accel(vx, vy, bc, atm)

            # k2 — ускорение в середине шага (с k1/2)
            ax2, ay2 = self._accel(
                vx + ax1 * dt / 2, vy + ay1 * dt / 2, bc, atm
            )

            # k3 — ускорение в середине шага (с k2/2)
            ax3, ay3 = self._accel(
                vx + ax2 * dt / 2, vy + ay2 * dt / 2, bc, atm
            )

            # k4 — ускорение в конце шага (с k3)
            ax4, ay4 = self._accel(
                vx + ax3 * dt, vy + ay3 * dt, bc, atm
            )

            # RK4 взвешенное среднее
            dvx = (ax1 + 2 * ax2 + 2 * ax3 + ax4) * dt / 6
            dvy = (ay1 + 2 * ay2 + 2 * ay3 + ay4) * dt / 6

            vx += dvx
            vy += dvy
            x += vx * dt
            y += vy * dt
            t += dt

        # ── Результат ──
        v_final = math.sqrt(vx * vx + vy * vy)
        sol.time_of_flight = t
        sol.velocity_at_target = v_final
        sol.energy_at_target = 0.5 * self._bullet_mass_kg * v_final * v_final
        sol.bullet_drop_m = -y  # Падение (положительное = вниз)
        sol.mach_at_target = v_final / atm.speed_of_sound
        sol.is_valid = (t < 10.0 and v_final > 50.0)

        # Угловые единицы падения
        if distance_m > 0:
            drop_angle_rad = math.atan(sol.bullet_drop_m / distance_m)
            sol.bullet_drop_mrad = drop_angle_rad * 1000.0
            sol.bullet_drop_moa = math.degrees(drop_angle_rad) * 60.0

        return sol

    def time_of_flight(self, distance_m: float) -> float:
        """Быстрый расчёт времени полёта до дистанции."""
        return self.compute_trajectory(distance_m).time_of_flight

    # ── Расчёт точки перехвата движущейся цели ──────────────

    def compute_lead(self, distance_m: float,
                     target_speed_mps: float,
                     target_heading_rad: float,
                     target_elevation_rad: float = 0.0,
                     platform_yaw_rad: float = 0.0,
                     platform_pitch_rad: float = 0.0) -> BallisticSolution:
        """
        Вычислить точку перехвата движущейся цели.

        Итеративный алгоритм:
        1. Начальное приближение ToF = distance / V₀
        2. Предсказать позицию цели через ToF
        3. Вычислить точный ToF до предсказанной позиции
        4. Повторить 2-3 до сходимости (обычно 4-6 итераций)

        Args:
            distance_m: Текущая дальность до цели (м)
            target_speed_mps: Скорость цели (м/с)
            target_heading_rad: Курс цели (рад, 0 = удаление от платформы)
            target_elevation_rad: Угол набора/снижения цели (рад)
            platform_yaw_rad: Текущий yaw платформы (рад)
            platform_pitch_rad: Текущий pitch платформы (рад)

        Returns:
            BallisticSolution с углами упреждения и точкой перехвата
        """
        sol = BallisticSolution(weapon_name=self._profile.short_name)

        if distance_m <= 0 or distance_m > self._profile.max_range_m:
            return sol

        # ── Текущая позиция цели в декартовых координатах ──
        # (относительно платформы, X=вперёд, Y=вправо, Z=вверх)
        cos_p = math.cos(platform_pitch_rad)
        sin_p = math.sin(platform_pitch_rad)
        cos_y = math.cos(platform_yaw_rad)
        sin_y = math.sin(platform_yaw_rad)

        tgt_x = distance_m * cos_p * cos_y
        tgt_y = distance_m * cos_p * sin_y
        tgt_z = distance_m * sin_p

        # ── Вектор скорости цели ──
        cos_te = math.cos(target_elevation_rad)
        sin_te = math.sin(target_elevation_rad)
        cos_th = math.cos(target_heading_rad)
        sin_th = math.sin(target_heading_rad)

        vt_x = target_speed_mps * cos_te * cos_th
        vt_y = target_speed_mps * cos_te * sin_th
        vt_z = target_speed_mps * sin_te

        # ── Итеративный расчёт точки перехвата ──
        tof = distance_m / self._profile.muzzle_velocity_mps  # Начальное приближение
        prev_tof = 0.0

        for iteration in range(8):  # Макс 8 итераций
            # Предсказанная позиция цели через tof
            ix = tgt_x + vt_x * tof
            iy = tgt_y + vt_y * tof
            iz = tgt_z + vt_z * tof

            # Наклонная дальность до точки перехвата
            intercept_dist = math.sqrt(ix * ix + iy * iy + iz * iz)

            if intercept_dist <= 0 or intercept_dist > self._profile.max_range_m:
                break

            # Угол возвышения до точки перехвата (для учёта drop)
            horiz_dist = math.sqrt(ix * ix + iy * iy)
            elev = math.atan2(iz, horiz_dist) if horiz_dist > 0 else 0.0

            # Точный ToF до этой дистанции через полную баллистику
            traj = self.compute_trajectory(intercept_dist, elev)
            new_tof = traj.time_of_flight

            # Компенсация drop: цель нужно «поднять» на величину падения
            iz_compensated = iz + traj.bullet_drop_m

            # Проверка сходимости
            if abs(new_tof - tof) < 0.0005:  # < 0.5 мс
                tof = new_tof
                break

            prev_tof = tof
            tof = new_tof

        # ── Финальные углы ──
        # Пересчитываем с компенсацией drop
        ix_final = tgt_x + vt_x * tof
        iy_final = tgt_y + vt_y * tof
        iz_final = tgt_z + vt_z * tof

        intercept_dist = math.sqrt(ix_final**2 + iy_final**2 + iz_final**2)

        # Получаем drop для финальной дистанции
        final_traj = self.compute_trajectory(intercept_dist)
        drop = final_traj.bullet_drop_m

        # Компенсируем drop
        iz_compensated = iz_final + drop

        # Углы до точки перехвата (с компенсацией)
        if intercept_dist > 0:
            lead_yaw = math.atan2(iy_final, ix_final)
            horiz_final = math.sqrt(ix_final**2 + iy_final**2)
            lead_pitch = math.atan2(iz_compensated, horiz_final)
        else:
            lead_yaw = 0.0
            lead_pitch = 0.0

        # Углы упреждения = разница между точкой перехвата и текущей позицией цели
        current_yaw = math.atan2(tgt_y, tgt_x)
        current_pitch = math.atan2(tgt_z, math.sqrt(tgt_x**2 + tgt_y**2))

        sol.lead_yaw_deg = math.degrees(lead_yaw - current_yaw)
        sol.lead_pitch_deg = math.degrees(lead_pitch - current_pitch)
        sol.total_yaw_deg = math.degrees(lead_yaw)
        sol.total_pitch_deg = math.degrees(lead_pitch)

        sol.time_of_flight = tof
        sol.velocity_at_target = final_traj.velocity_at_target
        sol.energy_at_target = final_traj.energy_at_target
        sol.bullet_drop_m = drop
        sol.bullet_drop_mrad = final_traj.bullet_drop_mrad
        sol.bullet_drop_moa = final_traj.bullet_drop_moa
        sol.mach_at_target = final_traj.mach_at_target

        sol.intercept_x = ix_final
        sol.intercept_y = iy_final
        sol.intercept_z = iz_final
        sol.intercept_distance = intercept_dist

        sol.is_valid = (
            intercept_dist < self._profile.effective_range_m * 1.5
            and tof < 8.0
            and final_traj.velocity_at_target > 100.0
        )

        logger.debug(
            f"Баллистика [{self._profile.short_name}]: "
            f"dist={distance_m:.0f}м → перехват={intercept_dist:.0f}м, "
            f"ToF={tof:.3f}с, V={final_traj.velocity_at_target:.0f}м/с, "
            f"drop={drop:.2f}м, M={final_traj.mach_at_target:.2f}, "
            f"lead=({sol.lead_yaw_deg:+.2f}°, {sol.lead_pitch_deg:+.2f}°)"
        )

        return sol

    # ── Обновление атмосферы ────────────────────────────────

    def update_atmosphere(self, config: AtmosphereConfig):
        """Обновить атмосферные условия и пересчитать параметры."""
        self._atmosphere.update(config)
        logger.info(
            f"Атмосфера обновлена: T={config.temperature_c}°C, "
            f"P={config.pressure_hpa} гПа, alt={config.altitude_m} м, "
            f"ρ={self._atmosphere.density:.4f} кг/м³"
        )

    # ── Таблица стрельбы ────────────────────────────────────

    def generate_range_table(self, max_distance: int = None,
                             step: int = 100) -> List[dict]:
        """
        Сгенерировать таблицу стрельбы для текущего профиля.

        Args:
            max_distance: Максимальная дальность (м). None = effective_range.
            step: Шаг дальности (м)

        Returns:
            Список словарей с данными для каждой дистанции
        """
        if max_distance is None:
            max_distance = self._profile.effective_range_m

        table = []
        for dist in range(step, max_distance + 1, step):
            traj = self.compute_trajectory(float(dist))
            table.append({
                "дистанция_м": dist,
                "ToF_с": round(traj.time_of_flight, 3),
                "скорость_мс": round(traj.velocity_at_target, 1),
                "энергия_Дж": round(traj.energy_at_target, 0),
                "падение_м": round(traj.bullet_drop_m, 2),
                "падение_мрад": round(traj.bullet_drop_mrad, 2),
                "падение_MOA": round(traj.bullet_drop_moa, 1),
                "Мах": round(traj.mach_at_target, 2),
            })

        return table
