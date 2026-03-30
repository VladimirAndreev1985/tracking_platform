# -*- coding: utf-8 -*-
"""
Пакет конфигурации платформы слежения.
Загрузка и валидация настроек из YAML-файла.
"""

from config.app_config import AppConfig, load_config

__all__ = ["AppConfig", "load_config"]
