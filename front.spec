# -*- mode: python ; coding: utf-8 -*-

import sys
import os

block_cipher = None

a = Analysis(
    ['front.py'],
    pathex=[],
    binaries=[],
    datas=[
		# 🧠 Adiciona todos os classificadores básicos do OpenCV usados pelo DeepFace
		('C:\\Users\\user318\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml', 'cv2/data'),
		('C:\\Users\\user318\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml', 'cv2/data'),
		('C:\\Users\\user318\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\cv2\\data\\haarcascade_profileface.xml', 'cv2/data'),
		('C:\\Users\\user318\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\cv2\\data\\haarcascade_smile.xml', 'cv2/data'),

		# 🗂️ Pastas do seu projeto
		('usuarios', 'usuarios'),
		('imagens', 'imagens'),
	],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='front',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # True = mostra terminal | False = sem console
)
