@echo off
REM Launch Enhanced Car GAN Interface

echo.
echo ========================================
echo  🚗 Car GAN Interface
echo ========================================
echo.

call .venv\Scripts\activate.bat

echo 🚀 Launching interface...
echo.
echo Features:
echo  ✨ Single Car Generation
echo  🎯 Batch Generation
echo  🔄 Interpolation + GIF Export
echo  🎲 Variations Generator
echo  🎆 Mega Showcase (64+ cars!)
echo.
echo Opening at: http://localhost:7863
echo.

python app_enhanced.py

pause
