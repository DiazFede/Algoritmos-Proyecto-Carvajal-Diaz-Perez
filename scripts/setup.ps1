param(
  [string]$Python = "python"
)

# Crear venv
Write-Host "Creando entorno virtual .venv..."
& $Python -m venv .venv

# Activar venv
Write-Host "Activando entorno..."
& .\.venv\Scripts\activate.ps1

# Instalar dependencias
Write-Host "Instalando dependencias desde requirements.txt..."
pip install -r requirements.txt

Write-Host "Listo. Usa '.\.venv\Scripts\activate' para activar el entorno en nuevas terminales."
