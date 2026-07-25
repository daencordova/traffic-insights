#!/bin/bash
# Script para formatear automáticamente el código

set -e

echo "📏 Formateando código con Ruff..."

# Verificar que ruff está instalado
if ! command -v ruff &> /dev/null; then
    echo "❌ Ruff no está instalado. Instalando..."
    pip install ruff
fi

# Ejecutar formateo
echo "🔧 Aplicando formato..."
ruff format .

# Ejecutar linting con auto-fix
echo "🔍 Aplicando correcciones de linting..."
ruff check --fix .

# Verificar que no hay errores
echo "✅ Verificando código..."
ruff check .

echo "✨ Formateo completado!"
