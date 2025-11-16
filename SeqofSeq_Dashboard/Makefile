# Makefile for SeqofSeq Dashboard
# Provides convenient shortcuts for common tasks

.PHONY: help build up down logs restart clean install dev test

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║         SeqofSeq Dashboard - Available Commands              ║"
	@echo "╠══════════════════════════════════════════════════════════════╣"
	@echo "║  make build      - Build Docker images                       ║"
	@echo "║  make up         - Start dashboard (detached)                ║"
	@echo "║  make down       - Stop dashboard                            ║"
	@echo "║  make logs       - View dashboard logs                       ║"
	@echo "║  make restart    - Restart dashboard                         ║"
	@echo "║  make clean      - Remove containers and volumes             ║"
	@echo "║  make install    - Install Python dependencies locally       ║"
	@echo "║  make dev        - Run in development mode (local)           ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"

# Docker commands
build:
	@echo "🔨 Building Docker image..."
	docker-compose build

up:
	@echo "🚀 Starting dashboard..."
	docker-compose up -d
	@echo "✅ Dashboard started at http://localhost:8050"

down:
	@echo "🛑 Stopping dashboard..."
	docker-compose down

logs:
	@echo "📋 Viewing logs (Ctrl+C to exit)..."
	docker-compose logs -f dashboard

restart:
	@echo "🔄 Restarting dashboard..."
	docker-compose restart dashboard

clean:
	@echo "🧹 Cleaning up..."
	docker-compose down -v
	@echo "✅ Cleanup complete"

# Local development
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Installation complete"

dev:
	@echo "🔧 Starting development server..."
	python run_local.py

# Testing (placeholder)
test:
	@echo "🧪 Running tests..."
	@echo "⚠️  Tests not yet implemented"

# Quick rebuild and start
rebuild: clean build up

# Full setup
setup: build up
	@echo "✅ Setup complete! Dashboard available at http://localhost:8050"
