# AWS Deployment Guide - Strategy Visualizer App

## Infraestructura Actual

| Componente | Detalle |
|------------|---------|
| **IP Publica** | 3.20.234.106 |
| **URL** | http://3.20.234.106 |
| **Instancia EC2** | t3.micro, Ubuntu 24.04 LTS |
| **Region** | us-east-2 (Ohio) |
| **Key Pair** | `tesis-key.pem` (en carpeta Downloads) |

### Arquitectura

```
Usuario --> Nginx (puerto 80) --> /home/ubuntu/dist (frontend estatico)
                              --> proxy /api/* --> Uvicorn (puerto 8000) --> FastAPI
```

### Rutas en el servidor

| Ruta | Contenido |
|------|-----------|
| `/home/ubuntu/dist/` | Frontend compilado (React) |
| `/home/ubuntu/backend/` | Backend FastAPI + datos JSON |
| `/home/ubuntu/backend/venv/` | Virtual environment Python |

---

## Conexion SSH

```powershell
ssh -i "$env:USERPROFILE\Downloads\tesis-key.pem" ubuntu@3.20.234.106
```

---

## Actualizar la App

### Opcion 1: Solo cambios en el Backend (Python/JSON)

Si modificaste `main.py` o los archivos en `app/backend/data/`:

```powershell
# 1. Subir archivos modificados (desde PowerShell local)
scp -i "$env:USERPROFILE\Downloads\tesis-key.pem" -r "E:\PycharmProjects\Tesis_2\Tesis_V3\app\backend" ubuntu@3.20.234.106:/home/ubuntu/

# 2. Conectar al servidor
ssh -i "$env:USERPROFILE\Downloads\tesis-key.pem" ubuntu@3.20.234.106

# 3. Reiniciar backend (en Ubuntu)
pkill -f uvicorn
cd /home/ubuntu/backend && source venv/bin/activate
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
```

### Opcion 2: Cambios en el Frontend (React/TypeScript)

Si modificaste archivos en `app/frontend/src/`:

```powershell
# 1. Recompilar frontend (desde PowerShell local)
cd E:\PycharmProjects\Tesis_2\Tesis_V3\app\frontend
npm run build

# 2. Subir dist al servidor
scp -i "$env:USERPROFILE\Downloads\tesis-key.pem" -r "E:\PycharmProjects\Tesis_2\Tesis_V3\app\frontend\dist" ubuntu@3.20.234.106:/home/ubuntu/

# 3. Conectar y arreglar permisos (en Ubuntu)
ssh -i "$env:USERPROFILE\Downloads\tesis-key.pem" ubuntu@3.20.234.106
chmod -R 755 /home/ubuntu/dist
```

### Opcion 3: Cambios en ambos

Ejecutar ambos procedimientos en orden.

---

## Iniciar/Detener la Instancia

### Apagar (para ahorrar costos)

1. AWS Console -> EC2 -> Instances
2. Seleccionar `tesis-strategy-app`
3. Instance state -> **Stop instance**

### Encender

1. AWS Console -> EC2 -> Instances
2. Seleccionar `tesis-strategy-app`
3. Instance state -> **Start instance**
4. Esperar 1-2 minutos
5. Conectar por SSH e iniciar backend:

```powershell
ssh -i "$env:USERPROFILE\Downloads\tesis-key.pem" ubuntu@3.20.234.106
```

```bash
cd /home/ubuntu/backend && source venv/bin/activate && nohup uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
```

---

## Verificar que todo funciona

### Verificar backend

```bash
curl http://localhost:8000/
# Debe mostrar: {"status":"ok","message":"Strategy Visualizer API"}
```

### Verificar frontend

Abrir en navegador: http://3.20.234.106

### Ver logs del backend

```bash
tail -50 /home/ubuntu/backend/backend.log
```

### Ver logs de nginx

```bash
sudo tail -50 /var/log/nginx/error.log
```

---

## Configuracion Nginx

Archivo: `/etc/nginx/sites-available/tesis`

```nginx
server {
    listen 80;
    server_name _;
    location / {
        root /home/ubuntu/dist;
        index index.html;
        try_files $uri $uri/ /index.html;
    }
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Para reiniciar nginx:

```bash
sudo systemctl restart nginx
```

---

## Security Group (Firewall AWS)

Reglas de entrada configuradas:

| Type | Port | Source |
|------|------|--------|
| SSH | 22 | 0.0.0.0/0 |
| HTTP | 80 | 0.0.0.0/0 |
| Custom TCP | 8000 | 0.0.0.0/0 |

---

## Troubleshooting

### Error "Connection refused"

El backend no esta corriendo. Iniciarlo:

```bash
cd /home/ubuntu/backend && source venv/bin/activate
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
```

### Error 403/404 en frontend

Permisos incorrectos:

```bash
chmod -R 755 /home/ubuntu/dist
chmod 755 /home/ubuntu
```

### Error 500 en nginx

Ver logs:

```bash
sudo tail -50 /var/log/nginx/error.log
```

### La IP cambio

Si no tienes Elastic IP, la IP cambia al reiniciar. Crear una:

1. EC2 -> Elastic IPs -> Allocate
2. Actions -> Associate -> Seleccionar instancia

---

## Costos Estimados

| Escenario | Costo/mes |
|-----------|-----------|
| Instancia apagada | ~$0.10 (solo disco) |
| Instancia 24/7 | ~$8-10 |
| Free Tier (primer ano) | $0 |

---

## Archivos Importantes Locales

| Archivo | Descripcion |
|---------|-------------|
| `app/frontend/src/services/api.ts` | API_BASE debe ser `''` para produccion |
| `app/backend/data/*.json` | Datos pre-computados de modelos |
| `app/backend/main.py` | Endpoints FastAPI |

---

**Ultima actualizacion:** Enero 2026
