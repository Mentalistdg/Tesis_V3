# AWS Deployment Guide - CRONOS Strategy Visualizer

## Infraestructura Actual

| Componente | Detalle |
|------------|---------|
| **IP Publica** | 3.20.234.106 |
| **URL** | http://3.20.234.106 |
| **Instancia EC2** | t3.micro, Ubuntu 24.04 LTS |
| **Instance ID** | i-0bae26943c56844d7 |
| **Region** | us-east-2 (Ohio) |
| **SSH Key** | `~/.ssh/tesis-ec2` (generada localmente, marzo 2026) |

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
ssh -i "$env:USERPROFILE\.ssh\tesis-ec2" ubuntu@3.20.234.106
```

### Para agentes Claude Code (desde bash)

Claude Code en Windows necesita usar `powershell.exe` para capturar output de SSH:

```bash
# Ejecutar comando remoto y capturar output:
powershell.exe -Command "(ssh -i C:\Users\dgonz\.ssh\tesis-ec2 ubuntu@3.20.234.106 'COMANDO') 2>&1 | Out-String"

# IMPORTANTE: ssh directo desde bash NO captura stdout. Siempre usar el wrapper de PowerShell.
```

Ejemplo verificar backend:

```bash
powershell.exe -Command "(ssh -i C:\Users\dgonz\.ssh\tesis-ec2 ubuntu@3.20.234.106 'curl -s http://localhost:8000/') 2>&1 | Out-String"
```

---

## Actualizar la App

### Paso 1: Build del frontend (local)

```powershell
npm run build --prefix "C:\Users\dgonz\PycharmProjects\Tesis_V3\app\frontend"
```

### Paso 2: Subir frontend

```powershell
scp -i "$env:USERPROFILE\.ssh\tesis-ec2" -r "C:\Users\dgonz\PycharmProjects\Tesis_V3\app\frontend\dist" ubuntu@3.20.234.106:/home/ubuntu/
```

### Paso 3: Subir backend (main.py + JSONs)

```powershell
# Solo main.py
scp -i "$env:USERPROFILE\.ssh\tesis-ec2" "C:\Users\dgonz\PycharmProjects\Tesis_V3\app\backend\main.py" ubuntu@3.20.234.106:/home/ubuntu/backend/

# Solo los JSONs de datos
scp -i "$env:USERPROFILE\.ssh\tesis-ec2" C:\Users\dgonz\PycharmProjects\Tesis_V3\app\backend\data\* ubuntu@3.20.234.106:/home/ubuntu/backend/data/
```

### Paso 4: Reiniciar backend y fijar permisos

```powershell
# Fijar permisos frontend
ssh -i "$env:USERPROFILE\.ssh\tesis-ec2" ubuntu@3.20.234.106 "chmod -R 755 /home/ubuntu/dist"

# Matar backend viejo
ssh -i "$env:USERPROFILE\.ssh\tesis-ec2" ubuntu@3.20.234.106 "pkill -f uvicorn"

# Levantar backend nuevo (usar ruta directa al binario, NO source activate)
ssh -i "$env:USERPROFILE\.ssh\tesis-ec2" ubuntu@3.20.234.106 "nohup /home/ubuntu/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --app-dir /home/ubuntu/backend > /home/ubuntu/backend/backend.log 2>&1 &"
```

### Paso 5: Verificar

```powershell
# Backend
ssh -i "$env:USERPROFILE\.ssh\tesis-ec2" ubuntu@3.20.234.106 "curl -s http://localhost:8000/"
# Debe devolver: {"status":"ok","message":"Strategy Visualizer API"}

# Frontend: abrir http://3.20.234.106 en navegador
```

**IMPORTANTE para reiniciar uvicorn:** No usar `source venv/bin/activate` en sesiones SSH no-interactivas — no funciona. Siempre usar la ruta directa: `/home/ubuntu/backend/venv/bin/uvicorn`.

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
5. Iniciar backend:

```powershell
ssh -i "$env:USERPROFILE\.ssh\tesis-ec2" ubuntu@3.20.234.106 "nohup /home/ubuntu/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --app-dir /home/ubuntu/backend > /home/ubuntu/backend/backend.log 2>&1 &"
```

**Nota:** Si no hay Elastic IP asignada, la IP publica cambia al reiniciar la instancia. Verificar la nueva IP en AWS Console -> EC2 -> Instances.

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

### Error "Permission denied (publickey)" al hacer SSH

La key SSH `~/.ssh/tesis-ec2` no esta autorizada en el servidor. Opciones:

1. Usar **EC2 Instance Connect** desde AWS Console (Connect -> pestaña "EC2 Instance Connect", NO "Serial Console") para acceder temporalmente y agregar la key publica a `~/.ssh/authorized_keys`
2. Generar una nueva key con `ssh-keygen -t rsa -b 2048 -f "$env:USERPROFILE\.ssh\tesis-ec2"` y agregar la `.pub` al servidor via Instance Connect

### Error "Connection refused"

El backend no esta corriendo. Iniciarlo:

```bash
nohup /home/ubuntu/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --app-dir /home/ubuntu/backend > /home/ubuntu/backend/backend.log 2>&1 &
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

Si no tienes Elastic IP, la IP cambia al reiniciar. Verificar en AWS Console -> EC2 -> Instances la nueva IP.

---

## Costos Estimados

| Escenario | Costo/mes |
|-----------|-----------|
| Instancia apagada | ~$0.10 (solo disco) |
| Instancia 24/7 | ~$8-10 |
| Free Tier (primer ano) | $0 |

---

## Archivos Importantes Locales

| Archivo | Ubicacion Local |
|---------|-----------------|
| SSH key privada | `C:\Users\dgonz\.ssh\tesis-ec2` |
| SSH key publica | `C:\Users\dgonz\.ssh\tesis-ec2.pub` |
| Frontend source | `C:\Users\dgonz\PycharmProjects\Tesis_V3\app\frontend\` |
| Backend source | `C:\Users\dgonz\PycharmProjects\Tesis_V3\app\backend\` |
| Backend data JSONs | `C:\Users\dgonz\PycharmProjects\Tesis_V3\app\backend\data\` |
| Data update script | `C:\Users\dgonz\PycharmProjects\Tesis_V3\paper\update_backend_data.py` |

---

## Clave Publica de tesis-ec2

Si necesitas autorizar `tesis-ec2` en el servidor (via EC2 Instance Connect):

```bash
echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCuBfB8hq/rh/AoLtDITu2pWUm2BV2Is5FFWErJh0E6F47k2aBwETSr1cj/QNklQi5ciVpn0joJpbWwneLYfic0OtlMJVFW7pL7enJvflf8MnB2hNHa3qIXFJo38ZHAvz8nsNgujnBQd7Gv69l91eXWX/oUEYXeYch8RI5Y5yeB0Ke9YqRCkRb1E52eBnF4oS3N7G2/c1tEjmoLnASnLt62nClfZXaQpIvtC6qpGo0etJPJF5rLcwqcm0D2lyVwySwsVdsHTSRnpuhNreqk7G7aNsNABxHTwiiIiOVL9IbvRUnxPfE26WKvI1KVQNNE8NHLMHXg6EnlJwMqDfgxuxPr dgonz@Mark" >> ~/.ssh/authorized_keys
```

---

**Ultima actualizacion:** Marzo 2026
