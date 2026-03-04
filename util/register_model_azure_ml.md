# Registrar pesos JointBERT en Azure ML (paso a paso)

Guía completa para registrar el modelo de este repositorio en Azure ML como **Model Asset** (custom model), desde la instalación/login hasta la verificación final.

## 0) Prerrequisitos

- Windows + PowerShell
- Proyecto clonado en `D:\Projects\bc_pibot_jointbert`
- Entorno virtual activo (opcional, recomendado)
- Permisos sobre suscripción/tenant de Azure

Activar entorno virtual (Windows):

```powershell
.venv\Scripts\activate
```

---

## 1) Dependencias Python para Azure ML SDK (opcional si usarás solo CLI)

> No usar `pip install azure` (está deprecado).

```powershell
python -m pip install --upgrade pip
pip install azure-ai-ml azure-identity
```

---

## 2) Instalar Azure CLI

Si `az` no existe:

```powershell
winget install -e --id Microsoft.AzureCLI
```

Validar:

```powershell
where.exe az
az --version
```

Si está instalado pero no se reconoce, validar ruta típica:

```powershell
Test-Path "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd"
```

Agregar al `PATH` (sesión actual + persistente):

```powershell
$cliPath = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin"
$env:Path += ";$cliPath"

$userPath = [Environment]::GetEnvironmentVariable("Path","User")
if ($userPath -notlike "*$cliPath*") {
	[Environment]::SetEnvironmentVariable("Path", "$userPath;$cliPath", "User")
}
```

---

## 3) Login en Azure (evitando error RPC/WAM)

Si `az login` da error tipo *remote procedure call failed*, usar login por código:

```powershell
az config set core.enable_broker_on_windows=false
az logout
az login --use-device-code
```

Listar suscripciones:

```powershell
az account list --output table
```

Seleccionar suscripción (ejemplo real):

```powershell
az account set --subscription "f46df88d-5e5e-4164-8a80-182995d2a7d7"
az account show
```

---

## 4) Preparar `model_package` autocontenido

El modelo debe incluir pesos, tokenizer, código custom y labels (`*.txt`) para decodificar clases en inferencia.

En este repositorio se usan rutas en duro para:

- `model_out`: resultado del entrenamiento del modelo (ejemplo: `pibot_model_v5_beto_crf`)
- `data`: resultado/procesamiento de la task o dataset (ejemplo: `pibimacecv5`)

Es decir, para este caso concreto se empaqueta:

- Modelo: `./model_out/pibot_model_v5_beto_crf`
- Labels/Vocab: `./data/pibimacecv5`

```powershell
# 1) Selección modular de carpetas/archivos
$MODEL_DIR   = ".\model_out\pibot_model_v5_beto_crf"
$LABELS_DIR  = ".\data\pibimacecv5"

# 2) Rutas derivadas
$LABELS_OUT_DIR = Join-Path ".\model_package" "labels"

# 3) Validaciones
if (-not (Test-Path $MODEL_DIR))  { throw "No existe MODEL_DIR: $MODEL_DIR" }
if (-not (Test-Path $LABELS_DIR)) { throw "No existe LABELS_DIR: $LABELS_DIR" }

# 4) Crear estructura de destino
New-Item -ItemType Directory -Force -Path ".\model_package" | Out-Null
New-Item -ItemType Directory -Force -Path $LABELS_OUT_DIR | Out-Null

# 5) Copiar artefactos del modelo entrenado
Copy-Item (Join-Path $MODEL_DIR "*") ".\model_package" -Recurse -Force

# 6) Copiar labels requeridos por el modelo
Copy-Item (Join-Path $LABELS_DIR "*_label.txt") $LABELS_OUT_DIR -Force

# 7) Incluir vocab si aplica
if (Test-Path ".\data\pibimacecv5\vocab.txt") {
	Copy-Item ".\data\pibimacecv5\vocab.txt" ".\model_package" -Force
} else {
	Write-Host "[INFO] vocab.txt no encontrado en .\data\pibimacecv5\vocab.txt (continuando)."
}
```

Para cambiar de modelo/dataset, solo ajusta estas variables:

- `$MODEL_DIR` (subcarpeta dentro de `model_out`, generada por `--model_out`)
- `$LABELS_DIR` (subcarpeta dentro de `data`, asociada a la task/dataset)

Estructura mínima esperada:

```text
model_package/
├─ config.json
├─ model.safetensors
├─ modeling_jointbert.py
├─ module.py
├─ tokenizer.json
├─ tokenizer_config.json
├─ special_tokens_map.json
├─ vocab.txt
└─ labels/
	 ├─ slot_label.txt
	 ├─ activity_label.txt
	 ├─ calc_mode_label.txt
	 ├─ investment_label.txt
	 ├─ region_label.txt
	 └─ req_form_label.txt
```

---

## 5) Configurar Azure ML CLI (Registry vs Workspace)

Instalar/actualizar extensión:

```powershell
az extension add -n ml -y
az extension update -n ml
```

Listar registries (caso recomendado para este proyecto):

```powershell
az ml registry list --output table
```

Ejemplo detectado:

- Registry: `mlr-rag-zsp`

Verificar acceso al registry:

```powershell
az ml registry show --name "mlr-rag-zsp" -o table
```

Si además quieres trabajar contra workspace (opcional), listar workspaces:

```powershell
az ml workspace list --output table
```

Ejemplo detectado:

- Workspace: `mlw-rag-zsp`

Obtener Resource Group de ese workspace:

```powershell
$RG = az resource list `
	--name "mlw-rag-zsp" `
	--resource-type "Microsoft.MachineLearningServices/workspaces" `
	--query "[0].resourceGroup" -o tsv

$RG
```

Configurar defaults de workspace (solo si usarás `--workspace-name`):

```powershell
az configure --defaults group="$RG" workspace="mlw-rag-zsp"
az configure --list-defaults
```

> Importante: `mlr-rag-zsp` es **Registry**, no Workspace. Para registry usa `--registry-name`.

> Importante: no usar placeholders con `< >` en PowerShell. Usa valores reales.

---

## 6) Registrar el modelo (pesos) en Azure ML

### Opción A (recomendada): registrar en Registry `mlr-rag-zsp`

```powershell
az ml model create `
	--registry-name "mlr-rag-zsp" `
	--name "pibert" `
	--version "1" `
	--type "custom_model" `
	--path ".\model_package" `
	--description "JointBERT BETO CRF (PIBERT) con tokenizer y labels"
```
---

## 7) Verificar registro

Para registry:

```powershell
az ml model show --registry-name "mlr-rag-zsp" --name "pibert" --version "1"
az ml model list --registry-name "mlr-rag-zsp" --name "pibert" --output table
```

Para workspace:

```powershell
az ml model show --name "pibert" --version "1"
```

Listar versiones:

```powershell
az ml model list --name "pibert" --output table
```

Si el comando `model create` termina con `Exit Code: 0`, el registro fue exitoso.

---


## 8) Siguiente paso (opcional)

Con el modelo ya registrado, el siguiente paso natural es crear:

1. `Environment` de inferencia
2. `Managed Online Endpoint`
3. `Deployment` (por ejemplo `blue`)

Esto permite exponer el modelo como endpoint HTTPS.

