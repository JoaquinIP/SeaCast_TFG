# -*- coding: utf-8 -*-
"""Construye **climatología diaria (day‑of‑year)** —por defecto— a partir de
los NetCDF diarios generados con *download_data_nc.py* y la guarda en el
subdirectorio **static/** del mismo proyecto, listo para ser utilizado por
`xarray`.

Opciones disponibles:
* `--freq dayofyear` *(por defecto)*  ➜  366 pasos (1–366)
* `--freq monthly`                   ➜  12  pasos (1–12)
* `--freq seasonal`                  ➜  4   pasos (DJF, MAM, JJA, SON)

Uso mínimo (climatología diaria, salida comprimida en static/):
```bash
python build_climatology.py \
    --input_dir data/atlantic/raw/reanalysis_nc \
    --compress
```
Resultado → `data/atlantic/static/climatology_dayofyear.nc`
"""

# -----------------------------------------------------------------------------
# Librerías estándar
# -----------------------------------------------------------------------------
import argparse
import glob
from pathlib import Path

# -----------------------------------------------------------------------------
# Dependencias externas
# -----------------------------------------------------------------------------
import xarray as xr
import tqdm

# -----------------------------------------------------------------------------
# Utilidades internas
# -----------------------------------------------------------------------------

def guess_static_dir(input_dir: Path) -> Path:
    """Deducción heurística de la carpeta *static* a partir de *input_dir*.

    Si *input_dir* contiene \"raw\", se asume estructura <root>/raw/…  → static
    vive en <root>/static.  Si no, se usa `input_dir.parent/static`.
    """
    input_dir = input_dir.expanduser().resolve()
    if "raw" in input_dir.parts:
        idx = input_dir.parts.index("raw")
        project_root = Path(*input_dir.parts[:idx])
    else:
        project_root = input_dir.parent
    return project_root / "static"


def open_all_daily_nc(directory: Path) -> xr.Dataset:
    """Abre cada NetCDF diario, añade la coordenada **time** usando el nombre
    del fichero (formato YYYYMMDD.nc) y concatena a lo largo de *time*.

    Esto evita el error «Could not find any dimension coordinates …» que
    aparece cuando los archivos se guardaron sin variable de coordenada
    *time* en el paso de descarga.
    """
    import numpy as np

    files = sorted(glob.glob(str(directory / "*.nc")))
    if not files:
        raise FileNotFoundError(f"No se encontraron NetCDF en {directory}")

    datasets = []
    for f in tqdm.tqdm(files, desc="🔍 Cargando diarios"):
        day = Path(f).stem  # 'YYYYMMDD'
        try:
            date_val = np.datetime64(f"{day[:4]}-{day[4:6]}-{day[6:8]}")
        except Exception as exc:
            raise ValueError(f"Nombre de archivo inesperado: {f}") from exc

        ds = xr.open_dataset(f, engine="netcdf4", chunks={})

        # Asegura dim "time" de longitud 1 con la fecha correcta
        if "time" not in ds.dims:
            ds = ds.expand_dims({"time": [date_val]})
        else:
            ds = ds.assign_coords(time=[date_val])  # sobrescribe

        datasets.append(ds)

    # Concatenamos por time
    combined = xr.concat(datasets, dim="time")
    combined = combined.chunk({"time": 365})  # opcional para dask/xarray
    return combined


def build_climatology(ds: xr.Dataset, freq: str) -> xr.Dataset:
    if freq == "dayofyear":
        clim = ds.groupby("time.dayofyear").mean("time", keep_attrs=True)
        return clim.rename({"dayofyear": "time"})
    if freq == "monthly":
        clim = ds.groupby("time.month").mean("time", keep_attrs=True)
        return clim.rename({"month": "time"})
    if freq == "seasonal":
        return ds.groupby("time.season").mean("time", keep_attrs=True)
    raise ValueError("freq debe ser dayofyear, monthly o seasonal")


def save_netcdf(ds: xr.Dataset, path: Path, compress: bool):
    enc = {v: {"zlib": True, "complevel": 4, "_FillValue": 0.0} for v in ds.data_vars} if compress else None
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, engine="netcdf4", encoding=enc)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Genera climatología diaria y la guarda en static/", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input_dir", required=True, help="Directorio con NetCDF diarios (raw/reanalysis_nc)")
    p.add_argument("--freq", choices=["dayofyear", "monthly", "seasonal"], default="dayofyear", help="Tipo de climatología a calcular")
    p.add_argument("--compress", action="store_true", help="Compresión zlib (complevel=4)")
    p.add_argument("--static_dir", help="Ruta al directorio static (si no se puede inferir)")
    p.add_argument("--output_file", help="Nombre/ruta del NetCDF resultante (anula static_dir)")
    return p.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    static_dir = Path(args.static_dir).expanduser().resolve() if args.static_dir else guess_static_dir(input_dir)

    # Ruta de salida
    if args.output_file:
        output_path = Path(args.output_file).expanduser().resolve()
    else:
        output_path = static_dir / f"climatology_{args.freq}.nc"

    print("📂 Leyendo NetCDF diarios …")
    ds = open_all_daily_nc(input_dir)

    print(f"🔄 Calculando climatología {args.freq} …")
    clim = build_climatology(ds, args.freq)
    clim.attrs.update({"history": f"Climatología {args.freq} creada por build_climatology.py"})

    print("💾 Guardando →", output_path)
    save_netcdf(clim, output_path, args.compress)
    print("[✓] Terminado →", output_path)


if __name__ == "__main__":
    main()
