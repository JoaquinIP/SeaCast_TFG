# -*- coding: utf-8 -*-
"""Descarga de datos diarios de Copernicus Marine en archivos NetCDF
**directamente en el directorio indicado** (sin crear carpetas extra),
para facilitar climatologías con *xarray*.

Ejemplos de uso
---------------
1. **Ruta de proyecto completa (data/atlantic):**

   ```bash
   python download_data_nc.py \
       --base_path data/atlantic \
       --start_date 1982-01-01 \
       --end_date   2019-12-31
   ```
   Crea: `data/atlantic/raw/reanalysis_nc/YYYYMMDD.nc` y `data/atlantic/static/`.

2. **Ruta ya dentro de *raw/reanalysis_nc*:**

   ```bash
   python download_data_nc.py \
       --base_path data/atlantic/raw/reanalysis_nc \
       --start_date 1982-01-01 \
       --end_date   2019-12-31
   ```
   Crea los ficheros **directamente** en ese directorio y busca la máscara
   en `data/atlantic/static/`.

3. **Especificar la ruta de la máscara manualmente:**

   ```bash
   python download_data_nc.py \
       --base_path data/atlantic/raw/reanalysis_nc \
       --static_path data/atlantic/static \
       --start_date 1982-01-01 \
       --end_date   2019-12-31 \
       --compress
   ```
"""

# ----------------------------------------------------------------------------
# Librerías estándar
# ----------------------------------------------------------------------------
import argparse
import calendar
import os
from datetime import datetime, timedelta
from pathlib import Path

# ----------------------------------------------------------------------------
# Dependencias externas
# ----------------------------------------------------------------------------
import copernicusmarine as cm
import numpy as np
import xarray as xr

# ----------------------------------------------------------------------------
# Módulos propios del repositorio seacast
# ----------------------------------------------------------------------------
from neural_lam import constants
from download_data import load_mask, select  # funciones reutilizadas

# ----------------------------------------------------------------------------
# Función principal de descarga
# ----------------------------------------------------------------------------

def download_data_nc(
    start_date: datetime,
    end_date: datetime,
    datasets: dict[str, list[str]],
    version: str,
    static_path: Path,
    out_dir: Path,
    mask: xr.Dataset,
    *,
    compress: bool = False,
):
    """Descarga datos diarios y los guarda como NetCDF en *out_dir*.

    Parameters
    ----------
    start_date, end_date : datetime
        Intervalo temporal (incluyente).
    datasets : dict
        Map {dataset_id: [variables]}.
    version : str
        Versión del producto Copernicus Marine.
    static_path : Path
        Carpeta donde reside (o se creará) ``bathy_mask.nc``.
    out_dir : Path
        Directorio destino de los archivos NetCDF diarios.
    mask : xr.Dataset
        Máscara batimétrica (0/1) usada por ``select``.
    compress : bool, default *False*
        Activa compresión *zlib* (*complevel* 4) en cada variable.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    current_date = start_date
    while current_date <= end_date:
        first_day = current_date.replace(day=1)
        last_day = current_date.replace(day=calendar.monthrange(current_date.year, current_date.month)[1])

        start_iso = first_day.strftime("%Y-%m-%dT00:00:00")
        end_iso   = last_day.strftime("%Y-%m-%dT00:00:00")

        # Descarga mensual
        monthly = {}
        for ds_id, vars_ in datasets.items():
            ds = cm.open_dataset(
                dataset_id=ds_id,
                dataset_version=version,
                dataset_part="default",
                service="arco-geo-series",
                variables=vars_,
                start_datetime=start_iso,
                end_datetime=end_iso,
                minimum_depth=constants.DEPTHS[0],
                maximum_depth=constants.DEPTHS[-1],
            )
            monthly[ds_id] = select(ds, mask)
            ds.close()

        # Exportación diaria
        for offset in range((last_day - first_day).days + 1):
            day = first_day + timedelta(days=offset)
            fname = out_dir / f"{day:%Y%m%d}.nc"
            if fname.exists():
                continue

            day_dsets = []
            for ds_id, vars_ in datasets.items():
                sel = monthly[ds_id].sel(time=day)
                sel = sel.drop_vars("time")  # elimina dim 0‑D
                if "bottomT" in vars_ and "bottomT" in sel:
                    sel["bottomT"] = sel["bottomT"].isel(depth=0)
                day_dsets.append(sel)

            merged = xr.merge(day_dsets, combine_attrs="override")
            merged = merged.where(mask, drop=False)  # aplica máscara

            enc = {
                v: {"zlib": True, "complevel": 4, "_FillValue": 0.0}
                for v in merged.data_vars
            } if compress else None

            merged.to_netcdf(fname, mode="w", engine="netcdf4", encoding=enc)
            print(f"[✓] Guardado {fname}")

        current_date = last_day + timedelta(days=1)

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Descarga diaria NetCDF para climatologías Copernicus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_path", required=True, help="Ruta base o directorio 'reanalysis_nc'.")
    p.add_argument("--start_date", required=True, help="AAAA-MM-DD")
    p.add_argument("--end_date", required=True, help="AAAA-MM-DD")
    p.add_argument("--compress", action="store_true", help="Activa compresión zlib")
    p.add_argument("--static_path", help="Ruta alternativa para la máscara batimétrica")
    return p.parse_args()


def resolve_paths(base_path: Path, static_path_arg: str | None):
    """Determina *raw_dir* y *static_dir* según dónde apunte *base_path*."""

    base = base_path.expanduser().resolve()

    # 1) Resolución del directorio de salida de los NetCDF
    if (base / "raw" / "reanalysis_nc").exists() or not ("raw" in base.parts):
        # Caso 1: base_path = data/atlantic  → raw/reanalysis_nc dentro
        raw_dir = base / "raw" / "reanalysis_nc"
        project_root = base
    else:
        # Caso 2: base_path = data/atlantic/raw/reanalysis_nc → usar tal cual
        raw_dir = base
        # project_root es dos niveles arriba de 'raw'
        try:
            idx_raw = base.parts.index("raw")
            project_root = Path(*base.parts[:idx_raw])
        except ValueError:
            project_root = base.parent  # fallback

    # 2) Directorio estático
    if static_path_arg:
        static_dir = Path(static_path_arg).expanduser().resolve()
    else:
        static_dir = project_root / "static"

    return raw_dir, static_dir


def main():
    args = parse_args()

    start = datetime.fromisoformat(args.start_date)
    end   = datetime.fromisoformat(args.end_date)

    raw_dir, static_dir = resolve_paths(Path(args.base_path), args.static_path)

    static_dir.mkdir(parents=True, exist_ok=True)

    # Máscara batimétrica (se crea si no existe)
    mask_file = static_dir / "bathy_mask.nc"
    mask = load_mask(mask_file)

    # Dataset de ejemplo (SST Atlántico); cambia según tu caso
    datasets = {
        "cmems-IFREMER-ATL-SST-L4-REP-OBS_FULL_TIME_SERIE": ["analysed_sst"]
    }
    version = "202411"

    download_data_nc(
        start,
        end,
        datasets,
        version,
        static_dir,
        raw_dir,
        mask,
        compress=args.compress,
    )


if __name__ == "__main__":
    main()
