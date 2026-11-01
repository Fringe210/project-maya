# Project Maya — note locali (questo PC)

## Impostazioni da non perdere

- Motore: DeepSeek-V4.1-Flash-1017 **Maya-S**, pack nativo in
  `E:\Maya-data\models\DeepSeek-V4.1-Flash-1017-GGUF\Maya-S\maya-ds4`.
- `E:\project-maya\maya-ds4.json` (ignorato da git) deve contenere:
  - `--dspark on`
  - `--mmap-experts off`
- Su questo PC (128 GB RAM, Windows) `--mmap-experts auto` lascia gli esperti della CPU lane nei file: decode ~13.9 tok/s.
  Con `off` vengono copiati ~92.2 GB in RAM (8383/9393 esperti) e il decode sale a ~33.7 tok/s.
- Non rimuovere `--mmap-experts off` durante update/setup. `maya.py` (`keep_args`) lo preserva, ma verificarne la presenza dopo ogni setup.
- Backup della configurazione: `E:\Maya-data\maya-ds4-config.backup.json`.
  Se un setup riscrive `maya-ds4.json`, ripristinare da qui (o aggiungere di nuovo `--mmap-experts off` negli `args`).
- Build: `cmd /c build-maya.bat` (CUDA 13.2, arch 120, `third_party/llama.cpp` locale).
  `build/MAYA-BUILD.json` può essere stantio: viene aggiornato all'avvio via `maya.py`.
- Ultima verifica: v1.1.43 (`acdab34bd58e`), 2026-10-31.
  320 token generati in 9.5 s di decode (33.7 tok/s), 92.2 GB di esperti copiati in RAM.
